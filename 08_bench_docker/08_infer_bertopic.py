'''
MB: This is the sample code from the BERTopic website with some
placeholder code for loading StackOverflow data. This currently
does both fitting and prediction, but we can split that into separate tasks.
'''
import os

from joblib import cpu_count

## Use every CPU available to the container (respects docker --cpus / --cpuset-cpus;
## os.cpu_count() reports all of the host's CPUs). Thread settings must be in the
## environment before torch, numpy and numba are imported to take effect.
N_JOBS = cpu_count()

# Enable CPU multithreading optimizations
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMBA_NUM_THREADS"] = str(N_JOBS)  # UMAP's parallel loops

import argparse
import joblib
import time
from pathlib import Path

import psutil
import torch
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import xmltodict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process an input file for BERTopic inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python predict.py input.txt --limit 1000 --model bertopic_model.pkl"
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input file"
    )

    parser.add_argument(
        "-m", "--model",
        type=Path,
        default=Path("bertopic_model.pkl"),
        help="Path to the trained BERTopic joblib model file (default: bertopic_model.pkl)"
    )

    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process for quick benchmarking (e.g., 500)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation on CPU (default: 64)"
    )

    return parser.parse_args()

# ---------------------------------------------------------
# Measurement Hooks
# ---------------------------------------------------------

def cpu_seconds():
    """
    Returns (main process CPU seconds, worker process CPU seconds).

    Any work done in separate processes is counted as worker time: the sum of
    live descendant processes plus children that have already exited and been
    reaped (reported by os.times()). The pipeline currently runs in threads of
    the main process, so worker time should stay at zero.
    """
    proc = psutil.Process()
    t = proc.cpu_times()
    main = t.user + t.system

    reaped = os.times()
    workers = reaped.children_user + reaped.children_system
    for child in proc.children(recursive=True):
        try:
            ct = child.cpu_times()
            workers += ct.user + ct.system + ct.children_user + ct.children_system
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return main, workers

def measure_execution(func, name="Task"):
    """
    Measurement hook wrapper to log execution timing and system memory utilization.
    """
    print(f"\n--- [START] {name} ---")
    n_cpus = cpu_count()
    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)

    start_wall = time.perf_counter()
    start_main, start_workers = cpu_seconds()

    result = func()

    end_wall = time.perf_counter()
    end_main, end_workers = cpu_seconds()
    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)

    wall_elapsed = end_wall - start_wall
    main_elapsed = end_main - start_main
    workers_elapsed = end_workers - start_workers
    cpu_elapsed = main_elapsed + workers_elapsed

    print(f"--- [END] {name} ---")
    print(f"Wall-clock time : {wall_elapsed:.4f} seconds")
    print(f"CPU time        : {cpu_elapsed:.4f} seconds "
          f"(main {main_elapsed:.4f} + workers {workers_elapsed:.4f})")
    print(f"Effective Cores : {cpu_elapsed / max(wall_elapsed, 1e-6):.2f} / {n_cpus}")
    print(f"RAM Used Delta  : {mem_after - mem_before:.2f} MB\n")

    return result

if __name__ == '__main__':
    args = parse_args()
    print(f"Running with {N_JOBS} CPUs")

    # --- Step 1: Data Ingestion & Parsing ---
    def load_documents():
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = f.read()
        xml_dict = xmltodict.parse(data)
        comments_list = xml_dict['comments']['row']

        # Handle single element vs list output from xmltodict
        if isinstance(comments_list, dict):
            comments_list = [comments_list]

        documents = [c['@Text'] for c in comments_list if '@Text' in c]

        # --- Slice to subset if --limit is specified ---
        if args.limit and args.limit > 0:
            documents = documents[:args.limit]
        return documents

    t0_data = time.perf_counter()
    documents = measure_execution(load_documents, name="Loading Documents")
    t1_data = time.perf_counter()
    num_docs = len(documents)
    print(f"Loaded {num_docs:,} documents for inference in {t1_data - t0_data:.2f} seconds.")

    # --- Step 2: Load Saved Model ---
    t0_model = time.perf_counter()
    print(f"Loading trained model from {args.model}...")
    topic_model = measure_execution(lambda: joblib.load(args.model), name="Loading Model")
    t1_model = time.perf_counter()
    print(f"Model loaded in {t1_model - t0_model:.2f} seconds.")

    # --- Step 3: Fast Parallelized Embedding Generation ---
    print("Generating embeddings across available CPU cores...")

    # Extract underlying SentenceTransformer instance from BERTopic wrapper wrapper
    if hasattr(topic_model.embedding_model, "embedding_model"):
        st_model = topic_model.embedding_model.embedding_model
    else:
        st_model = topic_model.embedding_model

    def generate_embeddings():
        # Batched CPU encoding in this process, with torch using one thread per
        # available CPU. This replaces encode_multi_process: on CPU its worker
        # pool was several times slower in testing (each worker spends 17-34 s
        # starting up, then encodes more slowly than one multi-threaded process),
        # and by default it starts 4 workers regardless of core count, each with
        # OMP_NUM_THREADS threads, oversubscribing the cores.
        torch.set_num_threads(N_JOBS)
        return st_model.encode(
            documents,
            batch_size=args.batch_size
        )

    t0_embed = time.perf_counter()
    embeddings = measure_execution(generate_embeddings, name="Generating Embeddings")
    t1_embed = time.perf_counter()
    embed_time = t1_embed - t0_embed
    embed_throughput = num_docs / embed_time if embed_time > 0 else 0

    # --- Step 4: Perform Topic Inference/Transform ---
    print("Assigning topics to new documents...")
    t0_predict = time.perf_counter()

    # Pass pre-computed embeddings directly into transform()
    topics, probs = measure_execution(
        lambda: topic_model.transform(documents, embeddings=embeddings),
        name="Assigning Topics"
    )

    t1_predict = time.perf_counter()
    predict_time = t1_predict - t0_predict
    predict_throughput = num_docs / predict_time if predict_time > 0 else 0

    total_inference_time = embed_time + predict_time
    total_throughput = num_docs / total_inference_time if total_inference_time > 0 else 0

    # --- Benchmarking Metrics Output ---
    print("\n" + "="*45)
    print("         INFERENCE BENCHMARK RESULTS       ")
    print("="*45)
    print(f"Total Documents Inferred  : {num_docs:,}")
    print(f"CPU Cores Utilized        : {N_JOBS}")
    print("-" * 45)
    print(f"Embedding Generation Time : {embed_time:.2f} s")
    print(f"Embedding Throughput      : {embed_throughput:.2f} docs/sec")
    print("-" * 45)
    print(f"Transform/Prediction Time : {predict_time:.2f} s")
    print(f"Transform Throughput      : {predict_throughput:.2f} docs/sec")
    print("-" * 45)
    print(f"Total Prediction Time     : {total_inference_time:.2f} s")
    print(f"OVERALL INFERENCE SPEED   : {total_throughput:.2f} docs/sec")
    print("="*45)
