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

# Enable CPU multithreading optimizations across libraries
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
from hdbscan import HDBSCAN
import xmltodict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process an input file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python main.py input.txt --verbose"
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input file"
    )

    # Benchmarking flags
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation on CPU (default: 64)"
    )

    # Added limit flag (default to None so full dataset runs if omitted)
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process for quick benchmarking (e.g., 500)"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output/bertopic_model.pkl"),
        help="Where to save the fitted model (default: output/bertopic_model.pkl)"
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
    print(f"Loaded {num_docs:,} documents in {t1_data - t0_data:.2f} seconds.")

    # --- Step 2: Initialize Embedding Model ---
    # The model is baked into the Docker image (HF_HUB_OFFLINE=1), so this loads from disk
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Configure SentenceTransformer to use all CPU threads & optimized PyTorch device
    embedding_model.max_seq_length = 256  # Keeps memory footprint lean and processing fast

    # --- Step 3: Explicit CPU Benchmarking of BERTopic ---
    # We pre-compute embeddings using batched CPU encoding to capture pure throughput
    print("Generating embeddings across available CPU cores...")

    def generate_embeddings():
        # Batched CPU encoding in this process, with torch using one thread per
        # available CPU. This replaces encode_multi_process: on CPU its worker
        # pool was several times slower in testing (each worker spends 17-34 s
        # starting up, then encodes more slowly than one multi-threaded process),
        # and by default it starts 4 workers regardless of core count, each with
        # OMP_NUM_THREADS threads, oversubscribing the cores.
        torch.set_num_threads(N_JOBS)
        return embedding_model.encode(
            documents,
            batch_size=args.batch_size
        )

    t0_embed = time.perf_counter()
    embeddings = measure_execution(generate_embeddings, name="Generating Embeddings")
    t1_embed = time.perf_counter()
    embed_time = t1_embed - t0_embed
    embed_throughput = num_docs / embed_time if embed_time > 0 else 0

    # Pass pre-computed embeddings to BERTopic fitting phase
    print("Fitting BERTopic model (Dimensionality Reduction & Clustering)...")

    def fit_model():
        # BERTopic uses pre-calculated embeddings directly. UMAP's parallel loops
        # use NUMBA_NUM_THREADS; HDBSCAN is BERTopic's default configuration plus
        # core_dist_n_jobs, which otherwise defaults to 4 threads.
        hdbscan_model = HDBSCAN(
            min_cluster_size=10,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            core_dist_n_jobs=N_JOBS,
        )
        topic_model = BERTopic(embedding_model=embedding_model, hdbscan_model=hdbscan_model,
                               calculate_probabilities=False)
        topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)
        return topic_model, topics

    t0_fit = time.perf_counter()
    topic_model, topics = measure_execution(fit_model, name="Fitting BERTopic Model")
    t1_fit = time.perf_counter()
    fit_time = t1_fit - t0_fit

    total_pipeline_time = (t1_fit - t0_data)
    total_throughput = num_docs / total_pipeline_time if total_pipeline_time > 0 else 0

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    measure_execution(lambda: joblib.dump(topic_model, args.output), name="Saving Model")
    print(f"Saved model to {args.output} ({len(set(topics)) - (1 if -1 in topics else 0)} topics)")

    # --- Benchmarking Metrics Output ---
    print("\n" + "="*45)
    print("           BENCHMARKING RESULTS           ")
    print("="*45)
    print(f"Total Documents Processed : {num_docs:,}")
    print(f"CPU Cores Utilized        : {N_JOBS}")
    print("-" * 45)
    print(f"Embedding Generation Time : {embed_time:.2f} s")
    print(f"Embedding Throughput      : {embed_throughput:.2f} docs/sec")
    print("-" * 45)
    print(f"BERTopic Fitting Time     : {fit_time:.2f} s")
    print(f"Total Pipeline Execution  : {total_pipeline_time:.2f} s")
    print(f"OVERALL THROUGHPUT        : {total_throughput:.2f} docs/sec")
    print("="*45)
