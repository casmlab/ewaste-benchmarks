import os

## CPUs available to the container (respects docker --cpuset-cpus; os.cpu_count()
## reports all of the host's CPUs). Thread settings must be in the environment
## before torch is imported to take effect.
N_CPUS = len(os.sched_getaffinity(0))
os.environ.setdefault("OMP_NUM_THREADS", str(N_CPUS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_CPUS))

import argparse
import asyncio
import json
import math
import subprocess
from pathlib import Path
import time
import aiohttp
import pandas as pd
import psutil
import requests
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

OLLAMA_URL = "http://" + os.environ.get("OLLAMA_HOST", "localhost:11434")


def physical_cores():
    """
    Physical cores among the CPUs available to the container. With
    hyperthreading, two CPUs share a core; the LLM runs fastest with one
    thread per core.
    """
    cores = set()
    for cpu in os.sched_getaffinity(0):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            cores.add(((topology / "physical_package_id").read_text().strip(),
                       (topology / "core_id").read_text().strip()))
        except OSError:
            return N_CPUS
    return len(cores) or N_CPUS


NUM_THREAD = physical_cores()  # overridden by --num-thread

device = "cuda" if torch.cuda.is_available() else "cpu"
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# --- Hardware Benchmarking Tracker ---
class BenchmarkMetrics:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.total_eval_tokens = 0
        self.total_prompt_tokens = 0
        self.total_eval_duration_ns = 0
        self.completed_requests = 0
        self.failed_requests = 0

    async def log_request(self, res_json):
        async with self.lock:
            self.completed_requests += 1
            # Ollama returns timing metrics in nanoseconds
            self.total_eval_tokens += res_json.get("eval_count", 0)
            self.total_prompt_tokens += res_json.get("prompt_eval_count", 0)
            self.total_eval_duration_ns += res_json.get("eval_duration", 0)

    async def log_failure(self):
        async with self.lock:
            self.failed_requests += 1

    def report(self, elapsed_wall_time):
        total_eval_sec = self.total_eval_duration_ns / 1e9
        gen_tps = self.total_eval_tokens / total_eval_sec if total_eval_sec > 0 else 0
        system_tps = self.total_eval_tokens / elapsed_wall_time if elapsed_wall_time > 0 else 0

        print("\n" + "="*50)
        print("          HARDWARE BENCHMARK SUMMARY          ")
        print("="*50)
        print(f"Total Requests Completed : {self.completed_requests}")
        print(f"Failed Requests         : {self.failed_requests}")
        print(f"Total Generated Tokens  : {self.total_eval_tokens}")
        print(f"Total Prompt Tokens     : {self.total_prompt_tokens}")
        print(f"Wall Clock Execution    : {elapsed_wall_time:.2f} s")
        print(f"Generation Speed (Model): {gen_tps:.2f} tokens/sec")
        print(f"System Throughput (Wall): {system_tps:.2f} tokens/sec")
        print("="*50 + "\n")

metrics = BenchmarkMetrics()


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel Ollama Throughput Benchmark")
    parser.add_argument("input_file", type=Path, help="Path to input CSV")
    parser.add_argument("--concurrency", type=int, default=10, help="Max parallel API calls")
    parser.add_argument("--model", type=str, default="gemma3:4b", help="Ollama model tag")
    parser.add_argument("-n", "--num-samples", type=int, default=1000,
                        help="Number of posts to paraphrase (default: 1000)")
    parser.add_argument("--generations", type=int, default=3,
                        help="Evolutionary generations per post (default: 3)")
    parser.add_argument("--num-thread", type=int, default=NUM_THREAD,
                        help="Threads the LLM uses (default: physical cores available to the container)")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/text_outputs_ollama.json"),
                        help="Where to append the synthetic texts (default: output/text_outputs_ollama.json)")
    return parser.parse_args()


# ---------------------------------------------------------
# Measurement Hooks
# ---------------------------------------------------------

def cpu_seconds():
    """
    Returns (main process CPU seconds, worker process CPU seconds).

    The LLM runs in the Ollama server, a child process of this script, so its
    CPU time is counted as worker time: the sum of live descendant processes
    plus children that have already exited and been reaped (os.times()).
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

def memory_mb():
    """Resident memory of this process plus its descendants (the Ollama server), in MB."""
    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return total / (1024 * 1024)

def measure_execution(func, name="Task"):
    """
    Measurement hook wrapper to log execution timing and system memory utilization.
    """
    print(f"\n--- [START] {name} ---")
    mem_before = memory_mb()

    start_wall = time.perf_counter()
    start_main, start_workers = cpu_seconds()

    result = func()

    end_wall = time.perf_counter()
    end_main, end_workers = cpu_seconds()
    mem_after = memory_mb()

    wall_elapsed = end_wall - start_wall
    main_elapsed = end_main - start_main
    workers_elapsed = end_workers - start_workers
    cpu_elapsed = main_elapsed + workers_elapsed

    print(f"--- [END] {name} ---")
    print(f"Wall-clock time : {wall_elapsed:.4f} seconds")
    print(f"CPU time        : {cpu_elapsed:.4f} seconds "
          f"(main {main_elapsed:.4f} + Ollama {workers_elapsed:.4f})")
    print(f"Effective Cores : {cpu_elapsed / max(wall_elapsed, 1e-6):.2f} / {N_CPUS}")
    print(f"RAM Used Delta  : {mem_after - mem_before:.2f} MB\n")

    return result


# --- Ollama Server ---
def ollama_options(num_thread):
    # Ollama otherwise picks a thread count from the host's physical cores,
    # ignoring the container's CPU limit
    return {"num_thread": num_thread}


def start_ollama(model, concurrency, num_thread):
    """
    Starts `ollama serve` as a child process and loads the model into memory,
    so the generation timing doesn't include server start-up or model loading.
    The server handles up to `concurrency` requests at once unless
    OLLAMA_NUM_PARALLEL is already set.
    """
    env = dict(os.environ)
    env.setdefault("OLLAMA_NUM_PARALLEL", str(concurrency))
    server = subprocess.Popen(["ollama", "serve"], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(120):
        if server.poll() is not None:
            raise RuntimeError("ollama serve exited during start-up")
        try:
            requests.get(OLLAMA_URL, timeout=1)
            break
        except requests.ConnectionError:
            time.sleep(0.5)
    else:
        raise RuntimeError("ollama serve didn't start within 60 seconds")

    # A request with no prompt loads the model and returns immediately. It uses
    # the same options as the benchmark's requests; different ones would make
    # Ollama reload the model.
    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": model, "keep_alive": -1,
                            "options": ollama_options(num_thread)})
    if r.status_code != 200:
        raise RuntimeError(f"Couldn't load model {model}: {r.text}")
    return server


# --- Async Ollama Generation ---
async def generate_single_ollama(session, semaphore, prompt, model="gemma3:4b"):
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # Return aggregated JSON with server-side metrics
        "keep_alive": -1,
        "options": ollama_options(NUM_THREAD),
    }

    async with semaphore:
        try:
            async with session.post(url, json=payload) as r:
                res = await r.json()
                if "error" in res:
                    raise RuntimeError(res["error"])
                await metrics.log_request(res)
                return res.get("response", "")
        except Exception as e:
            await metrics.log_failure()
            print(f"Request failed: {e}")
            return ""


async def generate_text_ollama_batched(session, semaphore, prompt, model="gemma3:4b", num_return_sequences=3):
    """Fires requests concurrently up to the semaphore concurrency limit."""
    tasks = [
        generate_single_ollama(session, semaphore, prompt, model)
        for _ in range(num_return_sequences)
    ]
    return await asyncio.gather(*tasks)


# --- Fitness Calculation ---
def fitness_score(original, candidate):
    if len(candidate.strip()) == 0 or candidate.strip() == ".":
        return 0.0

    embeddings = sbert.encode([original, candidate], convert_to_tensor=True)
    sim = util.cos_sim(embeddings[0], embeddings[1]).item()

    orig_words = set(original.lower().split())
    cand_words = set(candidate.lower().split())
    overlap_ratio = len(orig_words & cand_words) / len(cand_words) if cand_words else 1.0

    diversity = 1.0 - overlap_ratio
    return 2 * sim + diversity


# --- Async Evolutionary Corpus Generation ---
async def evolve_corpus_async(
    session,
    semaphore,
    text,
    target_size=10,
    offspring_per_sample=5,
    retain_top=10,
    generations=3,
    model="gemma3:4b",
    prompt="Paraphrase this:"
):
    population = [text]
    all_candidates = []

    for gen in range(generations):
        # Dispatch generation for all parents concurrently
        generation_tasks = [
            generate_text_ollama_batched(
                session, semaphore, f"{prompt} {parent}", model, offspring_per_sample
            )
            for parent in population
        ]
        results = await asyncio.gather(*generation_tasks)

        gen_candidates = []
        for parent, candidates in zip(population, results):
            for c in candidates:
                score = fitness_score(text, c)
                # Store candidate along with metadata; preserving original root text reference
                gen_candidates.append((c, score, gen + 1, text, parent))

        all_candidates.extend(gen_candidates)
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        survivors = all_candidates[:retain_top]
        population = [c[0] for c in survivors]

    all_candidates.sort(key=lambda x: x[1], reverse=True)
    return all_candidates[:target_size]


# --- Main Benchmarking Loop ---
async def run_corpus(args, seed_corpus):
    semaphore = asyncio.Semaphore(args.concurrency)
    prompt_prefix = (
        "Paraphrase this text. Do not include options or extra text. "
        "The text is: "
    )

    # Reuse TCP connection pool via aiohttp ClientSession. No overall timeout:
    # on CPU a queued request can wait many minutes.
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        with open(args.output, "a") as out_file:
            for root_text in tqdm(seed_corpus, desc="Benchmarking Corpus"):
                synthetic_text = await evolve_corpus_async(
                    session=session,
                    semaphore=semaphore,
                    text=root_text,
                    target_size=10,
                    offspring_per_sample=5,
                    retain_top=10,
                    generations=args.generations,  # 3 generations for benchmarking
                    model=args.model,
                    prompt=prompt_prefix,
                )

                # Persist output safely
                for syn, score, gen, orig_text, parent in synthetic_text:
                    record = {
                        "syn": syn,
                        "score": score,
                        "gen": gen,
                        "orig_text": orig_text,
                        "parent": parent,
                        "model": args.model,
                    }
                    out_file.write(json.dumps(record) + "\n")


def main():
    global NUM_THREAD
    args = parse_args()
    NUM_THREAD = args.num_thread
    print(f"Running with {N_CPUS} CPUs, {NUM_THREAD} LLM threads")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    server = measure_execution(lambda: start_ollama(args.model, args.concurrency, NUM_THREAD),
                               name="Starting Ollama & Loading Model")
    try:
        def load_posts():
            # Only all_text is used; reading just that column selects the same
            # rows as reading the whole file, with far less memory
            df = pd.read_csv(args.input_file, usecols=["all_text"])
            # Adjusted sample size for realistic benchmarking runs
            return df.sample(min(args.num_samples, len(df)), random_state=330)["all_text"].tolist()

        seed_corpus = measure_execution(load_posts, name="Loading Posts")
        print(f"Paraphrasing {len(seed_corpus):,} posts with {args.model}, "
              f"{args.generations} generations, concurrency {args.concurrency}")

        start_wall_time = time.perf_counter()
        measure_execution(lambda: asyncio.run(run_corpus(args, seed_corpus)),
                          name="Generating Synthetic Corpus")
        elapsed_wall_time = time.perf_counter() - start_wall_time
        metrics.report(elapsed_wall_time)
        print(f"Posts per second        : {len(seed_corpus) / elapsed_wall_time:.4f}")
        print(f"Wrote synthetic texts to {args.output}")
    finally:
        server.terminate()
        server.wait()


if __name__ == "__main__":
    main()
