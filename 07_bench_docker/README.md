# Benchmark 7: BERTopic model training

Trains a [BERTopic](https://maartengr.github.io/BERTopic/) topic model on 87,720 comments from the Astronomy Stack Exchange. The pipeline has two stages:

1. **Embedding:** each comment is turned into a vector with the `all-MiniLM-L6-v2` sentence-embedding model.
2. **Fitting:** BERTopic reduces the vectors' dimensions (UMAP), clusters them (HDBSCAN), and builds a word list for each topic.

Based on `benchset_1/07_fit_bertopic.py` in this repository.

## Data files required

| File | What it is | Where it comes from |
|---|---|---|
| `Comments.xml` | 87,720 Astronomy Stack Exchange comments | `astronomy.stackexchange.com.7z` from the [Internet Archive Stack Exchange dump](https://archive.org/details/stackexchange_20251231) (see the top-level README); extract it with e.g. `7z e astronomy.stackexchange.com.7z Comments.xml` |

None of the data files are committed to this repository. Copy each one into this folder before running `docker build`; the build fails if any are missing. The image also downloads the embedding model while it's being built, so runs don't need internet access.

## Running

    docker build -t ams-bertopic-fit .
    mkdir -p output
    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/output":/app/output ams-bertopic-fit

A full run takes about 20 minutes with 8 CPUs.

- The `-v` mount keeps the fitted model in `./output/bertopic_model.pkl`, which Benchmark 8 can use. Without the mount, it's deleted with the container.
- `--user` makes the output file belong to you instead of root.

**Quick test (under 2 minutes):** add `-n 5000` after the image name to use only the first 5,000 comments.

Arguments after the image name are passed to the script:

| Argument | Default | Meaning |
|---|---|---|
| `-n`, `--limit` | all (87,720) | Use only the first N comments |
| `--batch-size` | 64 | Batch size for embedding |
| `-o`, `--output` | `output/bertopic_model.pkl` | Where to save the fitted model |

## Limiting CPU cores

The script sets every library (PyTorch, OpenMP, MKL, numba, HDBSCAN) to use one thread per CPU the container is allowed. To run on fewer, pin the container to specific CPUs, e.g. add `--cpuset-cpus=0-3` to `docker run`. `--cpus=<n>` also works, but it's a time quota, so the work can still spread across all of the host's cores. On machines with hyperthreading, two "CPUs" can share one physical core: `cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list` shows which CPUs pair up.

## What it measures

For loading the comments, embedding, fitting, and saving the model, the script reports:

- wall-clock time
- CPU time
- effective cores used, out of the CPUs available to the container
- change in memory

A results summary at the end gives embedding throughput and overall throughput (loading + embedding + fitting) in documents per second.

## Differences from the original

- **Embedding runs in one multi-threaded process instead of a pool of worker processes.**
  - The original used sentence-transformers' `encode_multi_process`. On CPU, its pool was 4–5 times slower in every setup we tested.
  - Each worker spent 17–34 s starting up, then embedded more slowly than a single process did.
  - By default the pool starts 4 workers regardless of core count, each with one thread per CPU, which oversubscribes the cores.
  - On 2,000 comments with 8 CPUs, one multi-threaded process managed about 147 docs/s, against 27 docs/s for the original pool.
  - The original script's comment allowed either approach ("encode_multi_process or batched CPU encoding").
- **Thread counts follow the container's CPU limit.** The original used `os.cpu_count()`, which reports all of the host's CPUs even under `--cpus` or `--cpuset-cpus`. It also set the thread environment variables after PyTorch was imported, which is too late to take effect.
- **HDBSCAN uses all available cores.** It otherwise uses 4 threads. Everything else is BERTopic's default HDBSCAN configuration, so results are unchanged.
- **UMAP's and HDBSCAN's compiled code is cached in the image**, so timings don't include one-off compilation.
- **Timing hooks** matching the other benchmarks were added around each step. The original results summary is kept.
- **The model is saved to `output/`** instead of the working directory (configurable with `-o`), so a mount can keep it.
- **Library versions are pinned to those in the benchset_1 Python environment** (`.venv`). `benchset_1/requirements.txt` lists older version ranges that don't match what was actually installed there.

## Reference results

On an 8-CPU laptop (4 cores with hyperthreading), with all 8 CPUs, on all 87,720 comments:

| Step | Time | Effective cores |
|---|---|---|
| Embedding | 1,154 s (76 docs/s) | 7.86 |
| Fitting | 68 s | 4.28 |

- Overall throughput was 71.7 docs/s, and the model had 789 topics.
- Embedding is about 94% of the run, so it dominates core-scaling results.
- Fitting is partly parallel: UMAP and HDBSCAN use multiple cores, but the word lists are built on one core.
- On small runs (e.g. `-n 5000`), fitting averages only about 1.3 cores.
