# Benchmark 8: BERTopic topic inference

Assigns topics to 87,720 comments from the Astronomy Stack Exchange using a trained [BERTopic](https://maartengr.github.io/BERTopic/) model. Two stages:

1. **Embedding:** each comment is turned into a vector with the model's `all-MiniLM-L6-v2` sentence-embedding model.
2. **Topic assignment:** BERTopic maps each vector onto the trained model's reduced space and finds its nearest topic cluster.

Based on `benchset_1/08_infer_bertopic.py` in this repository.

## Data files required

| File | What it is | Where it comes from |
|---|---|---|
| `Comments.xml` | 87,720 Astronomy Stack Exchange comments | `astronomy.stackexchange.com.7z` from the [Internet Archive Stack Exchange dump](https://archive.org/details/stackexchange_20251231) (see the top-level README); extract it with e.g. `7z e astronomy.stackexchange.com.7z Comments.xml` |
| `bertopic_model.pkl` | Trained BERTopic model, including the embedding model | Produced by Benchmark 7: run it with the `-v` output mount (see `07_bench_docker/README.md`), then copy `07_bench_docker/output/bertopic_model.pkl` here |

None of the data files are committed to this repository. Copy each one into this folder before running `docker build`; the build fails if any are missing. Runs don't need internet access.

## Running

    docker build -t ams-bertopic-infer .
    docker run --rm --user "$(id -u):$(id -g)" ams-bertopic-infer

A full run takes about 21 minutes with 8 CPUs.

**Quick test (under 2 minutes):** add `-n 5000` after the image name to use only the first 5,000 comments.

Arguments after the image name are passed to the script:

| Argument | Default | Meaning |
|---|---|---|
| `-n`, `--limit` | all (87,720) | Use only the first N comments |
| `--batch-size` | 64 | Batch size for embedding |
| `-m`, `--model` | `bertopic_model.pkl` | The trained model to use |

To use a model saved by Benchmark 7's container instead of the included one, mount its output folder and point `-m` at it:

    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/../07_bench_docker/output":/app/model07:ro ams-bertopic-infer -m model07/bertopic_model.pkl

## Limiting CPU cores

The script sets every library (PyTorch, OpenMP, MKL, numba) to use one thread per CPU the container is allowed. To run on fewer, pin the container to specific CPUs, e.g. add `--cpuset-cpus=0-3` to `docker run`. `--cpus=<n>` also works, but it's a time quota, so the work can still spread across all of the host's cores. On machines with hyperthreading, two "CPUs" can share one physical core: `cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list` shows which CPUs pair up.

## What it measures

For loading the comments, loading the model, embedding, and assigning topics, the script reports:

- wall-clock time
- CPU time
- effective cores used, out of the CPUs available to the container
- change in memory

A results summary at the end gives embedding, topic-assignment and overall throughput (embedding + assignment) in documents per second.

## Differences from the original

These are the same changes as in Benchmark 7; see its README for the measurements behind the first one.

- **Embedding runs in one multi-threaded process instead of a pool of worker processes.** On CPU, `encode_multi_process`'s pool was 4–5 times slower in testing. Each worker spends 17–34 s starting up, and by default the pool starts 4 workers that each use one thread per CPU, which oversubscribes the cores.
- **Thread counts follow the container's CPU limit.** The original used `os.cpu_count()`, which reports all of the host's CPUs even under `--cpus` or `--cpuset-cpus`. It also set the thread environment variables after PyTorch was imported, which is too late to take effect.
- **UMAP's and HDBSCAN's compiled code is cached in the image**, so timings don't include one-off compilation.
- **Timing hooks** matching the other benchmarks were added around each step. The original results summary is kept.
- **Library versions are pinned to those in the benchset_1 Python environment** (`.venv`), which `bertopic_model.pkl` was saved with.

As in the original, the comments being labelled are the same ones the included model was trained on.

## Reference results

On an 8-CPU laptop (4 cores with hyperthreading), with all 8 CPUs, on all 87,720 comments:

| Step | Time | Effective cores |
|---|---|---|
| Embedding | 1,278 s (69 docs/s) | 7.89 |
| Topic assignment | 15 s | 1.20 |

- Overall inference speed was 67.9 docs/s.
- Embedding is about 99% of the time, so it dominates core-scaling results.
