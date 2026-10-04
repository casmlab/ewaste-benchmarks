# Dockerized benchmarks

Benchmarks 3–8 (described in the top-level `README.md`) each run in their own Docker container. Each one has a folder, `NN_bench_docker/`, holding:

- the benchmark script
- a Dockerfile that installs the right Python version and libraries
- a README covering what the benchmark does, which data files it needs and where they come from, how to run it, and how it differs from the original code

Benchmarks 1 and 2 aren't Dockerized yet; we're waiting on their data.

## Requirements

- **Docker**, usable by your user (`docker info` should work without `sudo`).
- **Each benchmark's data files**, copied into its folder before building. Most are git-ignored, so they won't be in a clone of this repository. Each benchmark's README lists them.
- **Internet access while building the images.** Running the benchmarks doesn't need it.

## The benchmarks

| Benchmark | Folder | Task | Full run* | Quick test |
|---|---|---|---|---|
| 3 | `03_bench_docker/` | Logistic regression training | ~1.5 min | — (already short) |
| 4 | `04_bench_docker/` | Logistic regression inference | ~1.5 min | — (already short) |
| 5 | `05_bench_docker/` | LDA training (MALLET) | ~26 min | `-e NUM_DOCS=20000 -e NUM_ITERATIONS=50` |
| 6 | `06_bench_docker/` | LDA inference (MALLET) | ~1.5 min | `-e INPUT_FILE=russell_processed_0.9.csv.gz` |
| 7 | `07_bench_docker/` | BERTopic training | ~20 min | `-n 5000` (after the image name) |
| 8 | `08_bench_docker/` | BERTopic inference | ~21 min | `-n 5000` (after the image name) |

\* With 8 CPUs on a laptop (4 cores with hyperthreading). Expect longer on fewer or slower cores.

Benchmark 6 currently runs on a different dataset from the one the top-level README specifies, pending a check with our collaborators. See `06_bench_docker/README.md`.

## Checking that everything works

`test_functionality.sh` builds each benchmark's image and runs it once, one after another. It uses the quick-test options for the long benchmarks, and checks that each finishes cleanly and prints its results:

    ./test_functionality.sh            # all benchmarks
    ./test_functionality.sh 05 07      # only the listed benchmarks

It prints PASS or FAIL for each benchmark and where its log is saved. Once the images are built, it takes about 6–8 minutes. The first build downloads several GB. The timings from this test aren't meaningful; it only checks that everything runs.

## Running with fewer CPU cores

Every benchmark uses all the CPUs its container is allowed, and reports "effective cores" out of that number. To measure performance on fewer cores, add `--cpuset-cpus` to `docker run`, which pins the container to specific CPUs:

    docker run --rm --cpuset-cpus=0-3 ams-create-best-classifiers

Two things to watch for:

- **`--cpus=<n>` isn't the same thing.** It also works, but it's a CPU-time quota: the work can still spread across all of the host's cores. For core-scaling runs, `--cpuset-cpus` is usually the better choice.
- **Hyperthreading.** On machines with hyperthreading, two "CPUs" share one physical core and don't add a full core's throughput. `lscpu` shows whether it's on, and `cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list` shows which CPUs pair up. Choose the `--cpuset-cpus` list accordingly.

## What every benchmark reports

Each timed step prints:

- wall-clock time
- CPU time, split into the main process and any parallel worker processes
- effective cores used (CPU time ÷ wall time, out of the CPUs available to the container)
- memory change

Each benchmark also reports a throughput figure suited to its task, e.g. tweets per second or documents per second. Its README explains exactly what that figure covers.
