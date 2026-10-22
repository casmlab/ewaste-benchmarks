# Benchmark 1: Synthetic data generation with an LLM

Creates synthetic versions of social media posts by federal candidates (Dataset 1) with an LLM. For each of 1,000 randomly sampled posts, it runs a small evolutionary search:

1. **Generate:** the LLM (`gemma3:4b`, served by [Ollama](https://ollama.com) inside the container) is asked for 5 paraphrases of each text in the current population, with up to 10 requests in flight at once.
2. **Score:** each paraphrase is scored for meaning kept (cosine similarity of `all-MiniLM-L6-v2` sentence embeddings) and wording changed (share of new words).
3. **Select:** the 10 best paraphrases so far become the next population.

After 3 generations (5 + 25 + 50 = 80 LLM requests per post), the 10 best paraphrases of each post are written out.

Based on `01_synthetic_data.py` in this repository.

## Data files required

| File | What it is | Where it comes from |
|---|---|---|
| `posts.csv` | 3,321,323 posts by 1,758 federal candidates across 10 platforms (3.6 GB) | Dataset 1, [ICPSR 300490](https://doi.org/10.3886/ICPSR300490.V2) (see the top-level README) |

None of the data files are committed to this repository. Copy `posts.csv` into this folder before running `docker build`; the build fails if it's missing. Only the `all_text` column is used.

The image downloads Ollama, the LLM (3.2 GB) and the embedding model while it's being built, so runs don't need internet access. With `posts.csv`, the image is about 9 GB, and building it needs about as much free disk space again, since Docker keeps a copy of the build files.

## Running

    docker build -t ams-synthetic-data .
    mkdir -p output
    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/output":/app/output ams-synthetic-data

**A full run is very long on CPU: roughly 30 hours with 8 CPUs** (estimated from short runs, at about 1.4 s per LLM request; 80,000 requests in all). It needs about 9 GB of memory.

- The `-v` mount keeps the synthetic texts in `./output/text_outputs_ollama.json` (one JSON record per line). The file is appended to, not overwritten. Without the mount, it's deleted with the container.
- `--user` makes the output file belong to you instead of root.

**Quick test (under a minute):** add `-n 2 --generations 1` after the image name: 2 posts, 10 LLM requests.

Arguments after the image name are passed to the script:

| Argument | Default | Meaning |
|---|---|---|
| `-n`, `--num-samples` | 1000 | Number of posts to paraphrase |
| `--generations` | 3 | Evolutionary generations per post |
| `--concurrency` | 10 | LLM requests in flight at once |
| `--num-thread` | physical cores available | Threads the LLM uses |
| `--model` | `gemma3:4b` | Ollama model; it must be built into the image (see below) |
| `-o`, `--output` | `output/text_outputs_ollama.json` | Where to append the synthetic texts |

To use a different LLM, build the image with e.g. `--build-arg MODEL=llama3.2:3b` and run it with `--model llama3.2:3b`.

## Limiting CPU cores

Pin the container to specific CPUs, e.g. add `--cpuset-cpus=0-3` to `docker run`. `--cpus=<n>` also works, but it's a time quota, so the work can still spread across all of the host's cores.

The LLM uses **one thread per physical core** among the CPUs the container is allowed, not one per CPU. With hyperthreading, two "CPUs" share a physical core, and on an 8-CPU laptop (4 cores) 4 threads generated about 28% faster than 8. As a result, "effective cores" for the generation step tops out near the number of physical cores, e.g. about 4 / 8 on that laptop. `--num-thread` overrides this. `cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list` shows which CPUs pair up.

## What it measures

For starting Ollama and loading the LLM, loading the posts, and generating the synthetic corpus, the script reports:

- wall-clock time
- CPU time, split into the script itself and the Ollama server, where the LLM runs
- effective cores used, out of the CPUs available to the container
- change in memory, for the script and the Ollama server together

The original hardware benchmark summary follows:

- requests completed and failed
- tokens generated and prompt tokens processed
- **generation speed:** tokens generated ÷ the sum of each request's generation time. Requests run in parallel, so this is the average speed of one request, not the total.
- **system throughput:** tokens generated per second of wall-clock time, across all requests. This is the main figure.
- posts per second

## Differences from the original

- **Ollama runs inside the container.** The script starts `ollama serve` itself and loads the model before the timed generation step, so timings don't include start-up or model loading. The original expected a server already running on `localhost:11434`.
- **The server processes requests in parallel.** `OLLAMA_NUM_PARALLEL` is set to `--concurrency` (10) unless already set, so the original's 10 concurrent requests are all processed at once. Ollama otherwise picks a limit itself, which can be lower, leaving requests queued.
- **The LLM's thread count follows the container's CPU limit** (one per physical core, as above). Ollama otherwise counts the host's physical cores, ignoring `--cpuset-cpus`.
- **Only the `all_text` column is read.** This selects the same 1,000 posts as reading the whole file, using about 1 GB of memory instead of several times the 3.6 GB file's size.
- **New options:** `-n` and `--generations` for quick tests, `--num-thread`, and `-o`. The defaults reproduce the original run.
- **Failed requests are counted** in the summary. The original printed each failure but counted it as completed. A failed request still yields an empty paraphrase, as before.
- **No overall request timeout.** On CPU, a queued request can wait longer than aiohttp's 5-minute default.
- **Timing hooks** matching the other benchmarks were added around each step. The original summary is kept, with "Generation Speed (GPU)" relabelled "(Model)", since it runs on CPU here.
- **The output is written to `output/`** instead of the working directory, so a mount can keep it.
- **The LLM is Google's Gemma 3 (4B)**, the original script's default. The top-level README describes this benchmark as using Llama models.

## Reference results

On an 8-CPU laptop (4 cores with hyperthreading), with all 8 CPUs, on 1 post with 2 generations (30 requests):

| Step | Time | Effective cores |
|---|---|---|
| Starting Ollama & loading the model | 6 s | 1.3 |
| Loading posts | 14 s | 1.1 |
| Generating | 42 s | 4.1 |

- System throughput was 11.3 tokens/s, or about 1.4 s per request.
- Loading the model takes about 7.6 GB of memory: 3.2 GB of weights plus working memory for 10 parallel requests.
- The scoring step (embeddings) is about 1% of the CPU time; nearly all of it is the LLM.
