# Benchmark 6: LDA topic inference with MALLET

Infers the topic mix of each of 1.48M congressional tweets using Hemphill et al.'s trained 50-topic LDA model and [MALLET](https://mimno.github.io/Mallet/). Based on the "Command Line Topic Inference" section of `notebooks/models/create/unsupervised/1.0-ams-unigram-lda-model-MALLET.ipynb` in [Hemphill et al.'s repository](https://github.com/casmlab/modeling-political-attention). The Python script runs the notebook's two MALLET commands, `import-file` and `infer-topics`, and times them.

## Data files required

| File | What it is | Where it comes from |
|---|---|---|
| `congress115.csv.gz` | 1,485,834 preprocessed congressional tweets (the default input) | `data/congress_115/` in Hemphill et al.'s repository |
| `russell_processed_0.9.csv.gz` | 41,716 labelled tweets (optional input; see below) | `data/russell/` in Hemphill et al.'s repository |
| `lda_tw_50_unigram_congress115_inferencer.mallet` | Hemphill et al.'s trained LDA inferencer | `models/best/lda/` in Hemphill et al.'s repository |
| `export_vocabulary.bsh` | Helper script used when building the image (part of this folder) | This repository |

The data files must be in this folder when you build the image. The `.csv.gz` files are git-ignored, so they aren't included in a clone of this repository. The image downloads MALLET 2.0.8 itself (and checks it against a fixed checksum).

## Running

    docker build -t ams-lda-inference .
    mkdir -p output
    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/output":/app/output ams-lda-inference

A full run takes about 1.5 minutes with 8 CPUs, and about 3 minutes with 2.

- The `-v` mount keeps the outputs in `./output`. That includes every tweet's topic proportions in `model_output/congress115-topic-composition.txt`, about 1.5 GB. Without the mount, they're deleted with the container.
- `--user` makes the output files belong to you instead of root.

**Quick test (about 20 seconds):** runs on the 41,716-tweet Russell corpus instead.

    docker run --rm --user "$(id -u):$(id -g)" -e INPUT_FILE=russell_processed_0.9.csv.gz ams-lda-inference

Settings, passed with `-e NAME=value`:

| Variable | Default | Meaning |
|---|---|---|
| `INPUT_FILE` | `congress115.csv.gz` | The tweets to infer topics for |
| `MALLET_MEMORY` | 1g | Java memory limit for each MALLET process |

## Limiting CPU cores

The script runs one MALLET process per CPU the container is allowed. To run on fewer, pin the container to specific CPUs, e.g. add `--cpuset-cpus=0-3` to `docker run`. `--cpus=<n>` also works, but it's a time quota, so the work can still spread across all of the host's cores. On machines with hyperthreading, two "CPUs" can share one physical core: `cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list` shows which CPUs pair up.

Memory use grows with the number of CPUs, since each MALLET process loads its own copy of the model. On the default input, allow roughly 550–700 MB per CPU.

## What it measures

For splitting the corpus, importing it into MALLET, inferring topics, and merging the results, the script reports:

- wall-clock time
- CPU time; the MALLET processes count as "workers"
- effective cores used, out of the CPUs available to the container
- peak memory, summed across the MALLET processes

Inference also reports throughput in tweets per second.

## Differences from the original

- **The default input is the public congress115 corpus, not the Russell corpus.** This is pending a check with our collaborators.
  - The top-level README specifies the 41,716-tweet Russell corpus (the same file as Benchmark 3).
  - At that size, each MALLET process's fixed startup cost outweighs the inference itself: starting Java, and loading the model and its 431,046-word vocabulary. Adding cores therefore *lowered* throughput: about 6,200 tweets/s with 2 CPUs, against 4,400 with 8.
  - The congress115 corpus is about 35 times larger, so throughput rises with core count: about 12,800 tweets/s with 2 CPUs, 15,800 with 4, and 29,800 with 8.
  - The model was trained on congress115 tweets, so this is inference on its own training data. That doesn't change the computation.
  - The notebook's own inference input, `russell_preprocessed.csv.gz` (the full Russell dataset), isn't public.
- **The training corpus's import settings are rebuilt from the inferencer.**
  - The notebook imports new tweets with `--use-pipe-from congress115_train.mallet`, so that each word gets the same numeric ID it had during training. That file isn't public.
  - When the image is built, `export_vocabulary.bsh` writes out the inferencer's vocabulary in word-ID order, and MALLET imports it as a single document.
  - The result has exactly the same word IDs as the inferencer: all 431,046 words match.
- **Each input line is split on its commas, not on whitespace.** This is the same fix as in Benchmark 5. MALLET's default would leave out the first two words of each tweet.
- **The work is split across all available cores.**
  - MALLET's `infer-topics` has no option for multiple threads. So the script splits the tweets into one chunk per CPU, imports and infers each chunk in a separate MALLET process at the same time, and merges the results back into the original tweet order.
  - Each chunk gets its own copy of the training import settings, because `import-file` writes newly seen words back into that file. Those new words get IDs the model doesn't know, so inference ignores them, just as in the original single-process run.
- **Output names.** Output files are named after the input file (e.g. `congress115-topic-composition.txt`). Tweets are identified by `id_str`; Russell tweets have no ID column, so they're identified by row number instead.

Inference uses MALLET's default settings (100 sampling iterations, random seed 0), as in the notebook.

## Reference results

On an 8-CPU laptop (4 cores with hyperthreading), default input:

| CPUs | Import | Inference | Effective cores (inference) | Throughput |
|---|---|---|---|---|
| 8 | 23 s | 50 s | 6.6 | 29,800 tweets/s |
| 4 | 31 s | 94 s | 3.95 | 15,800 tweets/s |
| 2 | 58 s | 116 s | 1.99 | 12,800 tweets/s |
