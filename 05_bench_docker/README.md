# Benchmark 5: LDA topic model training with MALLET

Trains a 50-topic LDA topic model on 1.7M preprocessed congressional tweets with [MALLET](https://mimno.github.io/Mallet/). Based on the "Command Line Model Training" section of `notebooks/models/create/unsupervised/1.0-ams-unigram-lda-model-MALLET.ipynb` in [Hemphill et al.'s repository](https://github.com/casmlab/modeling-political-attention). The Python script runs the notebook's two MALLET commands, `import-file` and `train-topics`, and times them.

## Data files required

| File | What it is | Where it comes from |
|---|---|---|
| `congress115_preprocessed.csv.gz` | 1,721,891 preprocessed congressional tweets | Provided by our collaborators (not public) |

The file must be in this folder when you build the image. It's git-ignored, so it isn't included in a clone of this repository. The image downloads MALLET 2.0.8 itself (and checks it against a fixed checksum).

## Running

    docker build -t ams-lda-mallet .
    mkdir -p output
    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/output":/app/output ams-lda-mallet

A full run takes about 26 minutes with 8 CPUs.

- The `-v` mount keeps the trained model and other outputs (about 2 GB, mostly the per-tweet topic proportions) in `./output`. Without it, they're deleted with the container.
- `--user` makes the output files belong to you instead of root.

**Quick test (about 5 seconds):** trains on 20,000 tweets for 50 iterations. Its throughput numbers aren't comparable to a full run.

    docker run --rm --user "$(id -u):$(id -g)" -e NUM_DOCS=20000 -e NUM_ITERATIONS=50 ams-lda-mallet

Settings, passed with `-e NAME=value`:

| Variable | Default | Meaning |
|---|---|---|
| `NUM_DOCS` | all | Train on only the first N tweets |
| `NUM_ITERATIONS` | 1000 | Gibbs sampling iterations (MALLET's default) |
| `NUM_TOPICS` | 50 | Number of topics |
| `MALLET_MEMORY` | 4g | Java memory limit. A full run peaked at about 3.4 GB, so lower values may fail. |

## Limiting CPU cores

MALLET trains with one thread per CPU the container is allowed. To run on fewer, pin the container to specific CPUs, e.g. add `--cpuset-cpus=0-3` to `docker run`. `--cpus=<n>` also works, but it's a time quota, so the work can still spread across all of the host's cores. On machines with hyperthreading, two "CPUs" can share one physical core: `cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list` shows which CPUs pair up.

## What it measures

For decompressing the corpus, importing it into MALLET, and training, the script reports:

- wall-clock time
- CPU time; MALLET's Java process counts as "workers"
- effective cores used, out of the CPUs available to the container
- MALLET's peak memory

Training also reports throughput in token-iterations per second, measured two ways:

- **whole training step:** includes loading the imported corpus and writing the output files, which mostly use one core;
- **Gibbs sampling only:** MALLET's own timer, measured to the whole second.

## Differences from the original

- **Each input line is split on its commas.** MALLET's default splits on whitespace instead. On this file, that made the first chunk of each line (`id,user_id,"['word1',`) the document name, and `'word2',` its label, so the first two words of every tweet were left out of training. It also treated the CSV header as a tweet. Now each line is split into tweet ID, user ID and tokens, and the header is skipped.
- **Training uses every available core** (`--num-threads`). The original ran single-threaded. Multi-threaded LDA gives slightly different, equally valid topics.
- **Two path problems in the original commands are fixed.** `train-topics` now reads the file `import-file` wrote; the notebook pointed it at a different path. And `--output-topic-keys` was given twice; it's now given once (the second value, `congress115_keys_50.txt`, which is the one that took effect).
- **The notebook's preprocessing step doesn't run.** It needs raw tweets that can't be shared, so the benchmark starts from the already-preprocessed corpus.

## Notes on the data

- `congress115_preprocessed.csv.gz` has 1,721,891 rows (8 duplicate tweet IDs). That's more than the 1,485,834 original tweets in the public `congress115.csv.gz`.
- Several topics are dominated by the tokens `nan` and `nanrt`. These look like empty text values converted to the string "nan" during preprocessing, and suggest retweets weren't filtered out. That would explain the extra rows.
- None of this changes the workload, but it matters if anyone interprets the topics.

## Reference results

On an 8-CPU laptop (4 cores with hyperthreading), with all 8 CPUs:

- import: 50 s (1 core)
- training: 1,496 s (5.7 effective cores; peak memory 3.4 GB)
  - Gibbs sampling alone: 1,320 s

Throughput was 11.2M token-iterations/s for the whole training step, or 12.7M for sampling alone (16.8M tokens).
