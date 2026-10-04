# Benchmark 4: Logistic regression inference

Labels 1,485,834 tweets from the 115th U.S. Congress with policy topics, using the logistic regression model trained by Hemphill et al. (`lr_0.9.pkl`, the model Benchmark 3 trains). Based on the "Label Data with Supervised Model Predictions" section of `notebooks/models/run/1.0-ams-label-comparisons.ipynb` in [Hemphill et al.'s repository](https://github.com/casmlab/modeling-political-attention).

## Data files required

| File | What it is | Where it comes from |
|---|---|---|
| `congress115.csv.gz` | 1,485,834 preprocessed congressional tweets | `data/congress_115/` in Hemphill et al.'s repository |
| `lr_0.9.pkl` | Trained logistic regression model | `models/best/lr/` in Hemphill et al.'s repository |

Both files must be in this folder when you build the image. `congress115.csv.gz` is git-ignored, so it isn't included in a clone of this repository.

## Running

    docker build -t ams-label-comparisons .
    docker run --rm ams-label-comparisons

A full run takes about 1–1.5 minutes with 8 CPUs. There's no shorter test mode; this is already short.

## Limiting CPU cores

The benchmark uses every CPU the container is allowed. To run on fewer, pin the container to specific CPUs, e.g. `docker run --rm --cpuset-cpus=0-3 ams-label-comparisons`. `--cpus=<n>` also works, but it's a time quota, so the work can still spread across all of the host's cores. On machines with hyperthreading, two "CPUs" can share one physical core: `cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list` shows which CPUs pair up.

## What it measures

For loading the tweets and for predicting their labels, the script reports:

- wall-clock time
- CPU time, split into the main process and the parallel worker processes
- effective cores used, out of the CPUs available to the container
- change in memory

Prediction also reports throughput in tweets per second.

## Differences from the original

- **Only the supervised-labelling section runs.** The notebook first merges in topic labels from an unsupervised (LDA) model. That needs the full private congress115 dataset and a hand-built topic-to-policy mapping (`congress115_keys_50.txt`), and neither is public.
- **The model predicts on `preprocessed_text`** (each tweet's token list). The notebook uses `text_tokenized` from the private dataset; the public file has no raw tweet text. The model's tokenizer turns the token list back into the same tokens.
- **Prediction runs in parallel on all available cores.** The tweets are split into chunks (4 per CPU), and each worker process loads the model once. The original ran in a single process. The predicted labels are identical either way.
- **Version mismatch warning.** `lr_0.9.pkl` was saved with scikit-learn 0.19.1, and the image uses 0.20.4. scikit-learn warns about this when loading the model; predictions are unaffected.

## Reference results

On an 8-CPU laptop (4 cores with hyperthreading), with all 8 CPUs: prediction took 59–72 s (7.4–7.6 effective cores, 20,500–25,100 tweets/s). With `--cpus=4`, it took 86 s (17,300 tweets/s).
