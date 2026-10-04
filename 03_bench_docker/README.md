# Benchmark 3: Logistic regression training

Trains Hemphill et al.'s supervised policy-topic classifier: a bag-of-words logistic regression (`LogisticRegressionCV`) on 41,716 labelled tweets, then evaluates it on a held-out 10%. Based on the "Make Training and Test Sets" section to the end of `notebooks/models/create/supervised/1.0-ams-create-best-classifiers.ipynb` in [Hemphill et al.'s repository](https://github.com/casmlab/modeling-political-attention).

## Data files required

| File | What it is | Where it comes from |
|---|---|---|
| `russell_processed_0.9.csv.gz` | Hemphill et al.'s balanced Russell training corpus (41,716 labelled tweets) | `data/russell/russell_processed_0.9.csv.gz` in [Hemphill et al.'s repository](https://github.com/casmlab/modeling-political-attention) |

None of the data files are committed to this repository. Copy each one into this folder before running `docker build`; the build fails if any are missing.

## Running

    docker build -t ams-create-best-classifiers .
    docker run --rm ams-create-best-classifiers

A full run takes about 1.5 minutes with 8 CPUs. There's no shorter test mode; this is already short.

## Limiting CPU cores

The benchmark uses every CPU the container is allowed. To run on fewer, pin the container to specific CPUs, e.g. `docker run --rm --cpuset-cpus=0-3 ams-create-best-classifiers`. `--cpus=<n>` also works, but it's a time quota, so the work can still spread across all of the host's cores. On machines with hyperthreading, two "CPUs" can share one physical core: `cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list` shows which CPUs pair up.

## What it measures

For fitting the model and for evaluating it, the script reports:

- wall-clock time
- CPU time, split into the main process and the parallel worker processes
- effective cores used, out of the CPUs available to the container
- change in memory
- throughput: training tweets per second for the fit (37,544 tweets), and test tweets per second for the evaluation (4,172 tweets)

Evaluation time includes printing the classification report and drawing the confusion matrices.

## Differences from the original

- **The model is fitted once.** The original notebook fits logistic regression twice:
  - first in a loop over nine dataset versions, with 10% to 90% of not-policy tweets removed;
  - then once more for the 90% version alone, to produce the final model (`lr_0.9.pkl`, which Benchmark 4 uses).

  Only the 90% version is public, so with this data both fits would be identical, and the benchmark runs it once.
- **The other eight dataset versions can't be rebuilt.** Each one would need the full Russell dataset (~20,122 not-policy tweets), which isn't public. Both public Russell files have only 2,012 not-policy tweets each (`russell_processed_0.9.csv.gz` and `russell_liwc_w2v_both.csv.gz`, drawn separately). Combined, that's 3,800 unique tweets, which isn't enough for any other version.
- **Only logistic regression is fitted.** The notebook also fits Dummy and Naive Bayes classifiers.
- **`max_iter=1000`.** The original used scikit-learn's default of 100. More iterations means more work per fit. scikit-learn still prints a few "failed to converge" warnings during the run (far fewer than the original), and they don't stop the benchmark.
- **Fitting runs in parallel on all available cores** (`n_jobs`). The original ran on a single core.
- **Confusion-matrix plots are drawn but not displayed**, since the container has no screen.
- **`stop-words` is version 2018.7.23.** The originally listed 2.0.1 doesn't exist for Python 3.6.

## Reference results

On an 8-CPU laptop (4 cores with hyperthreading), with all 8 CPUs: fitting took 90 s (6.5 effective cores, 417 training tweets/s), and evaluation took 1.6 s. Cohen's kappa was 0.775.
