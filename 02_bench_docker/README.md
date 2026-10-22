# Benchmark 2: Regressions of engagement on post topic

Models how the topic of federal candidates' social media posts (Dataset 1) relates to engagement. Posts are grouped by issue (policy issue, hybrid, or culture wars), and a separate linear regression of log likes on issue group is fitted for every combination of office (House or Senate), party and platform, spread across all available CPUs with `furrr`. The coefficients are then plotted for the Senate and the House.

Based on `02_random_effects.R` in this repository.

> **Note:** the top-level README describes this benchmark as a computationally intensive random effects model, with a random intercept for each of around 3,000 politicians. The script fits ordinary linear regressions instead (about 60 small ones), and the whole run takes seconds. If the random effects model is what's intended, the script needs changing. The synthetic data includes a candidate ID column (`candidate_UUID`) that such a model could use.

## Data files required

| File | What it is | Where it comes from |
|---|---|---|
| `all_posts_w_topic_and_candidate.csv` | Dataset 1's posts, each with its topic flags and its candidate's office and party | Not available yet: to be provided by our collaborators. The Dataset 1 files we have (`posts.csv`, `candidates.csv`) have no topic flags and no candidate ID for each post. |

The script uses these columns:

| Column | Values |
|---|---|
| `likes`, `shares` | counts; posts with no like count are dropped |
| `platform` | Dataset 1's codes: `facebook`, `twitter`, `instagram`, `rumble`, `t` (Telegram), `threads`, `tiktok`, `truthsocial`, `youtube`, `gettr` |
| `Office.Name` | `U.S. Representative` or `U.S. Senator` (other offices are dropped) |
| `Party.Standardized` | e.g. `Republican Party`, `Democratic Party`, `Third Party` |
| `crt_dei`, `abortion`, `cancel_culture`, `court`, `environ`, `guns`, `health`, `imm`, `lgbt`, `police`, `socialism` | 1 if the post is about the topic |

None of the data files are committed to this repository. Copy `all_posts_w_topic_and_candidate.csv` into this folder before running `docker build`. Without it, the build fails unless you use synthetic data, below.

### Synthetic data for testing

Until the real file arrives, the image can be built with synthetic data instead:

    docker build --build-arg SYNTHETIC=1 -t ams-engagement-regressions .

This runs `generate_synthetic_data.R` during the build. It writes a file with the columns above and Dataset 1's size and proportions: 3,321,323 posts, 1,758 candidates, the same platform mix, and similar House/Senate and party splits. Topics, likes and shares are random, so **the regression results mean nothing**; it's only for checking that the benchmark runs and for rough timings. The script prints a note when it's using synthetic data.

If the real file is also in the folder, the benchmark uses the real file.

## Running

    docker build -t ams-engagement-regressions .
    mkdir -p output
    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/output":/app/output ams-engagement-regressions

A run on the synthetic data takes about 7 seconds with 8 CPUs.

- The `-v` mount keeps the figures in `./output/figures/`. Without it, they're deleted with the container.
- `--user` makes the output files belong to you instead of root.

Arguments after the image name are passed to the script, in this order:

| Argument | Default | Meaning |
|---|---|---|
| input file | `all_posts_w_topic_and_candidate.csv`, or the synthetic file if the image has no real one | The posts file |
| max rows | all | Read only the first N rows, for a quicker test |

For example, `all_posts_w_topic_and_candidate.csv 200000` after the image name reads only the first 200,000 posts.

## Limiting CPU cores

The regressions use one worker session per CPU the container is allowed. To run on fewer, pin the container to specific CPUs, e.g. add `--cpuset-cpus=0-3` to `docker run`. `--cpus=<n>` also works, but it's a time quota, so the work can still spread across all of the host's cores. On machines with hyperthreading, two "CPUs" can share one physical core: `cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list` shows which CPUs pair up.

## What it measures

For starting the worker sessions, reading the data and fitting the regressions, the script reports:

- wall-clock time
- CPU time, split into the main R session and the worker sessions
- effective cores used, out of the CPUs available to the container
- change in memory, for the main session and the workers together

It also reports the regressions' throughput in posts per second (posts with an issue group and a party, which are the ones the regressions use). The original `tictoc` timings are kept: data ingestion and cleaning, the regressions, plotting, and the total.

## Differences from the original

The original script couldn't run as written. Three fixes were needed:

- **tidyr is loaded.** The script calls `nest()`, which is in tidyr, but didn't load it.
- **The worker sessions load broom.** furrr didn't send broom to the workers, so its `tidy()` method for regression results wasn't found.
- **Groups with only one issue group are skipped**, and listed in the output. A regression on issue group needs posts from at least two groups. In the synthetic data, two groups have a single post (Senate Republicans and third-party senators on Threads), and the real data may have groups like this too.

Other changes:

- **The input file and a row limit are arguments.** By default, the script reads `all_posts_w_topic_and_candidate.csv` from the working directory, as the original did.
- **The figures are saved to `output/figures/`** instead of `figures/`, so a mount can keep them. The copy of each plot that Rscript would also write to `Rplots.pdf` is discarded.
- **Timing hooks** matching the other benchmarks were added. CPU time and memory are read from `/proc` for every process in the container, which captures the worker sessions.
- **The R packages are those released with R 4.6.1**, from the `rocker/r-ver:4.6.1` image.

## Reference results

On an 8-CPU laptop (4 cores with hyperthreading), with all 8 CPUs, on the synthetic data (3,321,323 posts):

| Step | Time | Effective cores |
|---|---|---|
| Starting worker sessions | 0.8 s | 5.0 |
| Reading the data | 1.1 s | 2.9 |
| Regressions (58 groups, 860,764 posts) | 2.8 s | 3.9 |

The whole run took 6.7 s. Timings on the real data may differ: it will have different topic rates and, probably, more columns to read.
