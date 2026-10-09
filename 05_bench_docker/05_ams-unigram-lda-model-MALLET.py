import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gzip
import itertools
import os
import re
import shutil
import subprocess
import time

import psutil
from joblib import cpu_count

# ---------------------------------------------------------
# Benchmark 5: LDA topic model training with MALLET
# ---------------------------------------------------------
# Reproduces the 'Command Line Model Training' section of Hemphill et al.'s
# notebooks/models/create/unsupervised/1.0-ams-unigram-lda-model-MALLET.ipynb.
# The notebook's preprocessing cells need raw tweets that can't be shared, so
# we start from the already-preprocessed corpus (one row per tweet with
# columns id_str, user_id, preprocessed_text).
#
# Differences from the notebook's commands:
#   - import-file splits each line on the CSV commas (name = id_str,
#     label = user_id, data = tokens) and the CSV header is skipped. MALLET's
#     default line parsing splits on whitespace, which turned the first two
#     words of every tweet into the document name and label and trained on
#     the header as if it were a tweet.
#   - train-topics reads the corpus import-file actually wrote (the notebook
#     pointed at a different path) and passes --output-topic-keys once (it
#     was given twice; the last one, congress115_keys_50.txt, won).
#   - train-topics uses every available core (--num-threads) to measure
#     whole-system throughput. The notebook ran single-threaded.

MALLET_DIR = os.environ.get('MALLET_DIR', '/opt/mallet-2.0.8/bin/mallet')
INPUT_FILE = os.environ.get('INPUT_FILE', 'congress115_preprocessed.csv.gz')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
NUM_TOPICS = int(os.environ.get('NUM_TOPICS', '50'))
NUM_ITERATIONS = int(os.environ.get('NUM_ITERATIONS', '1000'))  # MALLET's default
## Train on only the first NUM_DOCS tweets (e.g. for quick functionality tests); unset = all
NUM_DOCS = int(os.environ['NUM_DOCS']) if os.environ.get('NUM_DOCS') else None

## Use every CPU available to the container (respects docker --cpus / --cpuset-cpus)
N_JOBS = cpu_count()

model_source = os.path.join(OUTPUT_DIR, 'model_source')
model_output = os.path.join(OUTPUT_DIR, 'model_output')
os.makedirs(model_source, exist_ok=True)
os.makedirs(model_output, exist_ok=True)

corpus_txt = os.path.join(model_source, 'congress115_preprocessed.csv')
train_mallet = os.path.join(model_source, 'congress115_train.mallet')

# ---------------------------------------------------------
# Measurement Hooks
# ---------------------------------------------------------

def cpu_seconds():
    """
    Returns (main process CPU seconds, worker process CPU seconds).

    MALLET runs in a separate Java process, so its CPU time is counted as
    worker time: the sum of live descendant processes plus children that
    have already exited and been reaped (reported by os.times()).
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

def measure_execution(func, name="Task"):
    """
    Measurement hook wrapper to log execution timing and system memory utilization.
    """
    print(f"\n--- [START] {name} ---")
    n_cpus = cpu_count()
    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)

    start_wall = time.perf_counter()
    start_main, start_workers = cpu_seconds()

    result = func()

    end_wall = time.perf_counter()
    end_main, end_workers = cpu_seconds()
    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)

    wall_elapsed = end_wall - start_wall
    main_elapsed = end_main - start_main
    workers_elapsed = end_workers - start_workers
    cpu_elapsed = main_elapsed + workers_elapsed

    print(f"--- [END] {name} ---")
    print(f"Wall-clock time : {wall_elapsed:.4f} seconds")
    print(f"CPU time        : {cpu_elapsed:.4f} seconds "
          f"(main {main_elapsed:.4f} + workers {workers_elapsed:.4f})")
    print(f"Effective Cores : {cpu_elapsed / max(wall_elapsed, 1e-6):.2f} / {n_cpus}")
    print(f"RAM Used Delta  : {mem_after - mem_before:.2f} MB")
    if isinstance(result, dict) and 'peak_rss_mb' in result:
        print(f"Peak MALLET RSS : {result['peak_rss_mb']:.2f} MB")
    print()

    return result, wall_elapsed

def parse_mallet_time(line):
    """
    Parse train-topics' 'Total time: [D days] [H hours] [M minutes] S seconds'
    line, printed when Gibbs sampling finishes (before outputs are written).
    """
    units = {'days': 86400, 'hours': 3600, 'minutes': 60, 'seconds': 1}
    return sum(int(n) * units[u] for n, u in re.findall(r'(\d+) (days|hours|minutes|seconds)', line))

def run_mallet(args):
    """
    Run a MALLET command, echoing its output. Returns the Java process's peak
    resident memory, plus MALLET's reported total token count and sampling
    time when train-topics prints them.
    """
    cmd = [MALLET_DIR] + args
    print(' '.join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    total_tokens = None
    sampling_seconds = None
    for line in proc.stdout:
        print(line, end='')
        match = re.match(r'total tokens: (\d+)', line)
        if match:
            total_tokens = int(match.group(1))
        if line.startswith('Total time:'):
            sampling_seconds = parse_mallet_time(line)
    proc.stdout.close()

    ## wait4 reaps the launcher script and reports resource usage for it and the
    ## java process it waited on, including peak RSS (KB on Linux)
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return {'peak_rss_mb': usage.ru_maxrss / 1024, 'total_tokens': total_tokens,
            'sampling_seconds': sampling_seconds}

# ---------------------------------------------------------
# Prepare Corpus
# ---------------------------------------------------------

def decompress_corpus():
    """
    MALLET reads plain text, so write the corpus out uncompressed, minus the
    CSV header. If NUM_DOCS is set, keep only the first NUM_DOCS tweets.
    """
    with gzip.open(INPUT_FILE, 'rt', encoding='utf-8') as src, \
         open(corpus_txt, 'w', encoding='utf-8') as dst:
        src.readline()  # header: id_str,user_id,preprocessed_text
        if NUM_DOCS is None:
            shutil.copyfileobj(src, dst)
        else:
            dst.writelines(itertools.islice(src, NUM_DOCS))

print(f"Training {NUM_TOPICS}-topic LDA model for {NUM_ITERATIONS} iterations "
      f"with {N_JOBS} threads on {'all' if NUM_DOCS is None else f'the first {NUM_DOCS:,}'} tweets")

measure_execution(decompress_corpus, name="Decompressing Corpus")

# ---------------------------------------------------------
# Command Line Model Training
# ---------------------------------------------------------

## Step 1: Import corpus into MALLET format
import_cmd = [
    'import-file',
    '--input', corpus_txt,
    '--line-regex', r'^([^,]*),([^,]*),(.*)$',
    '--name', '1', '--label', '2', '--data', '3',
    '--keep-sequence',
    '--output', train_mallet,
]

measure_execution(lambda: run_mallet(import_cmd), name="Importing Corpus (MALLET)")

## Step 2: Train the MALLET LDA model
train_cmd = [
    'train-topics',
    '--input', train_mallet,
    '--inferencer-filename', os.path.join(model_output, 'lda_tw_50_unigram_congress115_inferencer_.mallet'),
    '--num-topics', str(NUM_TOPICS),
    '--num-iterations', str(NUM_ITERATIONS),
    '--num-threads', str(N_JOBS),
    '--optimize-interval', '10',
    '--output-model', os.path.join(model_output, 'lda_tw_50_unigram_congress115.model'),
    '--output-state', os.path.join(model_output, 'congress115_topic_state_50.gz'),
    '--output-topic-keys', os.path.join(model_output, 'congress115_keys_50.txt'),
    '--output-doc-topics', os.path.join(model_output, 'congress115_compostion_50.txt'),
]

train_result, train_wall = measure_execution(
    lambda: run_mallet(train_cmd), name="Training LDA Model (MALLET)"
)

## The training step also loads the corpus and writes the model, state and
## doc-topics files, which is mostly single-threaded I/O. Report throughput
## both for the whole step and for Gibbs sampling alone (MALLET's own timer,
## whole seconds).
if train_result['total_tokens']:
    token_iters = train_result['total_tokens'] * NUM_ITERATIONS
    print(f"Total tokens    : {train_result['total_tokens']:,}")
    print(f"Throughput      : {token_iters / train_wall:,.0f} token-iterations/second (whole training step)")
    if train_result['sampling_seconds']:
        print(f"Sampling time   : {train_result['sampling_seconds']} seconds")
        print(f"Throughput      : {token_iters / train_result['sampling_seconds']:,.0f} "
              f"token-iterations/second (Gibbs sampling only)")
