import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import csv
import gzip
import os
import shutil
import subprocess
import time

import psutil
from joblib import cpu_count

# ---------------------------------------------------------
# Benchmark 6: LDA topic inference with MALLET
# ---------------------------------------------------------
# Reproduces the 'Command Line Topic Inference' section of Hemphill et al.'s
# notebooks/models/create/unsupervised/1.0-ams-unigram-lda-model-MALLET.ipynb:
# import new documents with the training corpus's pipe (--use-pipe-from), then
# infer their topic distributions with the published inferencer.
#
# Differences from the notebook's commands:
#   - By default the documents are the public congress115.csv.gz corpus
#     (1,485,834 tweets, preprocessed_text column). The top-level README
#     specifies Hemphill et al.'s balanced Russell corpus
#     (russell_processed_0.9.csv.gz, 41,716 tweets; set
#     INPUT_FILE=russell_processed_0.9.csv.gz to use it), but at that size
#     each MALLET process's fixed startup cost (starting Java, loading the
#     inferencer and its 431k-word vocabulary) outweighs the inference work,
#     so adding cores lowered throughput. The notebook's own input,
#     russell_preprocessed.csv.gz, isn't published.
#   - --use-pipe-from points at a rebuilt pipe whose vocabulary matches the
#     published inferencer word for word (built when the image is built; see
#     the Dockerfile and export_vocabulary.bsh). The authors'
#     congress115_train.mallet isn't published.
#   - import-file splits each line on commas (name, label, data) instead of
#     MALLET's default whitespace splitting, which dropped the first two words
#     of every document (see Benchmark 5).
#   - infer-topics is single-threaded, so to measure whole-system throughput
#     the corpus is split into one chunk per available CPU, each chunk is
#     imported and inferred by its own MALLET process in parallel, and the
#     per-chunk outputs are merged into one doc-topics file.

MALLET_DIR = os.environ.get('MALLET_DIR', '/opt/mallet-2.0.8/bin/mallet')
## Default is the full public congress115 corpus (1,485,834 tweets); see the notes below
INPUT_FILE = os.environ.get('INPUT_FILE', 'congress115.csv.gz')
INFERENCER = os.environ.get('INFERENCER', 'model/lda_tw_50_unigram_congress115_inferencer.mallet')
PIPE_FROM = os.environ.get('PIPE_FROM', 'model/congress115_vocabulary_pipe.mallet')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')

## Use every CPU available to the container (respects docker --cpus / --cpuset-cpus)
N_JOBS = cpu_count()

model_source = os.path.join(OUTPUT_DIR, 'model_source')
model_output = os.path.join(OUTPUT_DIR, 'model_output')
os.makedirs(model_source, exist_ok=True)
os.makedirs(model_output, exist_ok=True)

## output files are named after the input corpus, e.g. congress115-topic-composition.txt
corpus_name = os.path.basename(INPUT_FILE).split('.csv')[0]
chunk_txt = [os.path.join(model_source, f'{corpus_name}_preprocessed_{i}.csv') for i in range(N_JOBS)]
chunk_mallet = [os.path.join(model_source, f'{corpus_name}_infer_{i}.mallet') for i in range(N_JOBS)]
chunk_pipe = [os.path.join(model_source, f'congress115_vocabulary_pipe_{i}.mallet') for i in range(N_JOBS)]
chunk_topics = [os.path.join(model_output, f'{corpus_name}-topic-composition_{i}.txt') for i in range(N_JOBS)]
composition_txt = os.path.join(model_output, f'{corpus_name}-topic-composition.txt')

# ---------------------------------------------------------
# Measurement Hooks
# ---------------------------------------------------------

def cpu_seconds():
    """
    Returns (main process CPU seconds, worker process CPU seconds).

    MALLET runs in separate Java processes, so their CPU time is counted as
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
        print(f"Peak MALLET RSS : {result['peak_rss_mb']:.2f} MB "
              f"(sum over {result['processes']} concurrent processes)")
    print()

    return result, wall_elapsed

def run_mallet_parallel(arg_lists):
    """
    Run one MALLET command per chunk, all at once. Each process's output goes
    to its own log file (printed if it fails). Returns the summed peak resident
    memory of the processes, an upper bound on their combined peak.
    """
    procs = []
    for i, args in enumerate(arg_lists):
        log = open(os.path.join(model_source, f'mallet_{args[0]}_{i}.log'), 'w')
        procs.append((subprocess.Popen([MALLET_DIR] + args, stdout=log, stderr=subprocess.STDOUT), log))
    print(f"{len(procs)} x {MALLET_DIR} {' '.join(arg_lists[0])}")

    peak_rss_kb = 0
    failed = []
    for proc, log in procs:
        ## wait4 reaps the launcher script and reports resource usage for it and
        ## the java process it waited on, including peak RSS (KB on Linux)
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        peak_rss_kb += usage.ru_maxrss
        log.close()
        if proc.returncode != 0:
            failed.append(log.name)
    for name in failed:
        print(open(name).read())
    if failed:
        raise RuntimeError(f"MALLET failed; see {', '.join(failed)}")
    return {'peak_rss_mb': peak_rss_kb / 1024, 'processes': len(procs)}

# ---------------------------------------------------------
# Prepare Corpus
# ---------------------------------------------------------

## Columns holding each document's tokens and ID: congress115.csv.gz has
## (preprocessed_text, id_str); russell_processed_0.9.csv.gz has text_tokenized
## and no ID column, so its documents are named by row number
TEXT_COLUMNS = ['preprocessed_text', 'text_tokenized']
ID_COLUMN = 'id_str'

def split_corpus():
    """
    Write the corpus as plain-text MALLET import lines (name = tweet ID or row
    number, no label, data = tokens), dealing rows round-robin into one chunk
    per CPU so row r lands in chunk r % N_JOBS.

    Each chunk also gets its own copy of the training pipe: when documents
    contain words the model never saw, import-file adds them to the pipe's
    vocabulary and rewrites the --use-pipe-from file, so parallel imports
    can't share one. (The new words get IDs beyond the inferencer's
    vocabulary, so inference ignores them.)
    """
    for path in chunk_pipe:
        shutil.copyfile(PIPE_FROM, path)
    outs = [open(path, 'w', encoding='utf-8') for path in chunk_txt]
    with gzip.open(INPUT_FILE, 'rt', encoding='utf-8', newline='') as src:
        reader = csv.DictReader(src)
        text_column = next(c for c in TEXT_COLUMNS if c in reader.fieldnames)
        num_rows = 0
        for row_num, row in enumerate(reader):
            name = row[ID_COLUMN] if ID_COLUMN in row else row_num
            outs[row_num % N_JOBS].write(f"{name},,{row[text_column] or ''}\n")
            num_rows += 1
    for out in outs:
        out.close()
    return num_rows

def merge_compositions():
    """
    Merge per-chunk doc-topics files into one in the original row order, by
    reading the chunks round-robin (the inverse of split_corpus).
    """
    ins = [open(path, encoding='utf-8') for path in chunk_topics]
    for f in ins:
        f.readline()  # '#doc name topic proportion ...' header
    num_rows = 0
    with open(composition_txt, 'w', encoding='utf-8') as out:
        out.write('#doc name topic proportion ...\n')
        while True:
            line = ins[num_rows % N_JOBS].readline()
            if not line:
                break
            doc, rest = line.split('\t', 1)
            out.write(f"{num_rows}\t{rest}")
            num_rows += 1
    for f in ins:
        f.close()
    return num_rows

print(f"Inferring topics with {N_JOBS} parallel MALLET processes")

(num_docs, _) = measure_execution(split_corpus, name="Splitting Corpus")
print(f"Split {num_docs:,} tweets from {INPUT_FILE} into {N_JOBS} chunks")

# ---------------------------------------------------------
# Command Line Topic Inference
# ---------------------------------------------------------

## Step 1: Import documents into MALLET format, using the training pipe so word
## IDs match the inferencer
import_cmds = [[
    'import-file',
    '--input', chunk_txt[i],
    '--line-regex', r'^([^,]*),([^,]*),(.*)$',
    '--name', '1', '--label', '0', '--data', '3',
    '--keep-sequence',
    '--output', chunk_mallet[i],
    '--use-pipe-from', chunk_pipe[i],
] for i in range(N_JOBS)]

measure_execution(lambda: run_mallet_parallel(import_cmds), name="Importing Documents (MALLET)")

## Step 2: Infer the topic distribution of each document with the trained inferencer
infer_cmds = [[
    'infer-topics',
    '--input', chunk_mallet[i],
    '--inferencer', INFERENCER,
    '--output-doc-topics', chunk_topics[i],
] for i in range(N_JOBS)]

_, infer_wall = measure_execution(lambda: run_mallet_parallel(infer_cmds), name="Inferring Topics (MALLET)")
print(f"Throughput      : {num_docs / infer_wall:,.1f} tweets/second ({num_docs:,} tweets)")

(merged, _) = measure_execution(merge_compositions, name="Merging Topic Compositions")
print(f"Wrote topic distributions for {merged:,} tweets to {composition_txt}")
