import warnings
warnings.simplefilter('ignore')

import os
import re
import time
import string
import pandas as pd
import numpy as np
import psutil
from sklearn.externals import joblib
from joblib import Parallel, delayed, cpu_count
from stop_words import get_stop_words
from nltk.tokenize.casual import TweetTokenizer

# ---------------------------------------------------------
# Tokenization & Stop Words Setup
# ---------------------------------------------------------

stop_words_en = get_stop_words('en')
stop_words_sp = get_stop_words('spanish')

stop_words_2 = []
for word in stop_words_en:
    stop_words_2.append(word)
for word in stop_words_sp:
    stop_words_2.append(word)

def tokenize_func(doc, tknzr=TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)):
    # remove URLs
    doc = re.sub(r'^(https|http)?:\/\/.*(\r|\n|\b)', '', doc, flags=re.MULTILINE)
    
    # tokenize
    tokens = tknzr.tokenize(doc)

    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    nopunct_tokens = [w.translate(table) for w in tokens]
    for tok in nopunct_tokens:
        if 'httptco' in tok:
            ind = nopunct_tokens.index(tok)
            del(nopunct_tokens[ind])
        elif 'httpstco' in tok:
            ind = nopunct_tokens.index(tok)
            del(nopunct_tokens[ind])

    # keep words length 3 or more
    long_tokens = [token for token in nopunct_tokens if len(token) > 2]

    # remove remaining tokens that are not alphabetic
    alpha_tokens = [word for word in long_tokens if word.isalpha()]

    # remove stop words from tokens
    stopped_tokens = [i for i in alpha_tokens if not i in stop_words_2]
        
    return stopped_tokens

def replace_urls(text):
    text_clean = re.sub(r'^(https|http)?:\/\/.*(\r|\n|\b)', '', text, flags=re.MULTILINE)
    return text_clean

# ---------------------------------------------------------
# Measurement Hooks
# ---------------------------------------------------------

def cpu_seconds():
    """
    Returns (main process CPU seconds, worker process CPU seconds).

    Worker time is the sum of live descendant processes plus children that
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
    print(f"RAM Used Delta  : {mem_after - mem_before:.2f} MB\n")
    
    return result

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------

## The original notebook (notebooks/models/run/1.0-ams-label-comparisons.ipynb)
## first merges in unsupervised MALLET topic labels. That step needs the private
## full congress115 dataset (full_text, text_tokenized, party, ...) and the
## hand-built congress115_keys_50.txt topic-to-policy mapping, neither of which
## is public. Benchmark 4 is only the 'Label Data with Supervised Model
## Predictions' section, so we run that on the public congress115.csv.gz, whose
## preprocessed_text column holds each tweet's tokens.

data_file = 'congress115.csv.gz'
tweets = measure_execution(
    lambda: pd.read_csv(data_file, compression='gzip', dtype=str),
    name="Loading Tweets"
)
tweets['preprocessed_text'] = tweets['preprocessed_text'].fillna('')
print(f"Loaded {len(tweets)} tweets")

# ---------------------------------------------------------
# Label Data with Supervised Model Predictions
# ---------------------------------------------------------

## load best model, logistic regression with no w2v or liwc features, and preserving 90% not-policy tweets
MODEL_PATH = 'lr_0.9.pkl'

## To measure the throughput of the whole system, prediction is split into
## chunks that run in parallel worker processes on every available core.
N_JOBS = cpu_count()
CHUNKS_PER_JOB = 4  # several chunks per core so faster workers pick up the slack

def predict_chunk(texts):
    """
    Predict policy labels for one chunk of tweets inside a worker process.

    Each worker loads the model once and caches it on its own __main__ module,
    so the 130 MB model isn't re-sent with every chunk. The pickled model
    refers to __main__.tokenize_func, so that name is registered first.
    """
    import __main__
    if not hasattr(__main__, '_lr_model'):
        warnings.simplefilter('ignore')  # workers don't inherit the main process filter
        __main__.tokenize_func = tokenize_func
        with open(MODEL_PATH, 'rb') as file:
            __main__._lr_model = joblib.load(file)
    return __main__._lr_model.predict(texts)

def predict_sup_labels(df):
    texts = df.preprocessed_text.values
    chunks = np.array_split(texts, N_JOBS * CHUNKS_PER_JOB)
    results = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(predict_chunk)(chunk) for chunk in chunks
    )
    df['policy_area_lr'] = np.concatenate(results)
    return df

## generate new predictions
print(f"Predicting with {N_JOBS} worker processes")
start = time.perf_counter()
tweets_complete_labeled = measure_execution(
    lambda: predict_sup_labels(tweets),
    name="Predicting Policy Labels"
)
elapsed = time.perf_counter() - start
print(f"Throughput      : {len(tweets) / elapsed:,.0f} tweets/second")

print("Predicted label counts (top 10):")
print(tweets_complete_labeled['policy_area_lr'].value_counts().head(10))
