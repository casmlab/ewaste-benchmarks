import warnings
warnings.simplefilter('ignore')

from sklearn import svm
import pickle
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.externals import joblib


from stop_words import get_stop_words
pd.options.mode.chained_assignment = None
import string

import time
import re
import os
import json
import subprocess
import csv
import gzip
from datetime import datetime
import itertools

import matplotlib.pyplot as plt


# Tokenizers
from nltk.tokenize.casual import TweetTokenizer

# Imbalanced classes
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTE, ADASYN

import eli5

# from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def show_conf_matrices(y_test, y_predictions):
    ## from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    ## compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_predictions)
    np.set_printoptions(precision=2)

    ## make class labels
    y_labels = []
    for item in y_test:
        if item not in y_labels:
            y_labels.append(item)
        else:
            continue
    
    ## plot non-normalized confusion matrix
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cnf_matrix,classes=y_labels,
                          title='Confusion matrix, without normalization')

    ## plot normalized confusion matrix
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cnf_matrix, classes=y_labels, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

def eval_classifier(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred) +  
          "\nCohen's kappa: " + str(cohen_kappa_score(y_test,y_pred)))
    
    show_conf_matrices(y_test, y_pred)  

# tokenizer = RegexpTokenizer(r'\w+')
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True) # which tokenizer to use

stop_words_en = get_stop_words('en') # English stoplist
stop_words_sp = get_stop_words('spanish') # Spanish stoplist

stop_words_2 = []

for word in stop_words_en:
    stop_words_2.append(word)
for word in stop_words_sp:
    stop_words_2.append(word)

def tokenize_func(doc):

    ## remove URLs
    doc = re.sub(r'^(https|http)?:\/\/.*(\r|\n|\b)', '', doc, flags=re.MULTILINE)
    
    ## tokenize
    tokens = tokenizer.tokenize(doc)

    ## remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    nopunct_tokens = [w.translate(table) for w in tokens]
    for tok in nopunct_tokens:
        if 'httptco' in tok:
            ind = nopunct_tokens.index(tok)
            del(nopunct_tokens[ind])
        elif 'httpstco' in tok:
            ind = nopunct_tokens.index(tok)
            del(nopunct_tokens[ind])

    ## keep words length 3 or more
    long_tokens = [token for token in nopunct_tokens if len(token) > 2]

    ## remove remaining tokens that are not alphabetic
    alpha_tokens = [word for word in long_tokens if word.isalpha()]

    ## remove stop words from tokens
    stopped_tokens = [i for i in alpha_tokens if not i in stop_words_2]
        
    return stopped_tokens

def replace_urls(text):
    text_clean = re.sub(r'^(https|http)?:\/\/.*(\r|\n|\b)', '', text, flags=re.MULTILINE)
    return text_clean

## load data here if attempting to reproduce the below code, this is Russell's processed data
data_file = 'russell_processed_0.9.csv.gz'
tweets = pd.read_csv(data_file, dtype=str, compression='gzip')

#filling in empties to prevent errors
tweets['text_no_urls'] = tweets['text_no_urls'].fillna('')

fractions_dfs = {0.9:tweets}

def make_training_docs(df):
    
    x = np.array(df.text_no_urls)
    y = np.asarray(df['new_topic_1'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)
    
    return x_train, x_test, y_train, y_test

def make_dict_balanced_training_docs(fractions_dfs):

    ## creating a dict of all of the different fractions of non-policy tweets to include in train and test sets
    train_test = {}

    for k,v in fractions_dfs.items():
        x_train, x_test, y_train, y_test = make_training_docs(v)
        train_test[k] = x_train, x_test, y_train, y_test
        # print("Using " + str(len(x_train)) + " original tweets to TRAIN the model. This excludes Retweets.")
        # print("Using " + str(len(x_test)) + " original tweets to TEST the model. This excludes Retweets.")
    
    return train_test

train_test = make_dict_balanced_training_docs(fractions_dfs)

## Turn train and test sets into vectors using CountVectorizer (Bag of Words)
vec = CountVectorizer(tokenizer=tokenize_func)

#%%time
lr = LogisticRegressionCV(multi_class='ovr')

pipeline = make_pipeline(vec,lr)

# # v[0] = x_train, v[2] = y_train, v[1] = x_test, v[3] = y_test
# for k,v in train_test.items():
#     print("WITH " + str(k*100) + "% 0S REMOVED")
#     pipeline.fit(v[0], v[2])
# #     joblib.dump(pipeline, '../../../../models/best/lr/lr_' + str(k) + '.pkl')
#     eval_classifier(pipeline, v[1], v[3])

# #%%time
# lr = LogisticRegressionCV(multi_class='ovr')

# pipeline = make_pipeline(vec,
#                         lr)

# ## reminder: v[0] = x_train, v[2] = y_train, v[1] = x_test, v[3] = y_test
# for k,v in train_test.items():
#     if k == 0.9:
#         print("WITH " + str(k*100) + "% 0S REMOVED")
#         pipeline.fit(v[0], v[2])
#         ## uncomment to save model
#         # joblib.dump(pipeline, '../../../../models/best/lr/lr_' + str(k) + '.pkl')
#         eval_classifier(pipeline, v[1], v[3])
#     else:
#         continue

import time
import psutil
from joblib import parallel_backend

def measure_execution(func, name="Task"):
    """
    Measurement hook wrapper to log execution timing and system memory utilization.
    """
    print(f"\n--- [START] {name} ---")
    cpu_count = psutil.cpu_count(logical=True)
    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
    
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    
    result = func()
    
    end_wall = time.perf_counter()
    end_cpu = time.process_time()
    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
    
    wall_elapsed = end_wall - start_wall
    cpu_elapsed = end_cpu - start_cpu
    
    print(f"--- [END] {name} ---")
    print(f"Wall-clock time : {wall_elapsed:.4f} seconds")
    print(f"CPU time        : {cpu_elapsed:.4f} seconds")
    print(f"Effective Cores : {cpu_elapsed / max(wall_elapsed, 1e-6):.2f} / {cpu_count}")
    print(f"RAM Used Delta  : {mem_after - mem_before:.2f} MB\n")
    
    return result

# ==========================================
# Loop 1: Scaled over all cores using n_jobs=-1
# ==========================================

# Enforce multi-core scaling with n_jobs=-1
lr_1 = LogisticRegressionCV(multi_class='ovr',max_iter=1000, n_jobs=-1)
pipeline_1 = make_pipeline(vec, lr_1)

for k, v in train_test.items():
    print(f"WITH {k*100}% 0S REMOVED")
    
    # Enforce loky multi-processing backend across all machine CPU cores
    with parallel_backend('loky', n_jobs=-1):
        measure_execution(
            lambda: pipeline_1.fit(v[0], v[2]), 
            name=f"Fitting Model (Fraction {k})"
        )
    
    measure_execution(
        lambda: eval_classifier(pipeline_1, v[1], v[3]), 
        name=f"Evaluating Classifier (Fraction {k})"
    )

# ==========================================
# Loop 2: Filtered Loop with Measurement Hooks
# ==========================================

lr_2 = LogisticRegressionCV(multi_class='ovr',max_iter=1000, n_jobs=-1)
pipeline_2 = make_pipeline(vec, lr_2)

for k, v in train_test.items():
    if k == 0.9:
        print(f"WITH {k*100}% 0S REMOVED")
        
        with parallel_backend('loky', n_jobs=-1):
            measure_execution(
                lambda: pipeline_2.fit(v[0], v[2]), 
                name="Fitting Model (k=0.9)"
            )
            
        measure_execution(
            lambda: eval_classifier(pipeline_2, v[1], v[3]), 
            name="Evaluating Classifier (k=0.9)"
        )
    else:
        continue
