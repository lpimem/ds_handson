from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import logging
import traceback
import os
import numpy as np

from email_processor import extract_email_text

def iter_files(folder_path, handler, encoding=None):
    for base, folders, files in os.walk(folder_path):
        for file in files:
            try:
                with open(os.path.sep.join([base, file]), 'r', encoding=encoding) as f:
                    handler(f)
            except Exception as e:
                traceback.print_exc()


def pre_process_email(f):
    # logging.debug("pre-process email: %s", str(f.name))
    email_text = extract_email_text(email_file=f)
    # logging.debug(email_text)
    return email_text


def word_count(corpus):
    # count words that appears at least once.
    wc = CountVectorizer(min_df=1)
    return wc.fit_transform(corpus)


def tfidf(counts, smooth=False):
    tfidfc = TfidfTransformer(smooth)
    return tfidfc.fit_transform(counts)


def train_model(X, Y, model):
    return model.fit(X, Y)

def load_features(folders, extract_method, count_method, labels):
    corpus = []
    Y = []
    def extract(i):
        def ex(file):
            text = extract_method(file)
            corpus.append(text)
            Y.append(i)
        return ex

    for i in range(len(labels)):
        if i >= len(folders):
            break
        folder = folders[i]
        label = labels[i]
        iter_files(folder,
            extract(label),
            encoding='ISO-8859-1')
    return count_method(corpus), np.array(Y)
