import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import logging
import traceback
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split as train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.utils import shuffle
from collections import OrderedDict

from util import iter_files, pre_process_email, word_count, tfidf, train_model, load_features


def experiment(counts, Y, isTfIdf=False):
    if isTfIdf:
        counts = tfidf(counts)
    models = {
        "1. Gaussian Naive Bayes" : GaussianNB(),
        "2. Logistic Regression L2, C=1": LogisticRegression(C=1.0, penalty='l2'),
        "3. Logistic Regression L2, C=0.5": LogisticRegression(C=0.5, penalty='l2'),
        "4. Logistic Regression L1, C=1": LogisticRegression(C=1, penalty='l1'),
        "5. Logistic Regression L1, C=0.5": LogisticRegression(C=0.5, penalty='l1'),
    }
    score_methods = {
        "precision:" : precision_score,
        "recall:": recall_score,
        "f1_score:": f1_score
    }
    orderedModels = OrderedDict(sorted(models.items(), key=lambda t: t[0]))
    for config, model in orderedModels.items():
        print("- " * 10)
        print(config)
        kf_5 = cross_validation.KFold(len(Y), n_folds=5)
        scores = {}
        for train_idx, test_idx in kf_5:
            X_train, X_test = counts[train_idx], counts[test_idx]
            y_train, y_test = Y[train_idx], Y[test_idx]
            clf = train_model(X_train.toarray(), y_train, model)
            y_predict = clf.predict(X_test.toarray())
            for item, method in score_methods.items():
                score = method(y_test, y_predict)
                if item not in scores:
                    scores[item] = [score]
                else:
                    scores[item].append(score)
        for item, score_list in scores.items():
            print("    average", item, np.mean(score_list))
            print("    std", item, np.std(score_list))


def main(corpus_base_folder):
    spam_folder = os.path.sep.join([corpus_base_folder, "spam"])
    ham_folder = os.path.sep.join([corpus_base_folder, "ham"])
    counts, Y  = load_features(
        [spam_folder, ham_folder],
        pre_process_email,
        word_count,
        [1, -1])
    counts, Y = shuffle(counts, Y, random_state=40)
    print("Using word count:")
    experiment(counts, Y, isTfIdf=False)
    print()
    print("= " * 20)
    print()
    print("Using TF-IDF:")
    experiment(counts, Y, isTfIdf=True)


if __name__ == "__main__":
    import os
    import sys
    if len(sys.argv) > 1:
        corpus_base_folder = sys.argv[1]
    else:
        corpus_base_folder = r'spamassasin'
    main(corpus_base_folder)
