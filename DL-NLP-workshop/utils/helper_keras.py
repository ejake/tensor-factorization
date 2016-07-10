from __future__ import absolute_import
import pandas as pd 
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
from six.moves import cPickle
import gzip
from six.moves import zip
import numpy as np

def sentence_to_wordlist(raw_review, stop=False, tokenized=True):
    review_text = BeautifulSoup(raw_review).getText()
    only_letters = re.sub("[^a-zA-Z]", " ", review_text)
    words = only_letters.lower().split()
    if stop:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    if not tokenized:
        return(' '.join(words))
    return(words)

def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review).getText()        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return( " ".join( meaningful_words ))


def load_imdb(X, labels, nb_words=None, skip_top=0,
              maxlen=None, test_split=0.2, seed=113,
              start_char=1, oov_char=2, index_from=3):

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)

    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels
    if not X:
        raise Exception('After filtering for sequences shorter than maxlen=' +
                        str(maxlen) + ', no sequence was kept. '
                        'Increase maxlen.')
    if not nb_words:
        nb_words = max([max(x) for x in X])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX

    X_train = X[:int(len(X) * (1 - test_split))]
    y_train = labels[:int(len(X) * (1 - test_split))]

    X_test = X[int(len(X) * (1 - test_split)):]
    y_test = labels[int(len(X) * (1 - test_split)):]

    return (X_train, y_train), (X_test, y_test)