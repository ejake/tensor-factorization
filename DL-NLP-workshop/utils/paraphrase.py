import pandas as pd
import numpy as np
import keras
import sys
import os
sys.path.append(os.path.abspath('/home/datasets/datasets1/skip_thoughts_models/skip-thoughts'))
import skipthoughts
from collections import defaultdict
from nltk.tokenize import word_tokenize
from numpy.random import RandomState
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score as f1
from sklearn import cross_validation
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import re
from sklearn.ensemble import RandomForestClassifier
from nltk.util import ngrams
from collections import *
from sklearn.preprocessing import StandardScaler
from misc import build_histogram_new
from misc import get_config

def perform_cv(X_train, y_train, kfolds, seed, shuffle):
    '''
    Return The best s parameter using Logistic Regression
    
    Arguments:
    X_train: Features
    y_train: Training labels
    kfolds: Number of folds for Stratified KFold
    seed: Random number seed generator
    shuffle: Whether shuffle or not
    '''
    skf = StratifiedKFold(y_train, n_folds=kfolds, shuffle=shuffle)
    scan = [2**t for t in range(0,9,1)]
    scores = []
    for s in scan:
        scanscores = []
        for train, val in skf:
            clf = LogisticRegression(C=s)
            clf.fit(X_train[train], y_train[train])
            yhat = clf.predict(X_train[val])
            fscore = f1(y_train[val], yhat)
            scanscores.append(fscore)
            #print (s, fscore)
        # Append mean score
        scores.append(np.mean(scanscores))
        print 'Trying '+str(s)+' parameter'
    s_ind = np.argmax(scores)
    s = scan[s_ind]
    print scores
    print s
    return s

def get_class_distribution(count, x_labels=['No Plagiarism','Plagiarism']):
    #count = np.bincount(y)
    x = np.arange(len(count))
    plt.xticks(x, x_labels, rotation='45')
    plt.bar(x, count, width=0.5)
    plt.show()
    
def encode_word2vec(model, dataA, dataB):
    #EOS requires an additional treatment
    #Replace anything but a character for a space, lowercase everything and tokenize
    #Pending to add stop words
    feat_trainA = [[model[t] for t in sentence if model.vocab.has_key(t) ] for sentence in dataA]
    feat_trainB = [[model[t] for t in sentence if model.vocab.has_key(t) ] for sentence in dataB]
    agg_featA = np.array([np.sum(sentence, axis=0) for sentence in feat_trainA])
    agg_featB = np.array([np.sum(sentence, axis=0) for sentence in feat_trainB])
    return agg_featA, agg_featB

def encode_asobek(dataA, dataB):
    if len(dataA) != len(dataB):
        print 'Check length of your data'
        return
    features = []
    def get_cardinalities(ngramA, ngramB):
        vector = []
        vector.append(union(ngramA, ngramB))
        vector.append(intersect(ngramA, ngramB))
        vector.append(set(ngramA))
        vector.append(set(ngramB))
        return vector
    
    for x in np.arange(len(dataA)):
        unigram_1 = get_wordngram(dataA[x],1)
        unigram_2 = get_wordngram(dataB[x],1)
        bigram_1 = get_wordngram(dataA[x],2)
        bigram_2 = get_wordngram(dataB[x],2)
        unigram_c_1 = get_characterngram(dataA[x],1)
        unigram_c_2 = get_characterngram(dataB[x],1)
        bigram_c_1 = get_characterngram(dataA[x],2)
        bigram_c_2 = get_characterngram(dataB[x],2)
        w1 = [len(x) for x in get_cardinalities(unigram_1, unigram_2)]
        w2 = [len(x) for x in get_cardinalities(bigram_1, bigram_2)]
        c1 = [len(x) for x in get_cardinalities(unigram_c_1, unigram_c_2)]
        c2 = [len(x) for x in get_cardinalities(bigram_c_1, bigram_c_2)]
        features.append([w1, w2, c1, c2])
    return features

def union(list1, list2):
    cnt1 = Counter()
    cnt2 = Counter()
    for tk1 in list1:
        cnt1[tk1] += 1
    for tk2 in list2:
        cnt2[tk2] += 1
    inter = cnt1 | cnt2
    return set(inter.elements())
def intersect (list1, list2) :
    cnt1 = Counter()
    cnt2 = Counter()
    for tk1 in list1:
        cnt1[tk1] += 1
    for tk2 in list2:
        cnt2[tk2] += 1
    inter = cnt1 & cnt2
    return list(inter.elements())


def get_characterngram(string, n):
    char1 = [c for c in string]
    return list(ngrams(char1, n))

def get_wordngram(string, n):
    words = word_tokenize(string)
    return list(ngrams(words, n))


def feats(A, B):
    """
    Compute additional features (similar to Socher et al.)
    """
    tA = [t.split() for t in A]
    tB = [t.split() for t in B]
    
    nA = [[w for w in t if is_number(w)] for t in tA]
    nB = [[w for w in t if is_number(w)] for t in tB]

    features = np.zeros((len(A), 6))

    # n1
    for i in range(len(A)):
        if set(nA[i]) == set(nB[i]):
            features[i,0] = 1.
    # n2
    for i in range(len(A)):
        if set(nA[i]) == set(nB[i]) and len(nA[i]) > 0:
            features[i,1] = 1.
    # n3
    for i in range(len(A)):
        if set(nA[i]) <= set(nB[i]) or set(nB[i]) <= set(nA[i]): 
            features[i,2] = 1.
    # n4
    for i in range(len(A)):
        features[i,3] = 1.0 * len(set(tA[i]) & set(tB[i])) / len(set(tA[i]))
    # n5
    for i in range(len(A)):
        features[i,4] = 1.0 * len(set(tA[i]) & set(tB[i])) / len(set(tB[i]))
    # n6
    for i in range(len(A)):
        features[i,5] = 0.5 * ((1.0*len(tA[i]) / len(tB[i])) + (1.0*len(tB[i]) / len(tA[i])))

    return features
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract_feats(textA, textB, train_indices, test_indices, model, labels, add_feats, seed, strategy='skipthoughts', if_asobek=None):
    '''Extract features from sources
    w1w2 is equivalent to if_asobek=[0,1]
    c1w2 is equivalent to if_asobek=[2,1]
    c1w1 is equivalent to if_asobek=[2,0]
    c2w1 is equivalent to if_asobek=[3,0]
    c2w2 is equivalent to if_asobek=[3,1]
    c1c2 is equivalent to if_asobek=[2,3]
    
    Keyword arguments:
    textA and textB: Paraphrase Sources
    train_indices and test_indices: Indices 
    test_percentage: Percentage of dataset to be used as test
    model: Word2Vec or Skip-thoughts model, None in any other case
    labels: categories for paraphrase sources
    add_feats: Whether to add or not additional features over the text
    seed: Random Number Generator
    strategy: [word2vec, skipthoughts, asobek]
    if_asobek: What asobek features to add
    '''
      
    if strategy == 'skipthoughts':
        dataA, dataB = [], []
        #decode("utf-8") is really important
        dataA = [' '.join(word_tokenize(x.decode("utf-8").lower())) for x in textA]
        dataB = [' '.join(word_tokenize(x.decode("utf-8").lower())) for x in textB]
        print 'Computing training skipthoughts...'
        featA = skipthoughts.encode(model, dataA, verbose=False, use_eos=True)
        featB = skipthoughts.encode(model, dataB, verbose=False, use_eos=True)
        if add_feats:
            final_features = np.c_[np.abs(featA - featB), featA * featB, feats(textA, textB)]
        else:
            final_features = np.c_[np.abs(featA - featB), featA * featB]
    elif strategy == 'word2vec':
        dataA, dataB = [], []
        dataA = [word_tokenize(re.sub("[^a-zA-Z]", " ", t).lower()) for t in textA]
        dataB = [word_tokenize(re.sub("[^a-zA-Z]", " ", t).lower()) for t in textB]
        print 'Computing training word2vec...'
        featA, featB = encode_word2vec(model, dataA, dataB)
        if add_feats:
            final_features = np.c_[np.abs(featA - featB), featA * featB, feats(textA, textB)]
        else:
            final_features = np.c_[np.abs(featA - featB), featA * featB]
    elif strategy == 'asobek':
        if if_asobek is None:
            print 'Check Asobek parameter'
            return
        dataA = [' '.join(word_tokenize(x.decode("utf-8").lower())) for x in textA]
        dataB = [' '.join(word_tokenize(x.decode("utf-8").lower())) for x in textB]
        features = encode_asobek(dataA, dataB)
        final_features = np.array([x[if_asobek[0]]+x[if_asobek[1]] for x in features])
    elif strategy == 'histograms':
        final_features = build_histogram_new(textA, textB, model)
    
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(final_features, labels, test_size=test_percentage, random_state=seed)
    X_train = final_features[train_indices]
    X_test = final_features[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    
    return X_train, X_test, y_train, y_test

def preprocess_data(dataset):
    '''
    dataset must be a dataframe that follows the P4PIN original format.
    Returns the dataset with 1 or 0 instead of 'SI or NO' or 'Plagiarism or NoPlagiarism'.
    '''
    #Change 'SI' or 'NO' expressions
    for par_type in dataset.columns[-6:]:
        dataset[par_type][dataset[par_type] == 'NO'] = 0
        dataset[par_type][dataset[par_type] == 'SI'] = 1
    dataset['Class'][dataset['Class'] == 'NoPlagiarism'] = 0
    dataset['Class'][dataset['Class'] == 'Plagiarism'] = 1
    return dataset


