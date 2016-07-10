# -*- coding: utf-8 -*-

import csv
import os.path
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import pickle
import theano
import yaml
from random import shuffle
import collections
import matplotlib.pyplot as plt
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import re
from nltk.tokenize import word_tokenize
try:
    import nltk
    stopwords = nltk.corpus.stopwords.words('english')
except:
    stopwords = []


config = None


def get_config(filename=None, reload=False):
    global config
    if config is not None and not reload:
        return config
    if filename is None:
        raise Exception(
            'Configuration has not been loaded previously, filename parameter is required')
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(curr_dir, 'config.yaml')
    with open(filename) as f:
        config = yaml.load(f)
    return config


def load_data():
    conf = get_config()
    with open(conf['data_path']) as f:
        return pickle.load(f)


def get_sentences(data):
    conf = get_config()
    return [LabeledSentence(normalizeText(text), ['SENT_' + str(i)])
            for i, text in data[0].iteritems()]


def get_rng():
    conf = get_config()
    return make_np_rng(None, conf['rng_seed'], which_method='uniform')


def train_doc2vec(sentences):
    conf = get_config()
    rng = get_rng()

    shuffle(sentences, rng.rand)
    model = Doc2Vec(size=conf['dimvec'],
                    window=conf['window_size'],
                    alpha=conf['alpha'],
                    min_alpha=conf['min_alpha'],
                    min_count=conf['min_count'],
                    dm=conf['dm'],
                    sample=conf['sample'],
                    negative=conf['negative'],
                    workers=conf['workers'],
                    seed=conf['rng_seed'])
    model.build_vocab(sentences)
    for i in range(conf['n_iter']):
        shuffle(sentences, rng.rand)
        model.train(sentences)
        model.alpha -= conf['learning_rate']
        model.min_alpha = model.alpha
    return model


def get_paths(conf=None):
    if conf is None:
        conf = get_config()
    paths = {}
    data_path = os.path.join(
        preprocess('${PYLEARN2_DATA_PATH}'), conf['ds_dir'])
    paths['data_path'] = data_path
    data_path = os.path.join(data_path, conf['ds_name'])
    paths['twitterPIT'] = os.path.dirname(os.path.realpath(__file__))
    paths['train'] = os.path.join(data_path, 'train.data')
    paths['dev'] = os.path.join(data_path, 'dev.data')
    paths['test'] = os.path.join(data_path, 'test.data')

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    paths['models_path'] = os.path.join(conf['models_path'], conf['ds_name'])
    return paths


def get_rows(data):
    conf = get_config()
    rng = get_rng()
    train = [data[1][idx][:-1] for idx in data[2]]
    dev = [data[1][idx][:-1] for idx in data[3]]
    test = [data[1][idx][:-1] for idx in data[4]]
    shuffle(train, rng.rand)
    return train, dev, test


def get_rows_msr(data):
    conf = get_config()
    rng = get_rng()
    train = [data[1][idx] for idx in data[3]]
    test = [data[1][idx] for idx in data[2]]
    # test = [data[1][idx] for idx in data[4]]
    shuffle(train, rng.rand)
    train_y = [y for y, o, p in train]
    # build dev set
    sss = StratifiedShuffleSplit(
        train_y, 1, train_size=0.8, test_size=0.2, random_state=rng)
    train_index, dev_index = sss.__iter__().next()
    return [train[i] for i in train_index], [train[i] for i in dev_index], test


def build_dataset(rows, model):
    '''
    Returns representations for both sentences given from the model.
    rows is a list of samples, each one represented by a list with [class, phrase1, phrase2]
    '''
    conf = get_config()
    X1 = np.zeros((len(rows), conf['dimvec']))
    X2 = np.zeros((len(rows), conf['dimvec']))
    y = np.zeros((len(rows)))
    for i in range(len(rows)):
        y[i], org, prop = rows[i]
        X1[i] = model['SENT_' + str(org)]
        X2[i] = model['SENT_' + str(prop)]
    return X1, X2, y


def get_datasets(data, model):
    train, dev, test = get_rows(data)
    X1, X2, train_y = build_dataset(train, model)
    train_X = fuse_doc_vec(X1, X2)
    X1, X2, dev_y = build_dataset(dev, model)
    dev_X = fuse_doc_vec(X1, X2)
    X1, X2, test_y = build_dataset(test, model)
    test_X = fuse_doc_vec(X1, X2)
    return train_X, train_y, dev_X, dev_y, test_X, test_y


def get_asobek(data):
    train, dev, test = get_rows(data)
    train = [[t[0], data[0][t[1]], data[0][t[2]]] for t in train]
    dev = [[t[0], data[0][t[1]], data[0][t[2]]] for t in dev]
    test = [[t[0], data[0][t[1]], data[0][t[2]]] for t in test]

    out = asobek.combineFeatures(train)
    train_X, train_y = np.array(out[0]), out[-1]
    out = asobek.combineFeatures(dev)
    dev_X, dev_y = np.array(out[0]), out[-1]
    out = asobek.combineFeatures(test)
    test_X, test_y = np.array(out[0]), out[-1]
    return train_X, train_y, dev_X, dev_y, test_X, test_y


def build_histogram_new(textA, textB, model, norm=True, range=(0, 1.1)):
    if len(textA) != len(textB):
        print 'Check textA and B lengths'
        return
    import __builtin__
    conf = get_config()
    X = np.zeros((len(textA), conf['hist_bins'] * 7))
    dataA = [' '.join(word_tokenize(x.decode("utf-8").lower())) for x in textA]
    dataB = [' '.join(word_tokenize(x.decode("utf-8").lower())) for x in textB]
    for i in __builtin__.range(len(textA)):
        w1 = normalizeText(dataA[i])
        w2 = normalizeText(dataB[i])
        w_all = list(set(w1 + w2))
        P1 = doc2mat(model, w1)
        P2 = doc2mat(model, w2)
        B1 = get_ngrams(model, w1, ngrams=2)
        B1 = B1[:, :, 0] - B1[:, :, 1]
        B2 = get_ngrams(model, w2, ngrams=2)
        B2 = B2[:, :, 0] - B2[:, :, 1]
        Pall = doc2mat(model, w_all)

        def to_hist(M1, M2):
            sim = 1 - np.nan_to_num(distance.cdist(M1, M2, conf['hist_dist']))
            h = np.histogram(
                sim.flatten(), bins=conf['hist_bins'], range=range)[0]
            if norm:
                # Normalize histogram
                h = h / float(h.sum())
            return h

        h1 = to_hist(P1, P2)
        h2 = to_hist(P1, P1)
        h3 = to_hist(P2, P2)
        h4 = to_hist(B1, B2)
        h5 = to_hist(B1, B1)
        h6 = to_hist(B2, B2)
        h7 = to_hist(Pall, Pall)

        X[i] = np.concatenate((h1, h2, h3, h4, h5, h6, h7))

    return X

def build_histogram(rows, data, model, norm=True, range=(0, 1.1)):
    import __builtin__
    conf = get_config()
    X = np.zeros((len(rows), conf['hist_bins'] * 7))
    y = np.zeros((len(rows)))
    for i in __builtin__.range(len(rows)):
        y[i], org, prop = rows[i]
        w1 = normalizeText(data[0][org])
        w2 = normalizeText(data[0][prop])
        w_all = list(set(w1 + w2))
        P1 = doc2mat(model, w1)
        P2 = doc2mat(model, w2)
        B1 = get_ngrams(model, w1, ngrams=2)
        B1 = B1[:, :, 0] - B1[:, :, 1]
        B2 = get_ngrams(model, w2, ngrams=2)
        B2 = B2[:, :, 0] - B2[:, :, 1]
        Pall = doc2mat(model, w_all)

        def to_hist(M1, M2):
            sim = 1 - np.nan_to_num(distance.cdist(M1, M2, conf['hist_dist']))
            h = np.histogram(
                sim.flatten(), bins=conf['hist_bins'], range=range)[0]
            if norm:
                # Normalize histogram
                h = h / float(h.sum())
            return h

        h1 = to_hist(P1, P2)
        h2 = to_hist(P1, P1)
        h3 = to_hist(P2, P2)
        h4 = to_hist(B1, B2)
        h5 = to_hist(B1, B1)
        h6 = to_hist(B2, B2)
        h7 = to_hist(Pall, Pall)

        X[i] = np.concatenate((h1, h2, h3, h4, h5, h6, h7))

    return X, y


def get_histograms(data, model):
    train, dev, test = get_rows(data)
    train_X, train_y = build_histogram(train, data, model)
    dev_X, dev_y = build_histogram(dev, data, model)
    test_X, test_y = build_histogram(test, data, model)
    return train_X, train_y, dev_X, dev_y, test_X, test_y


def get_ngrams(model, doc, ngrams=2):
    doc = [w for w in doc if model.vocab.has_key(w)]
    if len(doc) < ngrams:
        doc = ngrams // 2 * ['NULL'] + doc + ngrams // 2 * ['NULL']

    grams = np.zeros((len(doc) - 1, model.layer1_size, ngrams))
    for i in range(len(doc) - ngrams - 1):
        for j in range(ngrams):
            key = doc[i + j] if model.vocab.has_key(doc[i + j]) else 'NULL'
            grams[i, :, j] = model[key]
    return grams


def fuse_doc_vec(X1, X2):
    '''
    Fuse representations of both paragraphs to get a single input for the classifier
    '''
    conf = get_config()
    X = None
    for fsn in conf['fusion_vec']:
        if fsn == 'concat':
            X = np.hstack((X1, X2)) if X is None else np.hstack((X, X1, X2))
        if fsn == 'diff':
            X = X1 - X2 if X is None else np.hstack((X, X1 - X2))
    return X


def doc2mat(model, doc, unique=True):
    '''
    Represents a document by a matrix with vector representations of each word.
    It ignores words that do not belong to model's vocabulary
    '''
    if unique:
        doc = list(set(doc))
    doc = [w for w in doc if model.vocab.has_key(w)]
    X = np.zeros((len(doc), model.layer1_size))
    for i in range(len(doc)):
        X[i] = model[doc[i]]
    return X


def export_results(filename, y, score):
    with open(filename, 'w') as fw:
        for l, s in zip(y, score):
            fw.write('{0}\t{1:0.4f}\n'.format('true' if l else 'false', s))


def normalizeText(text, remove_stop=False):
    conf = get_config()
    text = text.lower()
    text = re.sub(r'([\.?!\(\)",:;])', r' \1 ', text).strip()
    words = text.split()
    if remove_stop:
        words = [w for w in words if w not in stopwords]
    if conf['fill_null'] and len(words) < conf['window_size']:
        words = conf['window_size'] // 2 * ['NULL'] + \
            words + conf['window_size'] // 2 * ['NULL']
    return words


def train_classifier(train_X, train_y, dev_X, dev_y):
    conf = get_config()
    # Normalize data
    scaler = StandardScaler()
    if conf['normalize']:
        train_X = scaler.fit_transform(train_X)
        dev_X = scaler.transform(dev_X)

    # Explore param classifier
    clsf = LinearSVC(random_state=0)
    #clsf = LogisticRegression(random_state=0)
    C_opts = eval(conf['C_opts'])
    scores = np.zeros_like(C_opts)
    for i, c in enumerate(C_opts):
        print 'exploring {0}-th param'.format(i)
        clsf.set_params(C=c).fit(train_X, train_y)
        pred_y = clsf.predict(dev_X)
        pr, rc, f1, s = precision_recall_fscore_support(
            dev_y, pred_y, average='micro')
        scores[i] = f1

    best_c = C_opts[scores.argmax()]
    clsf.set_params(C=best_c).fit(
        np.vstack((train_X, dev_X)), np.hstack((train_y, dev_y)))
    return clsf, scaler


def train_rf(train_X, train_y, dev_X, dev_y):
    conf = get_config()
    # Normalize data
    scaler = StandardScaler()
    if conf['normalize']:
        train_X = scaler.fit_transform(train_X)
        dev_X = scaler.transform(dev_X)

    # Explore param classifier
    clsf = RandomForestClassifier(random_state=0, n_jobs=8)
    n_trees_opts = eval(conf['n_trees_opts'])
    scores = np.zeros_like(n_trees_opts)
    for i, n_trees in enumerate(n_trees_opts):
        clsf.set_params(n_estimators=n_trees).fit(train_X, train_y)
        pred_y = clsf.predict(dev_X)
        pr, rc, f1, s = precision_recall_fscore_support(
            dev_y, pred_y, average='micro')
        scores[i] = f1

    best_n_trees = n_trees_opts[scores.argmax()]
    clsf.set_params(n_estimators=best_n_trees).fit(
        np.vstack((train_X, dev_X)), np.hstack((train_y, dev_y)))
    return clsf, scaler


def create_hdf5_file(X, y, filename):
    f = h5py.File(filename, "w")
    X = X.reshape(X.shape + (1,))  # Include channel dimension
    f.create_dataset('X', data=X)
    f.create_dataset('y', data=y)
    f.close()


def load_csv(file=None):
    conf = get_config()
    if file is None:
        file = conf['csv_file']
    path = os.path.join(get_paths()['data_path'], file)
    with open(path) as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        return [dict(zip(headers, map(str.strip, row))) for row in reader]


def get_filtered_rows():
    conf = get_config()
    return [r for r in load_csv() if check_filter(r, conf['filters'])]


def check_filter(row, filter):
    for key, value in filter.iteritems():
        if str(row[key]) not in value:
            return False
    return True


def init_hdf5(path, shapes, title="Pylearn2 Dataset", filters=None):
    from theano import config
    if filters is None:
        filters = tables.Filters(complib='blosc', complevel=5)
    x_shape, y_shape, feats_shape = shapes
    h5file = tables.open_file(path, mode="w", title=title)
    node = h5file.create_group(h5file.root, "Data", "Data")
    atom = (tables.Float32Atom() if config.floatX == 'float32'
            else tables.Float64Atom())
    h5file.create_carray(node, 'X', atom=atom, shape=x_shape,
                         title="Data values", filters=filters)
    h5file.create_carray(node, 'y', atom=atom, shape=y_shape,
                         title="Data targets", filters=filters)
    h5file.create_carray(node, 'feats', atom=atom, shape=feats_shape,
                         title="Data targets", filters=filters)
    return h5file, node


def h5py_to_tables(inputfile, outputfile, title='exported_pytables'):
    hf = tables.open_file(inputfile)
    shapes = (hf.root.X.shape, hf.root.y.shape, hf.root.feats.shape)
    h5file, node = init_hdf5(outputfile, shapes, title)
    for i, x in enumerate(hf.root.X):
        node.X[i] = x

    for i, y in enumerate(hf.root.y):
        node.y[i] = y

    for i, feats in enumerate(hf.root.feats):
        node.feats[i] = feats
    hf.close()
    return h5file, node


def flatten(d, parent_key='', sep='_'):
    """
    Flatten a dictionary with nested subdictionaries
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def plot_roc_curve(y, score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(y, score)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
             ''.format(roc_auc["micro"]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
