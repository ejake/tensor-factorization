# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 16:50:02 2014

@author: Jose David Bermeo
"""

import numpy
from time import clock

def KKmeans(K, latent_topics, epochs):
    t0 = clock()
    W = train(K, epochs, latent_topics)
    t1 = clock()
    return (W, t1-t0)

def train(K, epochs, latent_topics):
    data_set_size = K.shape[0]
    W = initW(latent_topics, data_set_size)
    for i in xrange(epochs):
        dist_to_centroids = -2*numpy.dot(W, K) \
            + (numpy.dot(W, K)*W).sum(axis = 1)[:, numpy.newaxis]
        cluster = numpy.argmin(dist_to_centroids, axis = 0)
        W = numpy.zeros((latent_topics, data_set_size))
        for j in xrange(data_set_size):
            W[cluster[j], j] = 1
        W = W*(1/W.sum(axis = 1))[:, numpy.newaxis]
    return W

def initW(latent_topics, data_set_size):
    permutation = numpy.random.permutation(data_set_size) 
    W = numpy.zeros((latent_topics, data_set_size))
    for i in xrange(latent_topics):
        W[i, permutation[i]] = 1
    return W*(1/W.sum(axis = 1))[:, numpy.newaxis]


def predictH(K, W, KX):
    dist_to_centroids = -2*numpy.dot(W, KX) \
        + (numpy.dot(W, K)*W).sum(axis = 1)[:, numpy.newaxis]
    return numpy.argmin(dist_to_centroids, axis = 0)
