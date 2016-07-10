# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:25:50 2015

@author: Esteban Paez-Torres
"""

"""
When i wrote this, only God and I knew what I was doing.
Now, only Gos knows.
"""

import numpy as np
from accuracy import accuracy
from L1Normalization import L1Normalization
from OKMF import OKMF
from sklearn.cluster import MiniBatchKMeans

def LinearExperiment(dataset,k,budget,Gamma,Lambda,Alpha,runs,target='acc',l1=True,K=False):
    """
    Performs an experiment using OKMF and returns the list of objectives or
    accuracies of each experiment.
    
    Parameters
    ----------
    dataset : H5 dataset
        The dataset to use in the experiment
    k : int
        The number of latent topics of the KMF
    budget : int
        The size of the budget
    Gamma : double
        The Gamma constant to be used
    Lambda : double
        The Lambda constant to be used
    Alpha : double
        The Alpha constant to be used
    runs : int
        The number of runs
    target : string
        A string to select the measure to use in The experiment.
        The posible measures are: 'acc' or 'obj' 
    l1 : boolean
        True if L1 Normalization is going to be used. Default True
    K : boolean
        True if KMeans budget is to be used. Default False.
    
    Returns
    -------
    result : ndarray
        An array representing the performance mesure for each run
    
    """
    X = dataset.get('/data')[:,:]
    if l1:
        X = L1Normalization(X)
    LG = dataset.get('/labels')[:]
    measures = np.zeros((runs,))
    ok = OKMF(budget,k,30,10,Gamma,Lambda,Alpha,'linear')
    Budget = None
    if K:
        KM = MiniBatchKMeans(n_clusters = budget)
        KM.fit(X)
        Budget = KM.cluster_centers_
    for i in xrange(runs):
        try:
            if K:
                ok.fit(X,Budget=Budget)
            else:
                ok.fit(X)
            if target=='acc':
                LF = np.argmax(ok.H,axis=0)
                measures[i]=accuracy(LF,LG)
            elif target=='obj':
                measures[i]=ok.Error(X)
        except np.linalg.LinAlgError :
            if target=='acc':
                LF = np.argmax(ok.H,axis=0)
                measures[i]=0.0
            elif target=='obj':
                measures[i]=float('inf')
    return measures

def RBFExperiment(dataset,k,budget,Gamma,Lambda,Alpha,Sigma,runs,target='acc',l1=True,K=False):
    """
    Performs an experiment using OKMF and returns the list of objectives or
    accuracies of each experiment.
    
    Parameters
    ----------
    dataset : H5 dataset
        The dataset to use in the experiment
    k : int
        The number of latent topics of the KMF
    budget : int
        The size of the budget
    Gamma : double
        The Gamma constant to be used
    Lambda : double
        The Lambda constant to be used
    Alpha : double
        The Alpha constant to be used
    Sigma : duble
        The Sigma to be used
    runs : int
        The number of runs
    target : string
        A string to select the measure to use in The experiment.
        The posible measures are: 'acc' or 'obj' 
    l1 : boolean
        True if L1 Normalization is going to be used. Default True
    K : boolean
        True if KMeans budget is to be used. Default False.
    
    Returns
    -------
    result : ndarray
        An array representing the performance mesure for each run
    
    """
    X = dataset.get('/data')[:,:]
    if l1:
        X = L1Normalization(X)
    LG = dataset.get('/labels')[:]
    measures = np.zeros((runs,))
    gam = 1.0 / (2.0 * (2.0**Sigma)**2.0)
    ok = OKMF(budget,k,30,10,Gamma,Lambda,Alpha,'rbf',gamma=gam)
    Budget = None
    if K:
        KM = MiniBatchKMeans(n_clusters = budget)
        KM.fit(X)
        Budget = KM.cluster_centers_
    for i in xrange(runs):
        try:
            if K:
                ok.fit(X,Budget=Budget)
            else:
                ok.fit(X)
            if target=='acc':
                LF = np.argmax(ok.H,axis=0)
                measures[i]=accuracy(LF,LG)
            elif target=='obj':
                measures[i]=ok.Error(X)
        except np.linalg.LinAlgError :
            if target=='acc':
                LF = np.argmax(ok.H,axis=0)
                measures[i]=0.0
            elif target=='obj':
                measures[i]=float('inf')
    return measures

