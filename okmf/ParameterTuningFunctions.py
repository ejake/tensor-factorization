# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 17:34:04 2014

@author: Esteban Paez-Torres
"""

import numpy as np
from accuracy import accuracy
from L1Normalization import L1Normalization
from OKMF import OKMF
from sklearn.cluster import MiniBatchKMeans

def LinearTuning(dataset,k,budgets,Gammas,Lambdas,Alphas,runs,target='acc',l1=True,K=False):
    """
    Performs a parameter tunning for OKMF using linear kernels and random
    budgets.
    
    Parameters
    ----------
    dataset : H5 File
        A file containing the dataset and labels for the clustering task.
    k : int
        The number of clusters to be found.
    budgets : Array like
        An array containing the size of budgets to be tuned.
    Gammas : array like
        An array containing the learning rates to be tuned.
    Lambdas : array like
        An array containing the W regularization parameters to be tuned.
    Alphas : array like
        An array containing the H regularization parameters to be tuned.
    runs : int
        The number of runs to tune each parameter.
    target : string
        A string to select the measure to use in the parameter tuning
        The posible measures are: 'acc' or 'obj' 
    l1 : boolean
        True if L1 Normalization is going to be used. Default True
    K : boolean
        True if KMeans budget is to be used. Default False.
    """
    # Printing info
    print 'Linear tuning' + dataset.filename
    # Dataset loading
    X = dataset.get('/data')[:,:]
    if l1:
        X = L1Normalization(X)
    LG = dataset.get('/labels')[:]
    n = len(budgets)
    # Budget size tuning
    averageAcc = np.zeros((n,))
    averageObj = np.zeros((n,))
    for i in xrange(n):
        print 'Tuning budget %i of %i'%(i+1,n)
        budget = budgets[i]
        if K:
            KM = MiniBatchKMeans(n_clusters = budget)
            KM.fit(X)
            Budget = KM.cluster_centers_
        accs = np.zeros((runs,))
        objs = np.zeros((runs,))
        for j in xrange(runs):
            ok = OKMF(budget,k,30,10,0.8,0.1,0.2,'linear')
            try:
                if K:
                    ok.fit(X,Budget=Budget)
                else:
                    ok.fit(X)
                LF = np.argmax(ok.H,axis=0)
                if target == 'acc':
                    accs[j] = accuracy(LF,LG)
                elif target == 'obj':
                    objs[j] = ok.Error(X)
            except np.linalg.LinAlgError:
                print "LinAlgError found"
                if target == 'acc':
                    accs[j] = float('-inf')
                elif target == 'obj':
                    objs[j] = float('inf')
            del ok
        averageAcc[i] = np.average(accs)
        averageObj[i] = np.average(objs)
    # Tuned budget
    if target == 'acc':
        budget = budgets[np.argmax(averageAcc)]
    elif target == 'obj':
        budget = budgets[np.argmin(averageObj)]
    del averageAcc,averageObj
    Budget = None
    if K:
        KM = MiniBatchKMeans(n_clusters = budget)
        KM.fit(X)
        Budget = KM.cluster_centers_
    # Gamma tuning
    n = len(Gammas)
    averageAcc = np.zeros((n,))
    averageObj = np.zeros((n,))
    for i in xrange(n):
        print 'Tuning Gamma %i of %i'%(i+1,n)
        Gamma = Gammas[i]
        accs = np.zeros((runs,))
        objs = np.zeros((runs,))
        for j in xrange(runs):
            ok = OKMF(budget,k,30,10,Gamma,0.1,0.2,'linear')
            try:
                if K:
                    ok.fit(X,Budget=Budget)
                else:
                    ok.fit(X)
                LF = np.argmax(ok.H,axis=0)
                if target == 'acc':
                    accs[j] = accuracy(LF,LG)
                elif target == 'obj':
                    objs[j] = ok.Error(X)
            except np.linalg.LinAlgError:
                print "LinAlgError found"
                if target == 'acc':
                    accs[j] = float('-inf')
                elif target == 'obj':
                    objs[j] = float('inf')
            del ok
        averageAcc[i] = np.average(accs)
        averageObj[i] = np.average(objs)
    # Tuned Gamma
    if target == 'acc':
        Gamma = Gammas[np.argmax(averageAcc)]
    elif target == 'obj':
        Gamma = Gammas[np.argmin(averageObj)]
    del averageAcc,averageObj
    # Lambda tuning
    n = len(Lambdas)
    averageAcc = np.zeros((n,))
    averageObj = np.zeros((n,))
    for i in xrange(n):
        print 'Tuning Lambda %i of %i'%(i+1,n)
        Lambda = Lambdas[i]
        accs = np.zeros((runs,))
        objs = np.zeros((runs,))
        for j in xrange(runs):
            ok = OKMF(budget,k,30,10,Gamma,Lambda,0.2,'linear')
            try:
                if K:
                    ok.fit(X,Budget=Budget)
                else:
                    ok.fit(X)
                LF = np.argmax(ok.H,axis=0)
                if target == 'acc':
                    accs[j] = accuracy(LF,LG)
                elif target == 'obj':
                    objs[j] = ok.Error(X)
            except np.linalg.LinAlgError:
                print "LinAlgError found"
                if target == 'acc':
                    accs[j] = float('-inf')
                elif target == 'obj':
                    objs[j] = float('inf')
            del ok
        averageAcc[i] = np.average(accs)
        averageObj[i] = np.average(objs)
    # Tuned Lamnda
    if target == 'acc':
        Lambda = Lambdas[np.argmax(averageAcc)]
    elif target == 'obj':
        Lambda = Lambdas[np.argmin(averageObj)]
    del averageAcc,averageObj
    # Alpha tuning
    n = len(Alphas)
    averageAcc= np.zeros((n,))
    averageObj = np.zeros((n,))
    for i in xrange(n):
        print 'Tuning Alpha %i of %i'%(i+1,n)
        Alpha = Alphas[i]
        accs = np.zeros((runs,))
        objs = np.zeros((runs,))
        for j in xrange(runs):
            ok = OKMF(budget,k,30,10,Gamma,Lambda,Alpha,'linear')
            try:
                if K:
                    ok.fit(X,Budget=Budget)
                else:
                    ok.fit(X)
                LF = np.argmax(ok.H,axis=0)
                if target == 'acc':
                    accs[j] = accuracy(LF,LG)
                elif target == 'obj':
                    objs[j] = ok.Error(X)
            except np.linalg.LinAlgError:
                print "LinAlgError found"
                if target == 'acc':
                    accs[j] = float('-inf')
                elif target == 'obj':
                    objs[j] = float('inf')
            del ok
        averageAcc[i] = np.average(accs)
        averageObj[i] = np.average(objs)
    # Tuned Alpha
    if target == 'acc':
        Alpha = Alphas[np.argmax(averageAcc)]
    elif target == 'obj':
        Alpha = Alphas[np.argmin(averageObj)]
    del averageAcc,averageObj
    result = dict()
    result['budget'] = budget
    result['Gamma'] = Gamma
    result['Lambda'] = Lambda
    result['Alpha'] = Alpha
    return result

def RBFTuning(dataset,k,budgets,Gammas,Lambdas,Alphas,Sigmas,runs,target='acc',l1=True,K=False):
    """
    Performs a parameter tunning for OKMF using rbf kernels and random
    budgets.
    
    Parameters
    ----------
    dataset : H5 File
        A file containing the dataset and labels for the clustering task.
    k : int
        The number of clusters to be found.
    budgets : Array like
        An array containing the size of budgets to be tuned.
    Gammas : array like
        An array containing the learning rates to be tuned.
    Lambdas : array like
        An array containing the W regularization parameters to be tuned.
    Alphas : array like
        An array containing the H regularization parameters to be tuned.
    Sigmas : array like
        An array containing the RBF's sigma parameters to be tuned.
    runs : int
        The number of runs to tune each parameter.
    target : string
        A string to select the measure to use in the parameter tuning
        The posible measures are: 'acc' or 'obj' 
    l1 : boolean
        True if L1 Normalization is going to be used. Default True
    K : boolean
        True if KMeans budget is to be used. Default False.
    """
    print 'RBF tuning' + dataset.filename
    # Dataset loading
    X = dataset.get('/data')[:,:]
    if l1:
        X = L1Normalization(X)
    LG = dataset.get('/labels')[:]
    n = len(budgets)
    # Budget size tuning
    averageAcc = np.zeros((n,))
    averageObj = np.zeros((n,))
    for i in xrange(n):
        print 'Tuning Budget %i of %i'%(i+1,n)
        budget = budgets[i]
        if K:
            KM = MiniBatchKMeans(n_clusters = budget)
            KM.fit(X)
            Budget = KM.cluster_centers_
        accs = np.zeros((runs,))
        objs = np.zeros((runs,))
        for j in xrange(runs):
            ok = OKMF(budget,k,30,10,0.8,0.1,0.2,'rbf')
            try:
                if K:
                    ok.fit(X,Budget=Budget)
                else:
                    ok.fit(X)
                LF = np.argmax(ok.H,axis=0)
                if target == 'acc':
                    accs[j] = accuracy(LF,LG)
                elif target == 'obj':
                    objs[j] = ok.Error(X)
            except np.linalg.LinAlgError:
                print "LinAlgError found"
                if target == 'acc':
                    accs[j] = float('-inf')
                elif target == 'obj':
                    objs[j] = float('inf')
            del ok
        averageAcc[i] = np.average(accs)
        averageObj[i] = np.average(objs)
    # Tuned budget
    if target == 'acc':
        budget = budgets[np.argmax(averageAcc)]
    elif target == 'obj':
        budget = budgets[np.argmin(averageObj)]
    del averageAcc,averageObj
    Budget = None
    if K:
        KM = MiniBatchKMeans(n_clusters = budget)
        KM.fit(X)
        Budget = KM.cluster_centers_
    # Gamma tuning
    n = len(Gammas)
    averageAcc = np.zeros((n,))
    averageObj = np.zeros((n,))
    for i in xrange(n):
        print 'Tuning Gamma %i of %i'%(i+1,n)
        Gamma = Gammas[i]
        accs = np.zeros((runs,))
        objs = np.zeros((runs,))
        for j in xrange(runs):
            ok = OKMF(budget,k,30,10,Gamma,0.1,0.2,'rbf')
            try:
                if K:
                    ok.fit(X,Budget=Budget)
                else:
                    ok.fit(X)
                LF = np.argmax(ok.H,axis=0)
                if target == 'acc':
                    accs[j] = accuracy(LF,LG)
                elif target == 'obj':
                    objs[j] = ok.Error(X)
            except np.linalg.LinAlgError:
                print "LinAlgError found"
                if target == 'acc':
                    accs[j] = float('-inf')
                elif target == 'obj':
                    objs[j] = float('inf')
            del ok
        averageAcc[i] = np.average(accs)
        averageObj[i] = np.average(objs)
    # Tuned Gamma
    if target == 'acc':
        Gamma = Gammas[np.argmax(averageAcc)]
    elif target == 'obj':
        Gamma = Gammas[np.argmin(averageObj)]
    del averageAcc,averageObj
    # Lambda tuning
    n = len(Lambdas)
    averageAcc = np.zeros((n,))
    averageObj = np.zeros((n,))
    for i in xrange(n):
        print 'Tuning Lambda %i of %i'%(i+1,n)
        Lambda = Lambdas[i]
        accs = np.zeros((runs,))
        objs = np.zeros((runs,))
        for j in xrange(runs):
            ok = OKMF(budget,k,30,10,Gamma,Lambda,0.2,'rbf')
            try:
                if K:
                    ok.fit(X,Budget=Budget)
                else:
                    ok.fit(X)
                LF = np.argmax(ok.H,axis=0)
                if target == 'acc':
                    accs[j] = accuracy(LF,LG)
                elif target == 'obj':
                    objs[j] = ok.Error(X)
            except np.linalg.LinAlgError:
                print "LinAlgError found"
                if target == 'acc':
                    accs[j] = float('-inf')
                elif target == 'obj':
                    objs[j] = float('inf')
            del ok
        averageAcc[i] = np.average(accs)
        averageObj[i] = np.average(objs)
    # Tuned Lamnda
    if target == 'acc':
        Lambda = Lambdas[np.argmax(averageAcc)]
    elif target == 'obj':
        Lambda = Lambdas[np.argmin(averageObj)]
    del averageAcc,averageObj
    # Alpha tuning
    n = len(Alphas)
    averageAcc= np.zeros((n,))
    averageObj = np.zeros((n,))
    for i in xrange(n):
        print 'Tuning Alpha %i of %i'%(i+1,n)
        Alpha = Alphas[i]
        accs = np.zeros((runs,))
        objs = np.zeros((runs,))
        for j in xrange(runs):
            ok = OKMF(budget,k,30,10,Gamma,Lambda,Alpha,'rbf')
            try:
                if K:
                    ok.fit(X,Budget=Budget)
                else:
                    ok.fit(X)
                LF = np.argmax(ok.H,axis=0)
                if target == 'acc':
                    accs[j] = accuracy(LF,LG)
                elif target == 'obj':
                    objs[j] = ok.Error(X)
            except np.linalg.LinAlgError:
                print "LinAlgError found"
                if target == 'acc':
                    accs[j] = float('-inf')
                elif target == 'obj':
                    objs[j] = float('inf')
            del ok
        averageAcc[i] = np.average(accs)
        averageObj[i] = np.average(objs)
    # Tuned Alpha
    if target == 'acc':
        Alpha = Alphas[np.argmax(averageAcc)]
    elif target == 'obj':
        Alpha = Alphas[np.argmin(averageObj)]
    del averageAcc,averageObj
    # Sigma tuning
    n = len(Sigmas)
    averageAcc= np.zeros((n,))
    averageObj = np.zeros((n,))
    for i in xrange(n):
        print 'Tuning Sigma %i of %i'%(i+1,n)
        Sigma = Sigmas[i]
        gam = 1.0 / (2.0 * (2.0**Sigma)**2.0)
        accs = np.zeros((runs,))
        objs = np.zeros((runs,))
        for j in xrange(runs):
            ok = OKMF(budget,k,30,10,Gamma,Lambda,Alpha,'rbf',gamma=gam)
            try:
                if K:
                    ok.fit(X,Budget=Budget)
                else:
                    ok.fit(X)
                LF = np.argmax(ok.H,axis=0)
                if target == 'acc':
                    accs[j] = accuracy(LF,LG)
                elif target == 'obj':
                    objs[j] = ok.Error(X)
            except np.linalg.LinAlgError:
                print "LinAlgError found"
                if target == 'acc':
                    accs[j] = float('-inf')
                elif target == 'obj':
                    objs[j] = float('inf')
            del ok
        averageAcc[i] = np.average(accs)
        averageObj[i] = np.average(objs)
    # Tuned Sigma
    if target == 'acc':
        Sigma = Sigmas[np.argmax(averageAcc)]
    elif target == 'obj':
        Sigma = Sigmas[np.argmin(averageObj)]
    del averageAcc,averageObj
    result = dict()
    result['budget'] = budget
    result['Gamma'] = Gamma
    result['Lambda'] = Lambda
    result['Alpha'] = Alpha
    result['Sigma'] = Sigma
    return result
