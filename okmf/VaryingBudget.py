# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:57:27 2015

@author: aepaezt
"""

from accuracy import accuracy
from OKMF import OKMF
from h5py import File
from time import clock
import numpy as np
import pickle

def experimentsAbalone():
    budgets = [3,10,50,100,200,500,1000,1500,2000,3000,4000,4177]
    Sigmas = range(-10,5,1)
    Gammas = [1.0,0.9,0.8,0.7]
    Lambdas = [0.0,0.01,0.1,0.2,0.3,0.4]
    Alphas = [0.4,0.5,0.6,0.7]
    runs = 10
    acc = np.zeros((runs,),dtype=np.float64)
    exRuns = 30
    results = dict()
    f = File('../Datasets/abalone.h5','r')
    X = f.get('data')[:,:]
    lg = f.get('labels')[:]
    f.close()
    k = 3
    # Without K-means budget
    for budget in budgets:
        print "Tuning with budget size = {}".format(budget)
        finalAcc = np.zeros((exRuns,),dtype=np.float64)
        finalTime = np.zeros((exRuns,),dtype=np.float64)
        accT = np.zeros((len(Gammas),),dtype=np.float64)
        for i in xrange(len(Gammas)):
            print "Tuning Gamma {} of {}".format(i+1,len(Gammas))
            Gamma = Gammas[i]
            for j in xrange(runs):
                ok = OKMF(budget,k,50,4,Gamma,0.1,0.2,'rbf')
                try:
                    ok.fit(X)
                    lf = np.argmax(ok.H,axis=0)
                    acc[j] = accuracy(lf,lg)
                except np.linalg.LinAlgError as er:
                    print er.message
                    acc[j] = 0.0
            accT[i] = np.average(acc)
        Gamma = Gammas[np.argmax(accT)]
        accT = np.zeros((len(Lambdas),),dtype=np.float64)
        for i in xrange(len(Lambdas)):
            print "Tuning Lambda {} of {}".format(i+1,len(Lambdas))
            Lambda = Lambdas[i]
            for j in xrange(runs):
                ok = OKMF(budget,k,50,4,Gamma,Lambda,0.2,'rbf')
                try:
                    ok.fit(X)
                    lf = np.argmax(ok.H,axis=0)
                    acc[j] = accuracy(lf,lg)
                except np.linalg.LinAlgError as er:
                    print er.message
                    acc[j] = 0.0
            accT[i] = np.average(acc)
        Lambda = Lambdas[np.argmax(accT)]
        accT = np.zeros((len(Alphas),),dtype=np.float64)
        for i in xrange(len(Alphas)):
            print "Tuning Alpha {} of {}".format(i+1,len(Alphas))
            Alpha = Alphas[i]
            for j in xrange(runs):
                ok = OKMF(budget,k,50,4,Gamma,Lambda,Alpha,'rbf')
                try:
                    ok.fit(X)
                    lf = np.argmax(ok.H,axis=0)
                    acc[j] = accuracy(lf,lg)
                except np.linalg.LinAlgError as er:
                    print er.message
                    acc[j] = 0.0
            accT[i] = np.average(acc)
        Alpha = Alphas[np.argmax(accT)]
        accT = np.zeros((len(Sigmas),),dtype=np.float64)
        for i in xrange(len(Sigmas)):
            print "Tuning Sigma {} of {}".format(i+1,len(Sigmas))
            Sigma = Sigmas[i]
            sigmaR = 1.0 / ((2.0 ** Sigma) ** 2.0)
            for j in xrange(runs):
                ok = OKMF(budget,k,50,4,Gamma,Lambda,Alpha,'rbf',gamma=sigmaR)
                try:
                    ok.fit(X)
                    lf = np.argmax(ok.H,axis=0)
                    acc[j] = accuracy(lf,lg)
                except np.linalg.LinAlgError as er:
                    print er.message
                    acc[j] = 0.0
            accT[i] = np.average(acc)
        Sigma = Sigmas[np.argmax(accT)]
        sigmaR = 1.0 / ((2.0 ** Sigma) ** 2.0)
        for i in xrange(exRuns):
            ok = OKMF(budget,k,50,4,Gamma,Lambda,Alpha,'rbf',gamma=sigmaR)
            try:
                t0 = clock()
                ok.fit(X)
                lf = np.argmax(ok.H,axis=0)
                t1 = clock()
                finalAcc[i] = accuracy(lf,lg)
                finalTime[i] = t1 - t0
            except np.linalg.LinAlgError as er:
                print er.message
                finalAcc[i] = 0
                finalTime[i] = float('inf')
        results[str(budget)]=finalAcc,finalTime
    return results

if __name__ == '__main__':
    d = experimentsAbalone()
    outputFile = open('experimentsAbalone.pkl','w')
    pickle.dump(d,outputFile)
    outputFile.close()