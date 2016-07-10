# -*- coding: utf-8 -*-
"""
Created on Sat Nov 08 16:12:34 2014

@author: Esteban
"""

from OKMF import OKMF
import numpy as np
from numpy import sqrt
import h5py
import pylab as pl
from time import clock
from accuracy import accuracy
from sklearn.cross_validation import KFold
import online_kmeans as okm

def main4():
    f = h5py.File('MNIST6000.h5','r')
    X = f.get('/train/data')[:,:] / 255.
    Xt = f.get('/test/data')[:,:] / 255.
    kf = KFold(X.shape[0],n_folds=4,shuffle=True)
    f = open('Results_rbf.csv','w')
    f.write('sigma,Budget,Gamma,Lambda,Alpha,tr0,tr1,tr2,ts0,ts1,ts2,time\n')
    c = 0.0
    for exp in [-9,-5,-1,0,1,5,9]:
        sigma = 2**exp
        for budget in [10,100,1000,2000]:
            for Gamma in [0.1,0.8]:
                for Lambda in [0.1,0.3]:
                    for Alpha in [0.3,0.6]:
                        suma0 = 0.0
                        suma1 = 0.0
                        suma2 = 0.0
                        suma3 = 0.0
                        suma4 = 0.0
                        suma5 = 0.0
                        time = 0.0
                        c += 1
                        print 'rbf',(c / 224.0)
                        for train,test in kf:
                            ok = OKMF(budget,10,100,2,Gamma,Lambda,Alpha,'rbf',
                                      gamma=sigma)
                            t0 = clock()
                            ok.fit(X,Xt,True)
                            time += clock() - t0
                            suma0 += ok.trainErrors[0]
                            suma1 += ok.trainErrors[1]
                            suma2 += ok.trainErrors[2]
                            suma3 += ok.validationErrors[0]
                            suma4 += ok.validationErrors[1]
                            suma5 += ok.validationErrors[2]
                        suma0/=4.0
                        suma1/=4.0
                        suma2/=4.0
                        suma3/=4.0
                        suma4/=4.0
                        suma5/=4.0
                        time/=4.0
                        val = (sigma,budget,Gamma,Lambda,Alpha)
                        val += (suma0,suma1,suma2,suma3,suma4,suma5,time)
                        s = '{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(*val)
                        f.write(s)
    f.close()

def main5():
    f = h5py.File('MNIST6000.h5','r')
    X = f.get('/train/data')[:,:] / 255.
    Xt = f.get('/test/data')[:,:] / 255.
    kf = KFold(X.shape[0],n_folds=4,shuffle=True)
    f = open('Results_linear.csv','w')
    f.write('sigma,Budget,Gamma,Lambda,Alpha,tr0,tr1,tr2,ts0,ts1,ts2,time\n')
    c = 0.0
    for budget in [10,100,1000,2000]:
        for Gamma in [0.1,0.8]:
            for Lambda in [0.1,0.3]:
                for Alpha in [0.3,0.6]:
                    suma0 = 0.0
                    suma1 = 0.0
                    suma2 = 0.0
                    suma3 = 0.0
                    suma4 = 0.0
                    suma5 = 0.0
                    time = 0.0
                    c += 1
                    print 'linear',(c / 32.0)
                    for train,test in kf:
                        ok = OKMF(budget,10,100,2,Gamma,Lambda,Alpha,'linear')
                        t0 = clock()
                        ok.fit(X,Xt,True)
                        time += clock() - t0
                        suma0 += ok.trainErrors[0]
                        suma1 += ok.trainErrors[1]
                        suma2 += ok.trainErrors[2]
                        suma3 += ok.validationErrors[0]
                        suma4 += ok.validationErrors[1]
                        suma5 += ok.validationErrors[2]
                    suma0/=4.0
                    suma1/=4.0
                    suma2/=4.0
                    suma3/=4.0
                    suma4/=4.0
                    suma5/=4.0
                    time/=4.0
                    val = ('NA',budget,Gamma,Lambda,Alpha)
                    val += (suma0,suma1,suma2,suma3,suma4,suma5,time)
                    s = '{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(*val)
                    f.write(s)
    f.close()

def main3():
    f = h5py.File('MNIST6000.h5','r')
    X = f.get('/train/data')[:,:] / 255.
    V = f.get('/test/data')[:,:] / 255.
    lg = f.get('/test/target')[:]
    f.close()
    acc = 0.0
    t = 0.0
    accl = list()
    for i in xrange(40):
        s = '{},{},{},{},{},{}\n'
        ok = OKMF(2000,10,100,2,0.8,0.1,0.3,'rbf',gamma=(2.0**-9-0))
        t0 = clock()
        ok.fit(X)
        t1 = clock()
        #print 'OKMF',i
        lp = np.argmax(ok.predictH(V).T,axis=1)
        aux = accuracy(lp,lg,10)
        accl.append(aux)
        print 'OKMF',i,aux
#        print aux
        acc += aux
#        print t1-t0
        t += t1-t0
    f = open('ResultsCRBF.csv','w')
    f.write('Budget,Gamma,Lambda,Alpha,Acc,time\n')
    s = '{},{},{},{},{},{}\n'
    print acc/40.0
    val = (1000,0.01,0.01,0.8) + (acc/40.0,t/40.0)
    f.write(s.format(*val))
    f.close()
    f = open('series2000K.txt','w')
    f.write(str(accl)+'\n')
    f.close()
    acc = 0.0
    accl = list()
    """
    for i in xrange(40):
        F,clusters = okm.OnlineKmeans(X,10,2)
        print 'O kmeans',i
        lp = okm.predict(V,F)
        aux = accuracy(lp,lg,10)
        accl.append(aux)
        acc += aux
    f = open('series.txt','a')
    f.write(str(accl)+'\n')
    f.close()
    """

def main2():
    f = h5py.File('MNIST6000.h5','r')
    X = f.get('/train/data')[:,:] / 255.
    V = f.get('/test/data')[:,:] / 255.
    f.close()
    f = open('Results.csv','w')
    f.write('Budget,Gamma,Lambda,Alpha,0,1,2,3,4,0,1,2,3,4,time\n')
    c = 0.0
    for Budget in [10,100,1000,2000,6000]:
        for Gamma in [0.1,0.5,0.8]:
            for Lambda in [0.1,0.5,0.8]:
                for Alpha in [0.1,0.5,0.8]:
                    s = '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'
                    ok = OKMF(Budget,10,100,4,Gamma,Lambda,Alpha,'linear')
                    t0 = clock()
                    ok.fit(X,V,True)
                    t1 = clock()
                    c += 1.0
                    print 'Linear',c / 320
                    val = (Budget,Gamma,Lambda,Alpha)
                    val += tuple(ok.trainErrors) + tuple(ok.validationErrors)
                    val += (t1-t0,)
                    f.write(s.format(*val))
    f.close()
                    

def main():
    f = h5py.File('MNIST6000.h5','r')
    X = f.get('/train/data')[:,:] / 255.
    V = f.get('/test/data')[:,:] / 255.
    ok = OKMF(500,10,100,3,0.7,0.1,0.7,'linear')
    t = clock()
    ok.fit(X,V,calculateErrors=True)
    print clock() - t
    print sqrt(ok.trainErrors)
    print sqrt(ok.validationErrors)
    pl.figure()
    pl.title('Loss vs  Epochs')
    pl.xlabel('Epochs')
    pl.ylabel('Loss')
    pl.plot(ok.trainErrors,'b',ok.validationErrors,'r')
    pl.show()
    f.close()

if __name__ == '__main__':
    main()
