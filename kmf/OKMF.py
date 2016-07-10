# -*- coding: utf-8 -*-
"""
@author: Esteban Paez-Torres
"""

import numpy as np
import numpy.linalg as la
from numpy.random import permutation
from numpy import dot
from sklearn.metrics import pairwise_kernels as K
from KKmeans import KKmeans as kk
import kernel as Kernel

class OKMF:
    """Online Kernel Matrix Factorization.
    
    Parameters
    ----------
    
    budgetSize : int
        The size of the factorization budget.
    latentTopics : int
        The amount of latent topics.
    minibatchSize : int
        The number of elements in the minibatch
    epochs : int
        The number of epochs to train
    Gamma : double
        The initial learning rate.
    Lambda : double
        The regularization parameter for W.
    Alpha : double
        The regularization parameter for H.
    metric : string
        The kernel metric to be used.
    kwds : 
        Optional parameters for the kernel function.
    
    Attributes
    ----------
    
    W : ndarray
        Budget weights matrix.
    H : ndarray
        Latent space.
    Budget : ndarray
        Budget matrix.
    trainErrors : list
        A list containing the reconstruction errors for initial values
        and every epoch.
    """
    def __init__(self,budgetSize,latentTopics,minibatchSize,epochs,Gamma,
                 Lambda,Alpha,metric,**kwds):
        self._budgetSize = budgetSize
        self._latentTopics = latentTopics
        self._minibatchSize = minibatchSize
        self._epochs = epochs
        self._Gamma = Gamma
        self._Lambda = Lambda
        self._Alpha = Alpha
        self._metric = metric
        self._kwds = kwds
        self.W = None
        self.H = None
        self.Budget = None
        self._KB = None
        self._X = None
        self.trainErrors = list()
        self.validationErrors = list()
    
    def fit(self,X,V=None,calculateErrors=False,Budget=None):
        """
        Train the model with a set X of samples.
        
        Parameters
        ----------
        
        X : ndarray
            A matrix representing all the data to factorize.
        V : ndarray
            A matrix representing data to validate.
        calculateErrors : boolean
            If True, epoch error is calculated.
        Budget : ndarray
            Default budget
        """
        self._X = X
        if not(Budget is None):
            self.Budget = Budget
        else:
            indices = permutation(self._X.shape[0])[:self._budgetSize]
            indices.sort()
            self.Budget = X[indices,:]
        #self._KB = K(self.Budget,None,metric=self._metric,**self._kwds)#change
        Ks = Kernel.softKernel(self._X,0.5, 1, 30)#Kernel for persons AJP
        Kp = K(self.Budget,None,metric=self._metric,**self._kwds)#K(x,None,self._metric,**self._kwds)#Kernel for poses AJP
        Ki = K(self.Budget,None,'linear')#Kernel for illumination AJP
        #self._KB = K(self.Budget,None,metric=self._metric,**self._kwds)
        self._KB = dot(dot(Kp,Ki),Ks) #AJP
        self.W = self._initW()
        iteration = 0
        if calculateErrors:
            self.trainErrors.append(self.Error(self._X))
            if V != None:
                self.validationErrors.append(self.Error(V))
        for i in xrange(self._epochs):
            indices = permutation(self._X.shape[0])
            while len(indices) > 0:
                batchSize = min(self._minibatchSize,len(indices))
                batch = indices[:batchSize]
                batch.sort()
                indices = indices[batchSize:]
                x = self._X[batch,:]
                self._nextW(x,iteration)
                iteration += 1
            if calculateErrors:
                self.trainErrors.append(self.Error(self._X))
                if V != None:
                    self.validationErrors.append(self.Error(V))
        self.H = self.predictH(self._X)
        self.trainErrors = np.array(self.trainErrors)
        self.validationErrors = np.array(self.validationErrors)
        #return self.W,self.H,self.Budget
    
    def _nextGamma(self,t):
        return self._Gamma / (1 + (self._Gamma * self._Lambda * t))
    
    def _initW(self):
        return np.random.rand(self._budgetSize,self._latentTopics)
        #return kk(self._KB,self._latentTopics,5)[0].T
    
    def _nextKxi(self,x):
        #return K(self.Budget,x,self._metric,**self._kwds)#change
        #Kp = K(self.Budget,x,self._metric,**self._kwds)#K(x,None,self._metric,**self._kwds)#Kernel for poses AJP
        #Ki = K(self.Budget,x,'linear')#Kernel for illumination AJP
        return Kernel.prodKernel(self.Budget,x,self._metric,**self._kwds)
    
    def _nextH(self,kxi):
        A = dot(dot(self.W.T,self._KB),self.W)
        A += self._Alpha*np.eye(self._latentTopics)
        b = dot(self.W.T,kxi)
        try:
            return la.solve(A,b)
        except la.LinAlgError:
            print 'Using lstsq'
            return la.lstsq(A,b)[0]
    
    def _gradient(self,kxi):
        G = dot(dot(dot(self._KB,self.W),self.H),self.H.T)
        G -= dot(kxi,self.H.T)
        G += self._Lambda *  self.W
        return G
    
    def _nextW(self,x,t):
        kxi = self._nextKxi(x)
        self.H = self._nextH(kxi)
        G = self._gradient(kxi)
        gamma = self._nextGamma(t)
#        self.W = self.W - ((gamma/self._minibatchSize)*G)
        self.W = self.W - (gamma*G)
        self.W = self.W.clip(0)
    
    def predictH(self,Xp):
        """
        Calculate the latent representation of a given data set using the
        learned representation.
        
        Parameters
        ----------
        Xp : ndarray
            A matrix representing the data whose latent representation
            will be calculated.
        
        Returns
        -------
        H : ndarray
            The latent representation for the given data set.
        """
        kx = self._nextKxi(Xp)
        H = self._nextH(kx)
        return H
    
    def SampleLoss(self,x):
        """
        Calculates the objective function for a given sample x.
        
        Parameters
        ----------
        x : ndarray
            The sample whose objective function will be calculated.
        
        Returns
        -------
        sampleError : float
            The objective function evaluated for the given sample.
        """
        #E1 = np.trace(K(x,None,self._metric,**self._kwds))
        Kp = K(x,None,self._metric,**self._kwds)#K(x,None,self._metric,**self._kwds)#Kernel for poses AJP
        Ki = K(x,None,'linear')#Kernel for illumination AJP        
        E1 = np.trace(dot(Kp,Ki))#AJP        
        kxi = self._nextKxi(x)
        h = self._nextH(kxi)
        E2 = np.trace(dot(h.T,dot(self.W.T,kxi)))
        E3 = np.trace(dot(dot(kxi.T,self.W),h))
        E4 = np.trace(dot(dot(dot(h.T,dot(self.W.T,self._KB)),self.W),h))
        E5 = self._Lambda * np.trace(dot(self.W.T,self.W))
        E6 = self._Alpha * np.trace(dot(h.T,h))
        sampleError = (E1 - E2 - E3 + E4 + E5 + E6) / 2.0
        return sampleError
    
    def Error(self,X):
        """
        Calculate the objective function for a given set of data using the
        factorization learned.
        
        Parameters
        ----------
        X : ndarray
            A matrix representing the data whose objective will be calculated.
        
        Returns
        -------
        error : float
            The objective function evaluated for the gven data.
        """
        error = 0.0
        indices = range(X.shape[0])
        while len(indices) > 0:
            batchSize = min(self._minibatchSize,len(indices))
            batch = indices[:batchSize]
            batch.sort()
            indices = indices[batchSize:]
            x = X[batch,:]
            sampleError = self.SampleLoss(x)
            error += sampleError
        return error
    
    def KXx(self,indx):
        return self._KB[indx,:]
