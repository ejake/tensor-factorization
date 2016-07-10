# Kernel Non-negative Tensor Factorization
"""
Author: Andres Jaque (rajaquep@unal.edu.co)
Based on Andres Paez okmf algorithm
Created: Apr, 2016
Modified: 
"""
import numpy as np
import numpy.linalg as la
from numpy.random import permutation
from sklearn.metrics import pairwise_kernels as K

class Kntf:
    def __init__(self, latentTopics, epochs, Gamma, Lambda, Alpha, metric, **kwds):
        self._iterations = epochs
        self.W = None
        self.H = None
        self._latentTopics = latentTopics
        self._Gamma = Gamma
        self._Lambda = Lambda
        self._Alpha = Alpha
        self._metric = metric
        self._kwds = kwds
        self._KB = None
        self._X = None
        self._T = None
        self._Ks = None #Soft kernel for subjects
        self.trainErrors = list()
        self.validationErrors = list()
        self._objects = None
    
        
    def fit(self,T,V=None,calculateErrors=False):
        """
        Train the model, i.e. find W and H
        T : ndarray
            A tensor representing all the data to factorize.
        V : ndarray
            A matrix representing data to validate.
        """
        self._T = T
        self._X = self.Tensor2Matrix(T)
        
        if self._metric != 'precomputed':
            self._KB = K(self._X,None,metric=self._metric,**self._kwds)
        else:
            self._Ks = self.SofKernelSubj(T,None)
            self._KB = self.KernelFaces(T,None,self._Ks)
            
        self.W = self._initW()
        iteration = 0
        if calculateErrors:
            self.trainErrors.append(self.Error(self._X))
            if V != None:
                self.validationErrors.append(self.Error(V))
        for i in xrange(self._iterations):
            indices = permutation(self._X.shape[0])
            while len(indices) > 0:
                batchSize = len(indices)
                batch = indices[:batchSize]
                batch.sort()
                indices = indices[batchSize:]
                x = self._X[batch,:]
                self._nextW(x,iteration,batch)
                iteration += 1
            if calculateErrors:
                self.trainErrors.append(self.Error(self._X))
                if V != None:
                    self.validationErrors.append(self.Error(V))
        self.H = self.predictH(self._X)
        self.trainErrors = np.array(self.trainErrors)
        self.validationErrors = np.array(self.validationErrors)
    
    def Tensor2Matrix(self, T):
        """
        T : ndarray
            Tensor
        """
        size_tensor = T.shape
        self._objects = 1
        for i in range(len(size_tensor)-1):
            self._objects *= size_tensor[i]
            
        return np.reshape(T,(self._objects,size_tensor[-1]))
    
    def KernelFaces(self,X,y=None,Ks=None):
        """
        X : ndarray
            Tensor
        """
        if y==None:
            if len(X.shape)==1:#T is a sample/object (vector) k(x,x)
                #Kernel for poses
                Kp = K(X,None,'rbf',**self._kwds)#K(x,None,self._metric,**self._kwds)
                #Kernel for illumination
                Ki = K(X,None,'linear')
                Kr = Kp*Ki
            else:#T is a high-order tensor k(X,X)
                #Kernel for poses
                Kp = K(self._X,None,'rbf',**self._kwds)#K(x,None,self._metric,**self._kwds)
                #Kernel for illumination
                Ki = K(self._X,None,'linear')
                if Ks!=None:
                    Kr = np.dot(Ks, np.dot(Kp,Ki))
                else:
                    Kr = np.dot(Kp,Ki)
        else:            
            #Kernel for poses
            Kp = K(self._X,y,'rbf',**self._kwds)#K(x,None,self._metric,**self._kwds)
            #Kernel for illumination
            Ki = K(self._X,y,'linear')
            if Ks!=None:
                Kr = np.dot(Ks, np.dot(Kp,Ki))
            else:
                Kr = np.dot(Kp,Ki)
        return Kr
    
    def SofKernelSubj(self, T, indx = None):
        val_diff = 0.5
        val_eq = 1
        if indx == None:
            Ks = np.ones((self._objects,self._objects))            
            Ks = Ks*val_diff
            for i in range(T.shape[0]):
                #compute Soft kernel for subjects
                Ks[i:(i+1)*(T.shape[1]*T.shape[2]),i:(i+1)*(T.shape[1]*T.shape[2])] = val_eq
        else:
            Ks = np.ones(self._objects)
            Ks = Ks*val_diff
            #print indx
            #Ks[int(T.shape[1]*T.shape[2]*(np.ceil(indx/T.shape[1]*T.shape[2]*1.0)-1)):int(T.shape[1]*T.shape[2]*(np.ceil(indx/T.shape[1]*T.shape[2]*1.0)))]=1
        return Ks
    
    
    def _initW(self):#Initialize W matrix
        return np.random.rand(self._objects,self._latentTopics)#as random matrix
        
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
            batchSize = len(indices)
            batch = indices[:batchSize]
            batch.sort()
            indices = indices[batchSize:]
            x = X[batch,:]
            sampleError = self.SampleLoss(x,batch)
            error += sampleError
        return error
    
    def SampleLoss(self,x,indx):
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
        if self._metric != 'precomputed':
            E1 = np.trace(K(x,None,self._metric,**self._kwds))
        else:
            E1 = np.trace(self.KernelFaces(x))
        kxi = self._nextKxi(x, indx)
        h = self._nextH(kxi)
        E2 = np.trace(np.dot(h.T,np.dot(self.W.T,kxi)))
        E3 = np.trace(np.dot(np.dot(kxi.T,self.W),h))
        E4 = np.trace(np.dot(np.dot(np.dot(h.T,np.dot(self.W.T,self._KB)),self.W),h))
        E5 = self._Lambda * np.trace(np.dot(self.W.T,self.W))
        E6 = self._Alpha * np.trace(np.dot(h.T,h))
        sampleError = (E1 - E2 - E3 + E4 + E5 + E6) / 2.0
        return sampleError
    
    def _nextKxi(self,x, indx=None):
        if self._metric != 'precomputed':
            return K(self.Budget,x,self._metric,**self._kwds)
        else:
            Ks=None
            if indx!=None:
                Ks  = self.SofKernelSubj(self._T,indx)
            return self.KernelFaces(self._X,x,Ks)
        
    
    def _nextH(self,kxi):
        A = np.dot(np.dot(self.W.T,self._KB),self.W)
        A += self._Alpha*np.eye(self._latentTopics)
        b = np.dot(self.W.T,kxi)
        try:
            return la.solve(A,b)#Solve a linear equations system
        except la.LinAlgError:
            print 'Using lstsq'
            return la.lstsq(A,b)[0]
   
    def _nextW(self,x,t, indx=None):
        kxi = self._nextKxi(x, indx)
        self.H = self._nextH(kxi)
        G = self._gradient(kxi)
        gamma = self._nextGamma(t)
#        self.W = self.W - ((gamma/self._minibatchSize)*G)
        self.W = self.W - (gamma*G)
        self.W = self.W.clip(0)
    
    def _gradient(self,kxi):
        G = np.dot(np.dot(np.dot(self._KB,self.W),self.H),self.H.T)
        G -= np.dot(kxi,self.H.T)
        G += self._Lambda *  self.W
        return G



