# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:46:26 2015

@author: aepaezt
"""

from numbapro import cuda
import math
from numbapro.cudalib import cublas,curand
import numpy as np
from time import time
from KKmeans import KKmeans

@cuda.jit('void(float64[:,:])')
def fillWithZerosKernel(matrix):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    i = tx + bx * bw
    j = ty + by * bh
    if i >= matrix.shape[0] or j >= matrix.shape[1]:
        return
    matrix[i,j] = 0.0

@cuda.jit('void(float64[:,:],float64[:,:],float64,float64[:,:])')
def gaussianKernelKernel(X,Y,sigma,K):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    i = tx + bx * bw
    j = ty + by * bh
    if i >= X.shape[0] or j >= Y.shape[0]:
        return
    K[i,j] = 0.0
    for l in range(X.shape[1]):
        val = X[i,l] - Y[j,l]
        K[i,j] += math.pow(val,2.0)
    gamma = -1.0 / (2.0 * sigma * sigma)
    K[i,j] = math.exp(gamma * K[i,j])

@cuda.jit('void(float64[:,:],int32[:],float64[:,:])')
def initBudgetKernel(X,permutation,Budget):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    i = tx + bx * bw
    j = ty + by * bh
    if i >= Budget.shape[0] or j >= X.shape[1]:
        return
    pi = permutation[i]
    Budget[i,j] = X[pi,j]

@cuda.jit('void(float64[:,:],int32[:],int32,float64[:,:])')
def loadBatchKernel(KX,permutation,iteration,kxi):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    i = tx + bx * bw
    j = ty + by * bh
    if i >= kxi.shape[0] or j >= kxi.shape[1]:
        return
    pl = kxi.shape[1] * iteration
    pj = permutation[pl + j]
    kxi[i,j] = KX[i,pj]

def fillWithZeros(matrix):
    threads = 32,32,1
    blocks = (matrix.shape[0] + 32) / 32,(matrix.shape[1] + 32) / 32,1
    fillWithZerosKernel[blocks,threads](matrix)

def gaussianKernel(X,Y,sigma,K):
    threads = 32,32,1
    blocks = (X.shape[0] + 32) / 32,(Y.shape[0] + 32) / 32,1
    gaussianKernelKernel[blocks,threads](X,Y,sigma,K)

def linearKernel(X,Y,Blas,K):
    fillWithZeros(K)
    Blas.gemm('N','T',X.shape[0],Y.shape[0],X.shape[1],1.0,X,Y,1.0,K)

def initBudget(X,permutation,Budget):
    threads = 32,32,1
    blocks = (Budget.shape[0] + 32) / 32,(Budget.shape[1] + 32) / 32,1
    initBudgetKernel[blocks,threads](X,permutation,Budget)

def loadBatch(KX,permutation,iteration,kxi):
    threads = 32,32,1
    blocks = (kxi.shape[0] + 32) / 32,(kxi.shape[1] + 32) / 32,1
    loadBatchKernel[blocks,threads](KX,permutation,iteration,kxi)
    
def getPermutation(samples,miniBatchSize):
    permutation = np.random.permutation(samples)
    if (samples % miniBatchSize) != 0:
        dif = miniBatchSize - (samples % miniBatchSize)
        permutation = np.hstack((permutation,permutation[:dif]))
    return permutation

class cudaOKMF:
    def __init__(self,budgetSize,latentTopics,miniBatchSize,epochs,Gamma,Lambda,Alpha,metric='rbf',sigma=1.0):
        """
        OKMF class: OKMF is a method to perform a matrix factorization in a 
        feature space.
        
        Parameters
        ----------
        
        budgetSize : int
            Budget size.
        latentTopics: int
            Latent topics.
        miniBatchSize : int
            Size of minibatch.
        epochs : int
            Number of epochs.
        Gamma : float
            Gamma parameter
        Lambda : float
            Lambda parameter
        Alpha: float
            Alpha parameter
        metric : string
            Type of kernel. Default rbf
        sigma : float
            RBF kernel sigma parameter. Default 1.0.
        """
        self.budgetSize = budgetSize
        self.latentTopics = latentTopics
        self.miniBatchSize = miniBatchSize
        self.epochs = epochs
        self.Gamma = Gamma
        self.Lambda = Lambda
        self.Alpha = Alpha
        self.metric = metric
        self.sigma = sigma
        self.W = None
        self.h = cuda.device_array((latentTopics,miniBatchSize),dtype=np.float64,order='F')
        self.KB = cuda.device_array((budgetSize,budgetSize),dtype=np.float64,order='F')
        self.kxi = cuda.device_array((budgetSize,miniBatchSize),dtype=np.float64,order='F')
        self.Blas = cublas.Blas()
        self.X = None
        self.Budget = None
        self.permutation = None
        self.kx = None
        self.Wkx = None
        self.H = None
        self.KBW = cuda.device_array((budgetSize,latentTopics),dtype=np.float64,order='F')
        self.KBWh = cuda.device_array((budgetSize,miniBatchSize),dtype=np.float64,order='F')
        self.KBWhh = cuda.device_array((budgetSize,latentTopics),dtype=np.float64,order='F')
        self.grad = cuda.device_array((budgetSize,latentTopics),dtype=np.float64,order='F')
        self.kxih = cuda.device_array((budgetSize,latentTopics),dtype=np.float64,order='F')
        self.WKBW = cuda.device_array((latentTopics,latentTopics),dtype=np.float64,order='F')
        self.Wkxi = cuda.device_array((latentTopics,miniBatchSize),dtype=np.float64,order='F')
        eyeAlpha = np.eye(latentTopics) * Alpha
        self.eyeAlpha = cuda.to_device(eyeAlpha.astype(np.float64,order='F'))
        
    def fit(self,X,Budget=None,W=None):
        self.X = cuda.to_device(X.astype(np.float64,order='F'))
        self.Budget = cuda.device_array((self.budgetSize,self.X.shape[1]),dtype=np.float64,order='F')
        self.kx = cuda.device_array((self.budgetSize,self.X.shape[0]),dtype=np.float64,order='F')
        self.Wkx = cuda.device_array((self.latentTopics,self.X.shape[0]),dtype=np.float64,order='F')
        self.H = cuda.device_array((self.latentTopics,self.X.shape[0]),dtype=np.float64,order='F')
        if Budget is None:
            permutation = np.random.permutation(self.X.shape[0])
            self.permutation = cuda.to_device(permutation)
            initBudget(self.X,self.permutation,self.Budget)
        else:
            self.Budget = cuda.to_device(Budget.astype(np.float64,order='F'))
        self.calculateKB()
        self.calculateKX()
        if W is None:
            self.initW()
        else:
            self.W = cuda.to_device(W.astype(np.float64,order='F'))
        self.t = 0
        for i in xrange(self.epochs):
            print "Epoch " + str(i)
            samples,features = self.X.shape
            permutation = getPermutation(samples,self.miniBatchSize)
            self.permutation = cuda.to_device(permutation)
            for j in xrange((samples + self.miniBatchSize) / self.miniBatchSize):
                loadBatch(self.kx,self.permutation,j,self.kxi)
                self.nextW()
                self.t += 1
        self.predictH()

    def gradient(self):
        fillWithZeros(self.KBWh)
        self.Blas.gemm('N','N',self.budgetSize,self.miniBatchSize,self.latentTopics,1.0,self.KBW,self.h,1.0,self.KBWh)
        fillWithZeros(self.KBWhh)
        self.Blas.gemm('N','T',self.budgetSize,self.latentTopics,self.miniBatchSize,1.0,self.KBWh,self.h,1.0,self.KBWhh)
        fillWithZeros(self.kxih)
        self.Blas.gemm('N','T',self.budgetSize,self.latentTopics,self.miniBatchSize,1.0,self.kxi,self.h,1.0,self.kxih)
        self.Blas.geam('N','N',self.budgetSize,self.latentTopics,1.0,self.KBWhh,-1.0,self.kxih,self.grad)
        self.Blas.geam('N','N',self.budgetSize,self.latentTopics,1.0,self.grad,self.Lambda,self.W,self.grad)
    
    def nextH(self):
        fillWithZeros(self.WKBW)
        self.Blas.gemm('T','N',self.latentTopics,self.latentTopics,self.budgetSize,1.0,self.KBW,self.W,1.0,self.WKBW)
        self.Blas.geam('N','N',self.latentTopics,self.latentTopics,1.0,self.WKBW,1.0,self.eyeAlpha,self.WKBW)
        WKBW = self.WKBW.copy_to_host()
        WKBW = np.linalg.inv(WKBW).astype(np.float64,order='F')
        fillWithZeros(self.Wkxi)
        self.Blas.gemm('T','N',self.latentTopics,self.miniBatchSize,self.budgetSize,1.0,self.W,self.kxi,1.0,self.Wkxi)
        self.WKBW = cuda.to_device(WKBW)
        fillWithZeros(self.h)
        self.Blas.gemm('N','N',self.latentTopics,self.miniBatchSize,self.latentTopics,1.0,self.WKBW,self.Wkxi,1.0,self.h)

    def calculateKX(self):
        if self.metric == 'rbf':
            gaussianKernel(self.Budget,self.X,self.sigma,self.kx)
        elif self.metric == 'linear':
            linearKernel(self.Budget,self.X,self.Blas,self.kx)

    def calculateKB(self):
        if self.metric == 'rbf':
            gaussianKernel(self.Budget,self.Budget,self.sigma,self.KB)
        elif self.metric == 'linear':
            linearKernel(self.Budget,self.Budget,self.Blas,self.KB)
            
    def nextKBW(self):
        fillWithZeros(self.KBW)
        self.Blas.gemm('N','N',self.budgetSize,self.latentTopics,self.budgetSize,1.0,self.KB,self.W,1.0,self.KBW)

    def nextW(self):
        self.nextKBW()
        self.nextH()
        self.gradient()
        #GammaT = self.nextGamma() / (-1.0 * self.miniBatchSize)
        GammaT = self.nextGamma() / (-1.0)
        self.Blas.geam('N','N',self.budgetSize,self.latentTopics,1.0,self.W,GammaT,self.grad,self.W)
        # TODO Clip W

    def nextGamma(self):
        return self.Gamma / (1 + (self.Gamma * self.Lambda * self.t))
        
    def initW(self):
        """self.W = cuda.device_array((self.budgetSize*self.latentTopics,),dtype=np.float64,order='F')
        generator = curand.PRNG()
        generator.seed = long(time())
        generator.uniform(self.W)
        self.W = self.W.reshape((self.budgetSize,self.latentTopics),order='F')
        """
        KB = self.KB.copy_to_host()
        self.W = cuda.to_device(KKmeans(KB,self.latentTopics,5)[0].T.astype(np.float64,order='F'))
    
    def predictH(self):
        self.nextKBW()
        fillWithZeros(self.WKBW)
        self.Blas.gemm('T','N',self.latentTopics,self.latentTopics,self.budgetSize,1.0,self.KBW,self.W,1.0,self.WKBW)
        self.Blas.geam('N','N',self.latentTopics,self.latentTopics,1.0,self.WKBW,1.0,self.eyeAlpha,self.WKBW)
        WKBW = self.WKBW.copy_to_host()
        WKBW = np.linalg.inv(WKBW).astype(np.float64,order='F')
        fillWithZeros(self.Wkx)
        self.Blas.gemm('T','N',self.latentTopics,self.X.shape[0],self.budgetSize,1.0,self.W,self.kx,1.0,self.Wkx)
        self.WKBW = cuda.to_device(WKBW)
        fillWithZeros(self.H)
        self.Blas.gemm('N','N',self.latentTopics,self.X.shape[0],self.latentTopics,1.0,self.WKBW,self.Wkx,1.0,self.H)
