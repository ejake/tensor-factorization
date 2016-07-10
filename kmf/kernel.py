"""
Compute kernel function
Return Gram Matrix (Kernel matrix)
"""
import numpy as np
from numpy import dot
from sklearn.metrics import pairwise_kernels as K
from sklearn.metrics import pairwise_distances as D

def softKernel(X, valDiff, valEq, dim):
    num_objects = X.shape[0]
    val_diff = valDiff
    val_eq = valEq
    Ks = np.ones((num_objects,num_objects))            
    Ks = Ks*val_diff
    for i in range(dim):
        #compute Soft kernel for subjects
        Ks[i:(i+1)*(num_objects/dim),i:(i+1)*(num_objects/dim)] = val_eq            
    
    return Ks

def prodKernel(X,xi,metric,**kwds):
    Kp = K(X,None,metric,**kwds)#K(x,None,self._metric,**self._kwds)#Kernel for poses AJP
    Ki = K(X,xi,'linear')#Kernel for illumination AJP
    Ks = softKernel(X, 0.5, 1, 30)
    return dot(Ks,dot(Kp,Ki))

def prodKernel2(X,xi,metric,**kwds):
    Kp = K(X,None,metric,**kwds)#K(x,None,self._metric,**self._kwds)#Kernel for poses AJP
    Ki = K(X,xi,'linear')#Kernel for illumination AJP    
    return dot(Ks,dot(Kp,Ki))

def gaussian(X, gamma):
    """
    Return Gran matrix by a Gaussian-Kernel
    """
    return K

def preimageGaussian(X, sigma, alpha):
    """
    X: (numpy array nxm) Input data
    sigma: scalar
    alpha: numpy array (1xn)
    method: 
            fpi: Fixed-Point iterations
    """
    max_iter = 10 #halt criterium
    epsilon = 0.0001 #halt criterium
    
    #initialization
    x_pre = np.random.rand(1,X.shape[1])    
        
    for it in range(1,max_iter):
        sum_num = np.zeros((1,X.shape[1]))
        sum_den = 0
        #compute gaussian kernel
        kx = K(X,x_pre,'rbf',gamma = (2*sigma)**-2)
        for i in range(1,X.shape[0]):
            fact1 = alpha[0,i]*kx[i]
            sum_den += fact1
            sum_num += fact1*X[i,:]
            
        x_pre = sum_num/sum_den                     
        
        
    return x_pre

def preimageTensorKLinear(phi, K, X):
    return 0

def preimageTensorKLinearGaussian(phi, X, method='gdt', gamma = 0.01):
    """
    X: (numpy array) Input data
    method: gdt: Gradient Descent Technique
            fpi: Fixed-Point iterations
    """
    max_iter = 10 #halt criterium
    epsilon = 0.0001 #halt criterium
    
    #initialization
    x_pre = np.random.rand(1,X.shape[1])    
    
    for it in range(1,max_iter):
        if method=='gdt':
            xt = x_pre - gamma*(1)
            #X, Y = check_pairwise_arrays(X, Y)
            #if gamma is None:
            #    gamma = 1.0 / X.shape[1]

            #K = euclidean_distances(X, Y, squared=True)
            #K *= -gamma
            #np.exp(K, K)    # exponentiate K in-place
            #return K
        if method == 'fpi':
            xt = x_pre - gamma*(1)
        
        if abs(x_pre-x/x) < epsilon:
            break
            
    return 0
