"""
Compute kernel function
Return Gram Matrix (Kernel matrix)
"""
import numpy as np
from numpy import dot
from sklearn.metrics import pairwise_kernels as K
from sklearn.metrics import pairwise_distances as D

def createMatrixMod(order):#under construction
    if order == 1:#on CMU faces corresponds to subject
        eq = 1
        diff = 0.3
        poses = [175, 95, 93, 88, 10, 5, 180, 90, 80, 2, 170]
        illuminations = np.arange(1,22)
        TS = np.zeros((30,11,21))
        for i in range(30):
            for j in range(len(poses)):
                for k in range(len(illuminations)):
                    TS[i,j,k] = i
            
        M = np.reshape(TS,(30*11*21,1))
    elif order == 2:
        TL_ = np.repeat(np.array(np.arange(1,22), ndmin = 2),11,axis=0)
        TL = np.repeat(np.array(TL_,ndmin = 3),30,axis=0)
        M = np.reshape(TL,(30*11*21,1))
    elif order == 3:
        TP_ = np.repeat(np.array([175, 95, 93, 88, 10, 5, 180, 90, 80, 2, 170], ndmin = 2),21,axis=0)
        TP = np.repeat(np.array(TP_.T,ndmin = 3),30,axis=0)
        M = np.reshape(TP,(30*11*21,1))
    
    return M
        
def tensor_cmuKernel(X,sigmal,sigmap,idxs):#under construction
    ML = createMatrixMod(2)
    Kl = kernel._get_kernel(ML, ML,'rbf', gamma=(2*sigmal)**-2)
    MP = createMatrixMod(3)
    Kp = kernel._get_kernel(MP, MP,'rbf', gamma=(2*sigmap)**-2)
    Ki = kernel._get_kernel(MF_c, MF_c,'linear')
    return Kl*Kp*Ki



def softKernel(X, valDiff, valEq, dim):
    num_objects = X.shape[0]
    val_diff = valDiff
    val_eq = valEq
    Ks = np.ones((num_objects,num_objects))            
    Ks = Ks*val_diff
    for i in range(dim):
        #compute Soft kernel for subjects
        Ks[i*(num_objects/dim):(i+1)*(num_objects/dim), i*(num_objects/dim):(i+1)*(num_objects/dim)] = val_eq
    
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

def gaussian(X, xi, sigma):
    """
    Return Gran matrix by a Gaussian-Kernel
    """
    kx = K(X,xi,'rbf',gamma = (2*sigma)**-2)#compute Gaussian kernel
    return K


            #params = {"gamma": self.gamma,
#                      "degree": self.degree,
#                      "coef0": self.coef0}
def _get_kernel(X,y,kernel_func,**params):
    if kernel_func=='rbf':
        kx = K(X,y,'rbf',**params)#compute Gaussian kernel
    elif kernel_func=='linear':
        kx = K(X,y,'linear')
    elif kernel_func=='poly':
        kx = K(X,y,'poly',**params)
    elif kernel_func=='linxrbf':#Linear x Gaussian
        kx =  K(X,y,'rbf',**params) * K(X,y,'linear')
    elif kernel_func=='polyxrbf':#Polynomial x Gaussian
        #kx =  K(X,y,'rbf',**params) * K(X,y,'poly',**params)
        kx =  np.dot( np.exp(-1*np.linalg.norm(X-y)/params['gamma']) , (np.dot(X,y.T) + params['coef0'])**params['degree'])#:|    
    else:
        kx = K(X,y,'linear')
        
    return kx


def _get_kernel_deriv(x,y,kernel_func,**params):
    #kernel derivative respect to x
    if kernel_func=='rbf':
        #kx = -1*np.exp(-1*np.linalg.norm(x-y)/params['gamma'])*(x-y)#compute Gaussian kernel
        kx = np.exp(-1*np.linalg.norm(x-y)/params['gamma'])
    elif kernel_func=='linear':
        kx = y
    elif kernel_func=='poly':
        kx = params['degree']*(x*y + params['coef0'])**(params['degree']-1)
    elif kernel_func=='linxrbf':#Linear x Gaussian
        kx = y*np.exp(-1*np.linalg.norm(x-y)/params['gamma'])*(1-np.dot(x,x.T)/params['gamma']+params['coef0']/params['gamma'])
    elif kernel_func=='polyxrbf':#Polynomial x Gaussian
        kx = np.dot(x,y.T)**(params['degree']-1) * np.exp(-1*np.linalg.norm(x-y)/params['gamma']) * np.dot( np.squeeze(y.T), np.squeeze(params['degree'] - x*np.dot(x,y.T)/params['gamma']**2 + y/params['gamma']**2) )
    #elif kernel_func=='polyxrbf2':#Polynomial x Gaussian pendent
    else:
        kx = K(X,y,'linear')
        
    return kx

def _get_kernel_faces(X,y,kernel_func,**params):#under construction
    if kernel_func == 'rbfx3xlinear':
        ML = createMatrixMod(2)
        Kl = kernel._get_kernel(ML, ML,'rbf', gamma=(2*sigmal)**-2)
        MP = createMatrixMod(3)
        Kp = kernel._get_kernel(MP, MP,'rbf', gamma=(2*sigmap)**-2)
        Ki = kernel._get_kernel(MF_c, MF_c,'linear')
        kx =  K(X,y,'rbf',sigma1) * K(X,y,'linear')
        return Kl*Kp*Ki
        
    
def preimage_polyxrbf(X, alpha, kernel_fun,**params):
    """
    X: (numpy array nxm) Input data
    sigma: scalar
    alpha: numpy array (1xn)
    method: 
            fpi: Fixed-Point iterations
    """
    trainPre = list()
    max_iter = 10 #halt criterium
    epsilon = 0.0001 #halt criterium
    #error = np.array()
    #initialization
    x_pre = np.random.rand(1,X.shape[1])
        
    for it in range(1,max_iter):
        sum_num = np.zeros((1,X.shape[1]))
        sum_den = 0        
        for i in range(1,X.shape[0]):#might collapse this for with products
            #compute kernel
            kx = _get_kernel_deriv(X[i,:], x_pre, 'polyxrbf2', **params)            
            fact1 = alpha[i]*kx            
            sum_num += fact1*X[i,:]
            sum_den += fact1*X[i,:]*(x_pre - X[i,:])
            
        x_pre = sum_num/sum_den
        #compute error
        #error = np.norm(x_pre - k_x)
        trainPre.append(x_pre)
        
        
    return x_pre,trainPre
    
    
def preimage_linxrbf(X, alpha, kernel_fun,**params):
    """
    X: (numpy array nxm) Input data
    sigma: scalar
    alpha: numpy array (1xn)
    method: 
            fpi: Fixed-Point iterations
    """
    trainPre = list()
    max_iter = 10 #halt criterium
    epsilon = 0.0001 #halt criterium
    #error = np.array()
    #initialization
    x_pre = np.random.rand(1,X.shape[1])    
        
    for it in range(1,max_iter):
        sum_num = np.zeros((1,X.shape[1]))
        sum_den = 0        
        for i in range(1,X.shape[0]):#might collapse this for with products
            #compute kernel
            kx = _get_kernel_deriv(X[i,:], x_pre, kernel_fun, **params)            
            fact1 = alpha[i]*kx            
            sum_num += fact1*X[i,:]
            sum_den += fact1*X[i,:]*(x_pre - X[i,:])
            
        x_pre = sum_num/sum_den
        #compute error
        #error = np.norm(x_pre - k_x)
        trainPre.append(x_pre)
        
        
    return x_pre,trainPre
    
def preimage_faces(X, alpha, kernel_fun,**params):#under construction
    """
    X: (numpy array nxm) Input data
    sigma: scalar
    alpha: numpy array (1xn)
    method: 
            fpi: Fixed-Point iterations
    """
    trainPre = list()
    max_iter = 10 #halt criterium
    epsilon = 0.0001 #halt criterium
    #error = np.array()
    #initialization
    x_pre = np.random.rand(1,X.shape[1])    
        
    for it in range(1,max_iter):
        sum_num = np.zeros((1,X.shape[1]))
        sum_den = 0
        #compute kernel
        kx = _get_kernel_faces(X, x_pre, kernel_fun, **params)
        #kx = K(X,x_pre,'rbf',gamma = (2*sigma)**-2)#compute Gaussian kernel
        for i in range(1,X.shape[0]):#might collapse this for with products
            fact1 = alpha[i]*kx[i]
            sum_den += fact1
            sum_num += fact1*X[i,:]
            
        if sum_den == 0:
            x_pre = 0
        else:
            x_pre = sum_num/sum_den
        #compute error
        #error = np.norm(x_pre - k_x)
        trainPre.append(x_pre)
        
        
    return x_pre,trainPre

    
def preimage_isot(X, alpha, kernel_fun,**params):
    """
    X: (numpy array nxm) Input data
    sigma: scalar
    alpha: numpy array (1xn)
    method: 
            fpi: Fixed-Point iterations
    """
    trainPre = list()
    max_iter = 10 #halt criterium
    epsilon = 0.0001 #halt criterium
    #error = np.array()
    #initialization
    x_pre = np.random.rand(1,X.shape[1])    
        
    for it in range(1,max_iter):
        sum_num = np.zeros((1,X.shape[1]))
        sum_den = 0        
        for i in range(1,X.shape[0]):#might collapse this for with products
            #compute kernel
            kx = _get_kernel_deriv(X[i,:], x_pre, kernel_fun, **params)            
            fact1 = alpha[i]*kx
            sum_den += fact1
            sum_num += fact1*X[i,:]
            
        if sum_den == 0:
            x_pre = 0
        else:
            x_pre = sum_num/sum_den
        #compute error
        #error = np.norm(x_pre - k_x)
        trainPre.append(x_pre)
        
        
    return x_pre,trainPre

def preimage(X, alpha, kernel_fun,**params):
    """
    X: (numpy array nxm) Input data
    sigma: scalar
    alpha: numpy array (1xn)
    method: 
            fpi: Fixed-Point iterations
    """
    trainPre = list()
    max_iter = 10 #halt criterium
    epsilon = 0.0001 #halt criterium
    #error = np.array()
    #initialization
    x_pre = np.random.rand(1,X.shape[1])    
        
    for it in range(1,max_iter):
        sum_num = np.zeros((1,X.shape[1]))
        sum_den = 0
        #compute kernel
        kx = _get_kernel(X, x_pre, kernel_fun, **params)
        #kx = K(X,x_pre,'rbf',gamma = (2*sigma)**-2)#compute Gaussian kernel
        for i in range(1,X.shape[0]):#might collapse this for with products
            fact1 = alpha[i]*kx[i]
            sum_den += fact1
            sum_num += fact1*X[i,:]
            
        if sum_den == 0:
            x_pre = 0
        else:
            x_pre = sum_num/sum_den
        #compute error
        #error = np.norm(x_pre - k_x)
        trainPre.append(x_pre)
        
        
    return x_pre,trainPre

def preimage2(X, alpha, Xgt, kernel_fun,**params):
    """
    X: (numpy array nxm) Input data
    sigma: scalar
    alpha: numpy array (1xn)
    method: 
            fpi: Fixed-Point iterations
    """
    trainPre = list()
    max_iter = 10 #halt criterium
    epsilon = 0.0001 #halt criterium
    #error = np.array()
    #initialization
    x_pre = np.random.rand(1,X.shape[1])    
        
    for it in range(1,max_iter):
        sum_num = np.zeros((1,X.shape[1]))
        sum_den = 0
        #compute kernel
        print 'Preimage ',it,  X.shape, x_pre.shape
        kx = _get_kernel(X, x_pre, kernel_fun, **params)
        #kx = K(X,x_pre,'rbf',gamma = (2*sigma)**-2)#compute Gaussian kernel
        for i in range(1,X.shape[0]):#might collapse this for with products
            fact1 = alpha[i]*kx[i]
            sum_den += fact1
            sum_num += fact1*X[i,:]
            
        if sum_den == 0:
            x_pre = 0
        else:
            x_pre = sum_num/sum_den
        #compute error
        if Xgt is None:
            trainPre.append(x_pre)
        else:            
            trainPre.append(np.linalg.norm(x_pre - Xgt))        
        
        
    return x_pre,trainPre

