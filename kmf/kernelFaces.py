"""
Class to compute tensorial kernel for CMU faces data set
"""
import numpy as np
from numpy import dot
from sklearn.metrics import pairwise_kernels as K
from sklearn.metrics import pairwise_distances as D

class KernelFaces():

    def __init__(self, X, indx_c):
        self.MF = X
        self.MS = self.createMatrixMod(1)        
        self.ML = self.createMatrixMod(2)
        self.MP = self.createMatrixMod(3)
        self.eq = 1
        self.diff = 0.3
        self.indx = indx_c
    
    def createMatrixMod(self, mode):
        if mode == 1:#on CMU faces corresponds to subject
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
        elif mode == 2:#on CMU faces corresponds to illumination
            TL_ = np.repeat(np.array(np.arange(1,22), ndmin = 2),11,axis=0)
            TL = np.repeat(np.array(TL_,ndmin = 3),30,axis=0)
            M = np.reshape(TL,(30*11*21,1))
        elif mode == 3:#on CMU faces corresponds to pose
            TP_ = np.repeat(np.array([175, 95, 93, 88, 10, 5, 180, 90, 80, 2, 170], ndmin = 2),21,axis=0)
            TP = np.repeat(np.array(TP_.T,ndmin = 3),30,axis=0)
            M = np.reshape(TP,(30*11*21,1))

        return M
    
    def soft_kernel(self, x= None, eq =1, diff=0.3, indx=[]):
        if x is None:
            self.Ks = np.zeros((len(indx),len(indx)))
            for i in range(len(indx)):
                for j in range(len(indx)):
                    if self.MS[i]==self.MS[j]:
                        self.Ks[i,j] = eq
                    else:
                        self.Ks[i,j] = diff
            ks = self.Ks#a matrix
        else:
            ks = np.zeros((len(indx),1))
            for i in range(len(indx)):                
                if self.MS[i]==self.MS[x]:
                    ks[i] = eq
                else:
                    ks[i] = diff
        return ks#a vector
    
    def _get_kernel(self, X, y, idx_y, kernel_func, sigmal, sigmap):
        ks = self.soft_kernel(x=idx_y, eq=self.eq, diff=self.diff, indx=self.indx)
        kt =  ks * K(self.ML[self.indx],self.ML[idx_y],'rbf',gamma=(2*sigmal)**-2) * K(self.MP[self.indx],self.MP[idx_y],'rbf',gamma=(2*sigmap)**-2) * K(self.MF[self.indx],y,'linear')
        return kt

    def preimage_faces(self, X, alpha, idx_y, kernel_fun, sigmal,sigmap):
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
            #kx = K(X,x_pre,'rbf',gamma = (2*sigma)**-2)#compute Gaussian kernel
            kx = self._get_kernel(X, x_pre, idx_y, kernel_fun, sigmal,sigmap)
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

