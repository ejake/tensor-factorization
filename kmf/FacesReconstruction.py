"""
Class to compute tensorial kernel for CMU faces data set
"""
import numpy as np
import scipy.io as sio #to load mat files
#import
import kernel as kernel
from kcnmf import KCNMF

class FacesReconstruction:
    

    def __init__(self, rt_obinc, rt_miss):        
        self.rt_obinc = .4
        self.rt_miss = .5
        self.lamb = 1e-10 #regularization parameter to compute H~
        self.niter=10 # # iterations kcnmf

    def load_CMU_dataset(self, path):
        print sio.whosmat(path)
        matTF = sio.loadmat(path)#loading tensor of formatted faces' images
        self.TF = matTF['FullTensor']
        print 'Tensor order:', self.TF.shape
        self.MF = np.reshape(self.TF,(30*11*21,1024))
        print 'Matrization order:', self.MF.shape
	
    def incomplete_data(self, X, miss_rate, miss_obj):
        #incomplete data
        self.rt_obinc = miss_rate
        self.rt_miss = miss_obj # rate of minimun elements (attributes or variables) incomplete for each object
        self.idx_inc = np.sort(np.random.choice(X.shape[0], int(X.shape[0]*self.rt_obinc), replace=False))
        idx_v = np.ones(X.shape[0])
        idx_v[self.idx_inc] = 0
        idx_s = np.vstack((idx_v,idx_v)).repeat([1,X.shape[1]-1],0).T
        self.idx_com = np.arange(0,X.shape[0]-1)
        self.idx_com = np.delete(self.idx_com, self.idx_inc)
        #data_i = MF[idx_inc]#subset with incomplete elements
        #data_c = MF[idx_com]#subset with complete elements
        idx_s2 = np.copy(idx_s) # idx_s2 could have partially incomplete objects
        for i in self.idx_inc:
            idx_s2[i] = np.random.permutation(np.append(np.ones(int(round(X.shape[1]*(1-self.rt_miss)))),np.zeros(int(round(X.shape[1]*self.rt_miss)))))

        self.mask = idx_s2
        self.MF_ci = X * idx_s2

        return idx_s2

    def incomplete_face(self, index_image, pixel = None):
        self.pixel_image = index_image
        self.pixel_pixel = pixel
        self.Y = np.tile(np.arange(0,32),32)
        self.X = np.repeat(np.arange(0,32),32)
        if pixel is None:
            if index_image in self.idx_inc:
                msk = self.mask[index_image]
            else:
                msk = np.random.permutation(np.append(np.ones(int(round(self.MF.shape[1]*(1-self.rt_miss)))),np.zeros(int(round(self.MF.shape[1]*self.rt_miss)))))
        else:
            msk = np.squeeze(np.ones((self.MF.shape[1],1)))
            msk[pixel] = 0

        self.Pixel = self.MF[index_image]*msk
        self.pixel_completes = range(0,1024)*msk
        self.pixel_incompletes = range(0,1024)
        self.pixel_incompletes = np.delete(self.pixel_incompletes, self.pixel_completes).astype(int)
        self.pixel_completes  = np.delete(self.pixel_completes, self.pixel_incompletes).astype(int)
        
        return msk

    def model_training(self, params):
        self.parameters = params
        #Tensorial kernel over complete objects
        Kx = kernel._get_kernel(np.stack((self.X[self.pixel_completes].T,np.zeros(len(self.pixel_completes)).T), axis = 1), np.stack((self.X[self.pixel_completes].T,np.zeros(len(self.pixel_completes)).T), axis = 1),'rbf', gamma=(2*self.parameters['sigmax'])**-2)
        Ky = kernel._get_kernel(np.stack((self.Y[self.pixel_completes].T,np.zeros(len(self.pixel_completes)).T), axis = 1), np.stack((self.Y[self.pixel_completes].T,np.zeros(len(self.pixel_completes)).T), axis = 1),'rbf', gamma=(2*self.parameters['sigmay'])**-2)
        self.Kt = Kx*Ky
        #Factorize
        self.kcnmf = KCNMF(self.Kt,num_bases=self.parameters['bases'])
        self.kcnmf.factorize(niter=self.niter)
        
    def pseudinv_H(self):
        
        self.H4_i = np.zeros((self.parameters['bases'], len(self.pixel_incompletes)))
        for i in range(len(self.pixel_incompletes)):
            auxpi4 = np.linalg.inv(np.dot(self.kcnmf.W.T, self.kcnmf.G)+ self.lamb*np.identity(self.parameters['bases']))
            #auxpi4 = np.dot(np.linalg.inv(np.dot(np.dot(cnmf_kpg.G.T,Kt),cnmf_kpg.G)),cnmf_kpg.G.T)#OKMF
            
            #It's not feasible compute K(X,x) for pixel since all missing pixel are 0
            Kxx = kernel._get_kernel(np.stack((self.X[self.pixel_completes].T,np.zeros(len(self.pixel_completes)).T), axis = 1), np.append(self.X[self.pixel_incompletes[i]], 0),'rbf', gamma=(2*self.parameters['sigmax'])**-2)
            Kyx = kernel._get_kernel(np.stack((self.Y[self.pixel_completes].T,np.zeros(len(self.pixel_completes)).T), axis = 1), np.append(self.Y[self.pixel_incompletes[i]], 0),'rbf', gamma=(2*self.parameters['sigmay'])**-2)
            Ktx = Kxx*Kyx
            
            self.H4_i[:,i] = np.squeeze(np.dot(auxpi4, np.dot(self.kcnmf.G.T, Ktx) ))
            #H4_i[:,i] = np.squeeze(np.dot(auxpi4, Ktx )) #OKMF
            if self.parameters['normalizeh'] == 1:
                self.H4_i[:,i] = (self.H4_i[:,i]+abs(min(self.H4_i[:,i]))) / sum(self.H4_i[:,i]+abs(min(self.H4_i[:,i])))
            if self.parameters['normalizeh'] == 2:
                self.H4_i[:,i] = abs(self.H4_i[:,i])
            if self.parameters['normalizeh'] == 3:
                self.H4_i[:,i] = np.maximum(self.H4_i[:,i],0)
    
    def preimage(self, preimage_method = 2, kernel_method = 'rbf'):
        rse_f = []
        self.pixel_preim = np.copy(self.Pixel)
        for i in range(len(self.pixel_incompletes)):
            #print 'Compute pre-image for object',incompletes[i]
            #alpha = np.dot(cnmf_kpg.G, H4_i[:,i]) #np.mean(cnmf_kg.H,0)  
            alpha = np.dot(self.kcnmf.W, self.H4_i[:,i]) #np.mean(cnmf_kg.H,0)  
            if self.parameters['normalizea'] == 1:
                alpha = (alpha+abs(min(alpha))) / sum(alpha+abs(min(alpha)))*100
            if self.parameters['normalizea'] == 2:
                alpha = abs(alpha)
            if self.parameters['normalizea'] == 3:
                alpha = np.maximum(alpha,0)
            if self.parameters['normalizea'] == 4:
                alpha = (alpha-min(alpha))/(max(alpha)-min(alpha))
            if self.parameters['normalizea'] == 5:
                alpha = alpha*100
            if self.parameters['normalizea'] == 6:
                alpha = np.ones(alpha.shape)
            
            
            if preimage_method == 2:
                Kxx = kernel._get_kernel(np.stack((self.X[self.pixel_completes].T,np.zeros(len(self.pixel_completes)).T), axis = 1), np.append(self.X[self.pixel_incompletes[i]], 0),kernel_method, gamma=(2*self.parameters['sigmax'])**-2)
                Kyx = kernel._get_kernel(np.stack((self.Y[self.pixel_completes].T,np.zeros(len(self.pixel_completes)).T), axis = 1), np.append(self.Y[self.pixel_incompletes[i]], 0),kernel_method, gamma=(2*self.parameters['sigmay'])**-2)
                Ktx = Kxx*Kyx
                aux, trainErrorskl = kernel.pixel_preimage(self.Pixel[self.pixel_completes], Ktx, alpha, 'rbf', gamma=(2*self.parameters['sigmap'])**-2)
            if preimage_method == 1:
                gamma_t = (2*self.parameters['sigmax'])**-2+(2*self.parameters['sigmay'])**-2
                aux, trainErrorskl = kernel.preimage( np.stack( (self.Pixel[self.pixel_completes].T, np.zeros(len(self.pixel_completes)).T), axis = 1), alpha, kernel_method, gamma=gamma_t)#kernel_method='rbf'
                aux = aux[0,0]
            if preimage_method == 3:#alpha x intensity
                aux = np.dot(self.Pixel[self.pixel_completes].T, alpha)
            print aux
            #print X[i], Y[i]
            
            self.pixel_preim[self.pixel_incompletes[i]] = aux
            rse_f.append(np.linalg.norm(self.Pixel[self.pixel_incompletes[i]] - self.pixel_preim[self.pixel_incompletes[i]])/np.linalg.norm(self.Pixel[self.pixel_incompletes[i]]))
        
        #return Ktx, alpha, Kxx, Kyx

    
    def naive_avg(self):
        avg_intpv = []
        for pix_inc in self.pixel_incompletes:
            avg_intp = 0
            cnt = 0
            neigh_pix = [(self.X[pix_inc])*32+self.Y[pix_inc]-1, (self.X[pix_inc]-1)*32+self.Y[pix_inc], (self.X[pix_inc]+1)*32+self.Y[pix_inc], (self.X[pix_inc])*32+self.Y[pix_inc]+1, (self.X[pix_inc]-1)*32+self.Y[pix_inc]-1, (self.X[pix_inc]-1)*32+self.Y[pix_inc]+1, (self.X[pix_inc]+1)*32+self.Y[pix_inc]-1, (self.X[pix_inc]+1)*32+self.Y[pix_inc]+1 ]#(up), (left), (righ), (down), (up-left), (down-left), (up-right), (down-right)
            
            for m in neigh_pix:
                if m in self.pixel_completes:
                    avg_intp += self.Pixel[m]
                    cnt += 1

            try:
                avg_intpv.append(avg_intp/cnt)
            except ZeroDivisionError:
                avg_intpv.append(avg_intp)
                
        return avg_intpv