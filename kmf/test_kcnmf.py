import numpy as np
import kernel as kernel
from kernelFaces import KernelFaces as KF
import pymf
from pymf.cnmf import CNMF 
from kcnmf import KCNMF 
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def test(MF):
    filename = r'/home/rajaquep/outcomes/cmu_faces_12sep2016.csv'
    with open(filename, 'a') as text_file:
        text_file.write("kernel ; missing rate ; time ; rse completion ; rse whole; rse whole_inc\n")

    #Script to run experiments
    for i in range(0,10):
        m_rate = [.1, .40, .50, .60, .70, .80, .90, .92, .94, .96, .98]
        for rt_obinc in m_rate:
            MF_ci,idx_inc = incomplete_data(MF, rt_obinc)
            with open(filename, 'a') as text_file:
                text_file.write("// Iteration %i; Missing rate = %f; RSE (X, X_ic) = %f\n" % (i, rt_obinc, np.linalg.norm(MF - MF_ci)/np.linalg.norm(MF)))

            tic = time.clock()#get start time
            KlX_ci = kernel._get_kernel(MF_ci, MF_ci,'linear')
            num_bases = 15
            cnmf_kl_ci = factorize(KlX_ci, num_bases)
            preimxkl_ci = pre_image_solve(MF_ci, cnmf_kl_ci, idx_inc, num_bases, 'linear', 0)
            toc = time.clock()#get final time
            timec = toc - tic
            #compute error
            errorc = np.linalg.norm(MF[idx_inc] - preimxkl_ci[idx_inc])/np.linalg.norm(MF[idx_inc])
            errora = np.linalg.norm(MF - preimxkl_ci)/np.linalg.norm(MF)
            errori = np.linalg.norm(MF_ci - preimxkl_ci)/np.linalg.norm(MF_ci)
            #print 'Linear', rt_obinc, timec, errorc, errora, errori
            with open(filename, 'a') as text_file:
                text_file.write("%s ; %f ; %f ; %f ; %f; %f\n" % ('Linear', rt_obinc, timec, errorc, errora, errori))
        
            tic = time.clock()#get start time    
            sigma = 78275
            KgX_ci = kernel._get_kernel(MF_ci, MF_ci,'rbf', gamma=(2*sigma)**-2)
            num_bases = 5
            cnmf_kg_ci = factorize(KgX_ci, num_bases)
            preimxkg_ci = pre_image_solve(MF_ci, cnmf_kg_ci, idx_inc, num_bases, 'rbf', sigma)
            toc = time.clock()#get final time 
            timec = toc - tic
            #compute error                                                     
            errorc = np.linalg.norm(MF[idx_inc] - preimxkg_ci[idx_inc])/np.linalg.norm(MF[idx_inc])
            errora = np.linalg.norm(MF - preimxkg_ci)/np.linalg.norm(MF)
            errori =  np.linalg.norm(MF_ci - preimxkg_ci)/np.linalg.norm(MF_ci)
            with open(filename, 'a') as text_file:
                text_file.write("%s ; %f ; %f ; %f ; %f; %f\n" % ('rbf', rt_obinc, timec, errorc, errora, errori))


            tic = time.clock()#get start time
            sigma = 147368.421053
            KlxgX_ci = kernel._get_kernel(MF_ci, MF_ci,'linxrbf', gamma=(2*sigma)**-2)
            num_bases = 7
            cnmf_klxg_ci = factorize(KlxgX_ci, num_bases)
            preimxklxg_ci = pre_image_solve(MF_ci, cnmf_klxg_ci, idx_inc, num_bases, 'linxrbf', sigma)
            toc = time.clock()#get final time
            timec = toc - tic
            #compute error
            errorc = np.linalg.norm(MF[idx_inc] - preimxklxg_ci[idx_inc])/np.linalg.norm(MF[idx_inc])
            errora = np.linalg.norm(MF - preimxklxg_ci)/np.linalg.norm(MF)
            errori = np.linalg.norm(MF_ci - preimxklxg_ci)/np.linalg.norm(MF_ci)
            with open(filename, 'a') as text_file:
                text_file.write("%s ; %f ; %f ; %f ; %f; %f\n" % ('linxrbf', rt_obinc, timec, errorc, errora, errori))

# Including soft kernel
def test2(MF):
    filename = r'/home/rajaquep/outcomes/cmu_faces_14sep2016.csv'
    with open(filename, 'a') as text_file:
        text_file.write("kernel ; missing rate ; time ; rse completion ; rse whole; rse whole_inc\n")
    
    #30x11x21x1024
    Ks = kernel.softKernel(MF, 0.3, 1, 30)
    #Script to run experiments
    for i in range(0,10):
        m_rate = [.1, .40, .50, .60, .70, .80, .90, .92, .94, .96, .98]
        for rt_obinc in m_rate:
            MF_ci,idx_inc = incomplete_data(MF, rt_obinc)
            with open(filename, 'a') as text_file:
                text_file.write("// Iteration %i; Missing rate = %f; RSE (X, X_ic) = %f\n" % (i, rt_obinc, np.linalg.norm(MF - MF_ci)/np.linalg.norm(MF)))

            tic = time.clock()#get start time
            sigma = 76315.7894737
            KsxlxgX_ci = np.dot( kernel._get_kernel(MF_ci, MF_ci,'linxrbf', gamma=(2*sigma)**-2), Ks)
            num_bases = 7
            cnmf_ksxlxg_ci = factorize(KsxlxgX_ci, num_bases)
            preimxksxlxg_ci = pre_image_solve(MF_ci, cnmf_ksxlxg_ci, idx_inc, num_bases, 'linxrbf', sigma)
            toc = time.clock()#get final time
            timec = toc - tic
            #compute error
            errorc = np.linalg.norm(MF[idx_inc] - preimxksxlxg_ci[idx_inc])/np.linalg.norm(MF[idx_inc])
            errora = np.linalg.norm(MF - preimxksxlxg_ci)/np.linalg.norm(MF)
            errori = np.linalg.norm(MF_ci - preimxksxlxg_ci)/np.linalg.norm(MF_ci)
            with open(filename, 'a') as text_file:
                text_file.write("%s ; %f ; %f ; %f ; %f; %f\n" % ('softxlinxrbf', rt_obinc, timec, errorc, errora, errori))

# Including soft kernel
def test3(MF, MF_ci, idx_inc, m_rate):
    filename = r'/home/rajaquep/outcomes/cmu_faces_28sep2016.csv'
    with open(filename, 'a') as text_file:
        text_file.write("kernel ; missing rate ; time ; rse completion ; rse whole; rse whole_inc\n")
    
    #30x11x21x1024
    Ks = kernel.softKernel(MF, 0.3, 1, 30)
    #Script to run experiments
    for i in range(0,10):

        with open(filename, 'a') as text_file:
            text_file.write("// Iteration %i; Missing rate = %f; RSE (X, X_ic) = %f\n" % (i, m_rate, np.linalg.norm(MF - MF_ci)/np.linalg.norm(MF)))
        
        tic = time.clock()#get start time
        KlX_ci = kernel._get_kernel(MF_ci, MF_ci,'linear')
        num_bases = 15
        cnmf_kl_ci = factorize(KlX_ci, num_bases)
        preimxkl_ci = pre_image_solve(MF_ci, cnmf_kl_ci, idx_inc, num_bases, 'linear', 0)
        toc = time.clock()#get final time
        timec = toc - tic
        #compute error
        errorc = np.linalg.norm(MF[idx_inc] - preimxkl_ci[idx_inc])/np.linalg.norm(MF[idx_inc])
        errora = np.linalg.norm(MF - preimxkl_ci)/np.linalg.norm(MF)
        errori = np.linalg.norm(MF_ci - preimxkl_ci)/np.linalg.norm(MF_ci)
        #print 'Linear', rt_obinc, timec, errorc, errora, errori
        with open(filename, 'a') as text_file:
            text_file.write("%s ; %f ; %f ; %f ; %f; %f\n" % ('Linear', m_rate, timec, errorc, errora, errori))
    
        tic = time.clock()#get start time    
        sigma = 78275
        KgX_ci = kernel._get_kernel(MF_ci, MF_ci,'rbf', gamma=(2*sigma)**-2)
        num_bases = 5
        cnmf_kg_ci = factorize(KgX_ci, num_bases)
        preimxkg_ci = pre_image_solve(MF_ci, cnmf_kg_ci, idx_inc, num_bases, 'rbf', sigma)
        toc = time.clock()#get final time 
        timec = toc - tic
        #compute error                                                     
        errorc = np.linalg.norm(MF[idx_inc] - preimxkg_ci[idx_inc])/np.linalg.norm(MF[idx_inc])
        errora = np.linalg.norm(MF - preimxkg_ci)/np.linalg.norm(MF)
        errori =  np.linalg.norm(MF_ci - preimxkg_ci)/np.linalg.norm(MF_ci)
        with open(filename, 'a') as text_file:
            text_file.write("%s ; %f ; %f ; %f ; %f; %f\n" % ('rbf', m_rate, timec, errorc, errora, errori))

        tic = time.clock()#get start time
        sigma = 147368.421053
        KlxgX_ci = kernel._get_kernel(MF_ci, MF_ci,'linxrbf', gamma=(2*sigma)**-2)
        num_bases = 7
        cnmf_klxg_ci = factorize(KlxgX_ci, num_bases)
        preimxklxg_ci = pre_image_solve(MF_ci, cnmf_klxg_ci, idx_inc, num_bases, 'linxrbf', sigma)
        toc = time.clock()#get final time
        timec = toc - tic
        #compute error
        errorc = np.linalg.norm(MF[idx_inc] - preimxklxg_ci[idx_inc])/np.linalg.norm(MF[idx_inc])
        errora = np.linalg.norm(MF - preimxklxg_ci)/np.linalg.norm(MF)
        errori = np.linalg.norm(MF_ci - preimxklxg_ci)/np.linalg.norm(MF_ci)
        with open(filename, 'a') as text_file:
            text_file.write("%s ; %f ; %f ; %f ; %f; %f\n" % ('linxrbf', m_rate, timec, errorc, errora, errori))


        tic = time.clock()#get start time
        sigma = 76315.7894737
        KsxlxgX_ci = np.dot( kernel._get_kernel(MF_ci, MF_ci,'linxrbf', gamma=(2*sigma)**-2), Ks)
        num_bases = 7
        cnmf_ksxlxg_ci = factorize(KsxlxgX_ci, num_bases)
        preimxksxlxg_ci = pre_image_solve(MF_ci, cnmf_ksxlxg_ci, idx_inc, num_bases, 'linxrbf', sigma)
        toc = time.clock()#get final time
        timec = toc - tic
        #compute error
        errorc = np.linalg.norm(MF[idx_inc] - preimxksxlxg_ci[idx_inc])/np.linalg.norm(MF[idx_inc])
        errora = np.linalg.norm(MF - preimxksxlxg_ci)/np.linalg.norm(MF)
        errori = np.linalg.norm(MF_ci - preimxksxlxg_ci)/np.linalg.norm(MF_ci)
        with open(filename, 'a') as text_file:
            text_file.write("%s ; %f ; %f ; %f ; %f; %f\n" % ('softxlinxrbf', m_rate, timec, errorc, errora, errori))


#Input:
#X: Original Data nparray(objects, attributes)
#miss_rate: Missing data rate
#miss_obj: Missing data rate per object 
#Output:
#MF_ci: Incomplete Data nparray(objects, attributes)
#idx_inc: Index of incomplete objects
def incomplete_data(X, miss_rate,miss_obj=0.5):
    #incomplete data
    rt_obinc = miss_rate
    rt_incpob = miss_obj # rate of minimun elements (attributes or variables) incomplete for each object
    idx_inc = np.sort(np.random.choice(X.shape[0], X.shape[0]*rt_obinc, replace=False))
    idx_v = np.ones(X.shape[0])
    idx_v[idx_inc] = 0
    idx_s = np.vstack((idx_v,idx_v)).repeat([1,X.shape[1]-1],0).T
    idx_com = np.arange(0,X.shape[0]-1)
    idx_com = np.delete(idx_com, idx_inc)
    #data_i = MF[idx_inc]#subset with incomplete elements
    #data_c = MF[idx_com]#subset with complete elements
    idx_s2 = np.copy(idx_s) # idx_s2 could have partially incomplete objects
    for i in idx_inc:
        idx_s2[i] = np.random.permutation(np.append(np.ones(round(X.shape[1]*(1-rt_incpob))),np.zeros(round(X.shape[1]*rt_incpob))))
    MF_ci = X * idx_s2

    return MF_ci, idx_inc, idx_com, idx_s2
    
def factorize(X, num_bases, niter=10):
    cnmf_ci = CNMF(X, num_bases)
    cnmf_ci.factorize(niter)

    return cnmf_ci

#Input:
#X: incomplete data
#num_bases: number of bases (latent space size)
def pre_image_solve(X, cnmf, idx_inc, num_bases, typek, sigma):
    #Pre-image reconstruction
    num_atts = X.shape[1]
    preimxk_ci = np.copy(X)
    
    for j in idx_inc:
        # \phi(x_i) to linear projection
        xs = X[max(j-num_bases,0):max(j+num_bases,0)]
        xs = np.resize(xs,(num_bases,num_atts))    
        alpha = cnmf.H[:,j] #np.mean(cnmf_kg.H,0)
        if typek == 'linear':
            preimxk_ci[j], trainErrorskl = kernel.preimage(xs, alpha, 'linear')
        elif typek == 'rbf':
            preimxk_ci[j], trainErrorskg = kernel.preimage_isot(xs, alpha, 'rbf', gamma=(2*sigma)**-2)
        elif typek == 'linxrbf':
            preimxk_ci[j], trainErrorskg = kernel.preimage(xs, alpha, 'linxrbf', gamma=(2*sigma)**-2)
        
    return preimxk_ci

def param_exp_sigma(MF, idx_com, bases):
    filename = r'/home/rajaquep/outcomes/cmu_faces_param_20oct2016.csv'
    with open(filename, 'a') as text_file:
        text_file.write("sigmal ; sigmap ; rse\n")
    #Subject
    #30x11x21x1024
    eq = 1
    diff = 0.3
    poses = [175, 95, 93, 88, 10, 5, 180, 90, 80, 2, 170]
    illuminations = np.arange(1,22)
    TS = np.zeros((30,11,21))
    for i in range(30):
        for j in range(len(poses)):
            for k in range(len(illuminations)):
                TS[i,j,k] = i
            
    MS = np.reshape(TS,(30*11*21,1))
    #Kernel for subject
    Ks = np.zeros((len(idx_com),len(idx_com)))
    for i in range(len(idx_com)):
        for j in range(len(idx_com)):
            if MS[i]==MS[j]:
                Ks[i,j] = eq
            else:
                Ks[i,j] = diff
    #print Ks.shape
    #illumination
    TL_ = np.repeat(np.array(np.arange(1,22), ndmin = 2),11,axis=0)
    TL = np.repeat(np.array(TL_,ndmin = 3),30,axis=0)
    #print TL.shape
    ML = np.reshape(TL,(30*11*21,1))
    #print ML.shape
    ML_c = ML[idx_com]
    #print ML_c.shape
    #pose
    TP_ = np.repeat(np.array([175, 95, 93, 88, 10, 5, 180, 90, 80, 2, 170], ndmin = 2),21,axis=0)
    TP = np.repeat(np.array(TP_.T,ndmin = 3),30,axis=0)
    #print TP.shape
    MP = np.reshape(TP,(30*11*21,1))
    #print MP.shape
    MP_c = MP[idx_com]
    #print MP_c.shape
    #Exploring sigma
    sigmavl = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1e1, 1e2, 1e3, 1e4, 1e5 ])
    sigmavp = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1e1, 1e2, 1e3, 1e4, 1e5 ])
    for sigmal in sigmavl:
        for sigmap in sigmavp:
            Kl = kernel._get_kernel(ML_c, ML_c,'rbf', gamma=(2*sigmal)**-2)
            Kp = kernel._get_kernel(MP_c, MP_c,'rbf', gamma=(2*sigmap)**-2)
            Ki = kernel._get_kernel(MF, MF,'linear')
            #Kt = np.dot(np.dot(np.dot(Ks,Kp),Kl),Ki)
            Kt = Ks*Kp*Kl*Ki
            tic = time.clock()#get start time
            cnmf_kt_c = CNMF(Kt, num_bases=bases)
            cnmf_kt_c.factorize(niter=10)
            rse = np.linalg.norm(Kt - np.dot(cnmf_kt_c.W,cnmf_kt_c.H))/np.linalg.norm(Kt)
            toc = time.clock()#get final time 
            timec = toc - tic
            with open(filename, 'a') as text_file:
                text_file.write("%f ; %f ; %f ; %f \n" % (timec, sigmal, sigmap, rse))

#cmu rbfx4
def param_exp_sigma2(MF, idx_com, bases):
    filename = r'/home/rajaquep/outcomes/cmu_faces_param_8nov2016.csv'
    with open(filename, 'a') as text_file:
        text_file.write("sigmal ; sigmap ; sigmai ; rse\n")
    #Subject                                                                                                                                
    #30x11x21x1024                                                                                                                         
    eq = 1
    diff = 0.3
    poses = [175, 95, 93, 88, 10, 5, 180, 90, 80, 2, 170]
    illuminations = np.arange(1,22)
    TS = np.zeros((30,11,21))
    for i in range(30):
        for j in range(len(poses)):
            for k in range(len(illuminations)):
                TS[i,j,k] = i

    MS = np.reshape(TS,(30*11*21,1))
    #Kernel for subject                                                                                                                     
    Ks = np.zeros((len(idx_com),len(idx_com)))
    for i in range(len(idx_com)):
        for j in range(len(idx_com)):
            if MS[i]==MS[j]:
                Ks[i,j] = eq
            else:
                Ks[i,j] = diff
    #print Ks.shape                                                                                                                        
    #illumination                                                                                                                          
    TL_ = np.repeat(np.array(np.arange(1,22), ndmin = 2),11,axis=0)
    TL = np.repeat(np.array(TL_,ndmin = 3),30,axis=0)
    #print TL.shape                                                                                                                        
    ML = np.reshape(TL,(30*11*21,1))
    #print ML.shape                                                                                                                        
    ML_c = ML[idx_com]
    #print ML_c.shape                                                                                                                      
    #pose                                                                                                                                  
    TP_ = np.repeat(np.array([175, 95, 93, 88, 10, 5, 180, 90, 80, 2, 170], ndmin = 2),21,axis=0)
    TP = np.repeat(np.array(TP_.T,ndmin = 3),30,axis=0)
    #print TP.shape                                                                                                                        
    MP = np.reshape(TP,(30*11*21,1))
    #print MP.shape                                                                                                                        
    MP_c = MP[idx_com]
    #print MP_c.shape                                                                                                                          #Exploring sigma                                                                                                                      s
    sigmavl = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1e1, 1e2, 1e3, 1e4, 1e5 ])
    sigmavp = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1e1, 1e2, 1e3, 1e4, 1e5 ])
    sigmavi = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1e1, 1e2, 1e3, 1e4, 1e5 ])
    for sigmal in sigmavl:
        for sigmap in sigmavp:
            for sigmai in sigmavi:
                Kl = kernel._get_kernel(ML_c, ML_c,'rbf', gamma=(2*sigmal)**-2)
                Kp = kernel._get_kernel(MP_c, MP_c,'rbf', gamma=(2*sigmap)**-2)
                Ki = kernel._get_kernel(MF, MF,'rbf', gamma=(2*sigmai)**-2)
                #Kt = np.dot(np.dot(np.dot(Ks,Kp),Kl),Ki)                                                                                       
                Kt = Ks*Kp*Kl*Ki
                tic = time.clock()#get start time                                                                                               
                cnmf_kt_c = CNMF(Kt, num_bases=bases)
                cnmf_kt_c.factorize(niter=10)
                rse = np.linalg.norm(Kt - np.dot(cnmf_kt_c.W,cnmf_kt_c.H))/np.linalg.norm(Kt)
                toc = time.clock()#get final time                                                                                               
                timec = toc - tic
                with open(filename, 'a') as text_file:
                    text_file.write("%f ; %f ; %f ; %f ; %f \n" % (timec, sigmal, sigmap, sigmai, rse))


def test_cmu1(MF, sigmal, sigmap, bases):
    filename = r'/home/rajaquep/outcomes/cmu_faces_8nov2016.csv'
    with open(filename, 'a') as text_file:
        text_file.write("kernel ; missing rate ; time ; rse completion ; rse whole; rse whole_inc; rse feature_space\n")

    #Script to run experiments
    for i in range(0,10):
        m_rate = [.1, .40, .50, .60, .70, .80, .90, .92, .94, .96, .98]
        for rt_obinc in m_rate:
            tic = time.clock()#get start time
            MF_ci,idx_inc = incomplete_data(MF, rt_obinc)
            idx_com = np.arange(0,MF.shape[0]-1)
            idx_com = np.delete(idx_com, idx_inc)
            #with open(filename, 'a') as text_file:
                #text_file.write("// Iteration %i; Missing rate = %f; RSE (X, X_ic) = %f\n" % (i, rt_obinc, np.linalg.norm(MF - MF_ci)/np.linalg.norm(MF)))
            kf = KF(MF, idx_com)
            #Kernel for subject
            Ks = kf.soft_kernel( indx=idx_com )
            #Kernel for illumination            
            Kl = kernel._get_kernel(kf.ML[idx_com], kf.ML[idx_com], 'rbf', gamma=(2*sigmal)**-2)
            #Kernel for pose
            Kp = kernel._get_kernel(kf.MP[idx_com], kf.MP[idx_com], 'rbf', gamma=(2*sigmap)**-2)
            #Kernel for image
            Ki = kernel._get_kernel(kf.MF[idx_com], kf.MF[idx_com],'linear')
            #Kt = np.dot(np.dot(np.dot(Ks,Kp),Kl),Ki)
            Kt = Ks*Kp*Kl*Ki
            #Factorize
            cnmf_kt_c = CNMF(Kt, num_bases=bases)
            cnmf_kt_c.factorize(niter=10)
            rse_fs = np.linalg.norm(Kt - np.dot(cnmf_kt_c.W,cnmf_kt_c.H))/np.linalg.norm(Kt)#log
            #compute hj for each incomplete object
            lamb = 0
            H_i = np.zeros((bases, len(idx_inc)))
            k=0
            for j in idx_inc:
                #Kernel for illumination
                Klx = kernel._get_kernel(kf.ML[idx_com], kf.ML[j],'rbf', gamma=(2*sigmal)**-2)
                #Kernel for pose
                Kpx = kernel._get_kernel(kf.MP[idx_com], kf.MP[j],'rbf', gamma=(2*sigmap)**-2)
                #Kernel for image
                Kix = kernel._get_kernel(kf.MF[idx_com], MF_ci[j],'linear')
                #Kernle for subject
                #...
                Ksx = np.zeros((len(idx_com),1))
                for i in range(len(idx_com)):
                    if kf.MS[i]==kf.MS[j]: Ksx[i] = kf.eq
                    else: Ksx[i] = kf.diff
                #print Ksx.shape  
                Ktx = Ksx*Kpx*Klx*Kix
                H_i[:,k]=np.squeeze(np.dot( np.linalg.inv(np.dot(cnmf_kt_c.W.T, cnmf_kt_c.G)+ lamb*np.identity(bases)), np.dot(cnmf_kt_c.G.T, Ktx) ))
                k+=1
            #Pre-image reconstruction
            num_atts = MF_ci.shape[1]
            num_bases = 19
            MFpreim_ci = np.copy(MF_ci)            
            for j in range(len(idx_inc)):
                print 'Compute pre-image for object',idx_inc[j]
                alpha = np.dot(cnmf_kt_c.W, H_i[:,j]) #np.mean(cnmf_kg.H,0)
                MFpreim_ci[idx_inc[j]], trainErrorskl = kf.preimage_faces(kf.MF[idx_com], alpha, idx_inc[j],'cmu_softxrbfx3xlinear', sigmal,sigmap)    
            toc = time.clock()#get final time
            timec = toc - tic#log
            #compute error
            errorc = np.linalg.norm(MF[idx_inc] - MFpreim_ci[idx_inc])/np.linalg.norm(MF[idx_inc])
            errora = np.linalg.norm(MF - MFpreim_ci)/np.linalg.norm(MF)
            errori = np.linalg.norm(MF_ci - MFpreim_ci)/np.linalg.norm(MF_ci)
            #print 'Linear', rt_obinc, timec, errorc, errora, errori
            with open(filename, 'a') as text_file:
                text_file.write("%s ; %f ; %f ; %f ; %f; %f; %f\n" % ('cmu_softxrbfx3xlinear', rt_obinc, timec, errorc, errora, errori, rse_fs))

                
def test_cmu2(MF, sigmal, sigmap, sigmai, bases, type_h):
    filename = r'/home/rajaquep/outcomes/cmu_faces_11nov2016.csv'
    with open(filename, 'a') as text_file:
        text_file.write("kernel ; missing rate ; time ; rse completion ; rse whole; rse whole_inc; rse feature_space\n")

    #Script to run experiments
    for i in range(0,10):
        m_rate = [.1, .40, .50, .60, .70, .80, .90, .92, .94, .96, .98]
        for rt_obinc in m_rate:
            tic = time.clock()#get start time
            MF_ci,idx_inc = incomplete_data(MF, rt_obinc)
            idx_com = np.arange(0,MF.shape[0]-1)
            idx_com = np.delete(idx_com, idx_inc)
            #with open(filename, 'a') as text_file:
                #text_file.write("// Iteration %i; Missing rate = %f; RSE (X, X_ic) = %f\n" % (i, rt_obinc, np.linalg.norm(MF - MF_ci)/np.linalg.norm(MF)))
            kf = KF(MF, idx_com)
            #Kernel for subject
            Ks = kf.soft_kernel( indx=idx_com )
            #Kernel for illumination            
            Kl = kernel._get_kernel(kf.ML[idx_com], kf.ML[idx_com], 'rbf', gamma=(2*sigmal)**-2)
            #Kernel for pose
            Kp = kernel._get_kernel(kf.MP[idx_com], kf.MP[idx_com], 'rbf', gamma=(2*sigmap)**-2)
            #Kernel for image
            Ki = kernel._get_kernel(kf.MF[idx_com], kf.MF[idx_com], 'rbf', gamma=(2*sigmai)**-2)
            #Kt = np.dot(np.dot(np.dot(Ks,Kp),Kl),Ki)
            Kt = Ks*Kp*Kl*Ki
            #Factorize
            cnmf_kt_c = CNMF(Kt, num_bases=bases)
            cnmf_kt_c.factorize(niter=10)
            rse_fs = np.linalg.norm(Kt - np.dot(cnmf_kt_c.W,cnmf_kt_c.H))/np.linalg.norm(Kt)#log
            #compute hj for each incomplete object
            lamb = 1e-10
            H_i = np.zeros((bases, len(idx_inc)))
            k=0
            if type_h == 1:
                auxpi1 = np.linalg.pinv(cnmf_kt_c.W)
            elif type_h == 2:
                auxpi = np.dot( np.dot(cnmf_kt_c.W.T, cnmf_kt_c.W)+ lamb*np.identity(bases),cnmf_kt_c.W.T )
            elif type_h == 3:
                auxpi = np.dot(cnmf_kt_c.W.T, np.linalg.inv( np.dot(cnmf_kt_c.W,cnmf_kt_c.W.T)+ lamb*np.identity(cnmf_kt_c.W.shape[0]) ) )
            elif type_h == 4:
                auxpi = np.linalg.inv(np.dot(cnmf_kt_c.W.T, cnmf_kt_c.G)+ lamb*np.identity(bases))            
            for j in idx_inc:
                #Kernel for illumination
                Klx = kernel._get_kernel(kf.ML[idx_com], kf.ML[j],'rbf', gamma=(2*sigmal)**-2)
                #Kernel for pose
                Kpx = kernel._get_kernel(kf.MP[idx_com], kf.MP[j],'rbf', gamma=(2*sigmap)**-2)
                #Kernel for image
                Kix = kernel._get_kernel(kf.MF[idx_com], MF_ci[j],'rbf', gamma=(2*sigmai)**-2)
                #Kernle for subject
                #...
                Ksx = np.zeros((len(idx_com),1))
                for i in range(len(idx_com)):
                    if kf.MS[i]==kf.MS[j]: Ksx[i] = kf.eq
                    else: Ksx[i] = kf.diff
                #print Ksx.shape  
                Ktx = Ksx*Kpx*Klx*Kix
                if type_h == 1:
                    H_i[:,k] = np.squeeze(np.dot(auxpi,Ktx))
                elif type_h == 2:
                    H_i[:,k] = np.squeeze(np.dot(auxpi,Ktx))
                elif type_h == 3:
                    H_i[:,k] = np.squeeze(np.dot(auxpi,Ktx))
                elif type_h == 4:
                    H_i[:,k] = np.squeeze(np.dot(auxpi, np.dot(cnmf_kt_c.G.T, Ktx) ))
                k+=1
            #Pre-image reconstruction
            num_atts = MF_ci.shape[1]
            num_bases = bases
            MFpreim_ci = np.copy(MF_ci)            
            for j in range(len(idx_inc)):
                #print 'Compute pre-image for object',idx_inc[j]
                alpha = np.dot(cnmf_kt_c.W, H_i[:,j]) #np.mean(cnmf_kg.H,0)
                MFpreim_ci[idx_inc[j]], trainErrorskl = kernel.preimage(MF[idx_com], alpha, 'rbf', gamma=(2*sigmal)**-2+(2*sigmap)**-2+(2*sigmai)**-2)
            toc = time.clock()#get final time
            timec = toc - tic#log
            #compute error
            errorc = np.linalg.norm(MF[idx_inc] - MFpreim_ci[idx_inc])/np.linalg.norm(MF[idx_inc])
            errora = np.linalg.norm(MF - MFpreim_ci)/np.linalg.norm(MF)
            errori = np.linalg.norm(MF_ci - MFpreim_ci)/np.linalg.norm(MF_ci)
            #print 'Linear', rt_obinc, timec, errorc, errora, errori
            with open(filename, 'a') as text_file:
                text_file.write("%s ; %f ; %f ; %f ; %f; %f; %f\n" % ('cmu_rbfx4', rt_obinc, timec, errorc, errora, errori, rse_fs))
                
def param_exp_sigma3(MF, bases, rt_obinc, type_h):
    filename = r'/home/rajaquep/outcomes/cmu_faces_param_29nov2016.csv'
    with open(filename, 'a') as text_file:
        text_file.write("kernel ; sigmas; sigmal; sigmap; sigmai; missing rate ; time ; rse completion ; rse whole; rse whole_inc; rse feature_space\n")
    MF_ci,idx_inc = incomplete_data(MF, rt_obinc)
    idx_com = np.arange(0,MF.shape[0]-1)
    idx_com = np.delete(idx_com, idx_inc)
    #Subject
    eq = 1
    diff = 0.7
    poses = [175, 95, 93, 88, 10, 5, 180, 90, 80, 2, 170]
    illuminations = np.arange(1,22)
    TS = np.zeros((30,11,21))
    for i in range(30):
        for j in range(len(poses)):
            for k in range(len(illuminations)):
                TS[i,j,k] = i

    MS = np.reshape(TS,(30*11*21,1))
    #Kernel for subject
    Ks = np.zeros((len(idx_com),len(idx_com)))
    for i in range(len(idx_com)):
        for j in range(len(idx_com)):
            if MS[i]==MS[j]:
                Ks[i,j] = eq
            else:
                Ks[i,j] = diff
    #illumination
    TL_ = np.repeat(np.array(np.arange(1,22), ndmin = 2),11,axis=0)
    TL = np.repeat(np.array(TL_,ndmin = 3),30,axis=0)
    ML = np.reshape(TL,(30*11*21,1))
    ML_c = ML[idx_com]
    
    TP_ = np.repeat(np.array([175, 95, 93, 88, 10, 5, 180, 90, 80, 2, 170], ndmin = 2),21,axis=0)
    TP = np.repeat(np.array(TP_.T,ndmin = 3),30,axis=0)
    MP = np.reshape(TP,(30*11*21,1))
    MP_c = MP[idx_com]
    
    sigmas = 10.05
    #sigmavl = np.array([0.001, 0.1, 1.5, 5.2, 10.5, 15.5, 20.5, 50.5, 100.5, 150.5, 1000.0 ])
    sigmavl = np.array([20.5])
    sigmavp = np.array([50.5, 100.5, 150.5, 200.0, 500.0, 1000.0 ])
    sigmavi = np.array([50.0])
    for sigmal in sigmavl:
        for sigmap in sigmavp:
            for sigmai in sigmavi:
                Kl = kernel._get_kernel(ML_c, ML_c,'rbf', gamma=(2*sigmal)**-2)
                Kp = kernel._get_kernel(MP_c, MP_c,'rbf', gamma=(2*sigmap)**-2)
                Ki = kernel._get_kernel(MF[idx_com], MF[idx_com],'rbf', gamma=(2*sigmai)**-2)
                #Kt = np.dot(np.dot(np.dot(Ks,Kp),Kl),Ki)
                Kt = Ks*Kp*Kl*Ki
                tic = time.clock()#get start time
                cnmf_kt_c = KCNMF(Kt, num_bases=bases)
                cnmf_kt_c.factorize(niter=10)
                rse_fs = np.linalg.norm(Kt - np.dot(cnmf_kt_c.W,cnmf_kt_c.H))/np.linalg.norm(Kt)
                
                #compute hj for each incomplete object
                lamb = 1e-10
                H_i = np.zeros((bases, len(idx_inc)))
                k=0
                if type_h == 1:
                    auxpi1 = np.linalg.pinv(cnmf_kt_c.W)
                elif type_h == 2:
                    auxpi = np.dot( np.dot(cnmf_kt_c.W.T, cnmf_kt_c.W)+ lamb*np.identity(bases),cnmf_kt_c.W.T )
                elif type_h == 3:
                    auxpi = np.dot(cnmf_kt_c.W.T, np.linalg.inv( np.dot(cnmf_kt_c.W,cnmf_kt_c.W.T)+ lamb*np.identity(cnmf_kt_c.W.shape[0]) ) )
                elif type_h == 4:
                    auxpi = np.linalg.inv(np.dot(cnmf_kt_c.W.T, cnmf_kt_c.G)+ lamb*np.identity(bases))
                for j in idx_inc:
                    #Kernel for illumination
                    Klx = kernel._get_kernel(ML_c, ML[j],'rbf', gamma=(2*sigmal)**-2)
                    #Kernel for pose
                    Kpx = kernel._get_kernel(MP_c, MP[j],'rbf', gamma=(2*sigmap)**-2)
                    #Kernel for image
                    Kix = kernel._get_kernel(MF[idx_com], MF_ci[j],'rbf', gamma=(2*sigmai)**-2)
                    #Kernle for subject
                    #...
                    Ksx = np.zeros((len(idx_com),1))
                    for i in range(len(idx_com)):
                        if MS[i]==MS[j]: Ksx[i] = eq
                        else: Ksx[i] = diff
                    #print Ksx.shape  
                    Ktx = Ksx*Kpx*Klx*Kix
                    if type_h == 1:
                        H_i[:,k] = np.squeeze(np.dot(auxpi,Ktx))
                    elif type_h == 2:
                        H_i[:,k] = np.squeeze(np.dot(auxpi,Ktx))
                    elif type_h == 3:
                        H_i[:,k] = np.squeeze(np.dot(auxpi,Ktx))
                    elif type_h == 4:
                        H_i[:,k] = np.squeeze(np.dot(auxpi, np.dot(cnmf_kt_c.G.T, Ktx) ))
                    k+=1
                #Pre-image reconstruction
                num_atts = MF_ci.shape[1]
                num_bases = bases
                MFpreim_ci = np.copy(MF_ci)            
                for j in range(len(idx_inc)):
                    #print 'Compute pre-image for object',idx_inc[j]
                    alpha = np.dot(cnmf_kt_c.G, H_i[:,j]) #np.mean(cnmf_kg.H,0)
                    try:
                        MFpreim_ci[idx_inc[j]], trainErrorskl = kernel.preimage(MF[idx_com], alpha, 'rbf', gamma=(2*sigmas)**-2+(2*sigmal)**-2+(2*sigmap)**-2+(2*sigmai)**-2)
                    except:
                        print 'Problems computing preimage', j
                        continue
                    
                toc = time.clock()#get final time 
                #Results
                errorc = np.linalg.norm(MF[idx_inc] - MFpreim_ci[idx_inc])/np.linalg.norm(MF[idx_inc])
                errora = np.linalg.norm(MF - MFpreim_ci)/np.linalg.norm(MF)
                errori = np.linalg.norm(MF_ci - MFpreim_ci)/np.linalg.norm(MF_ci)
                #Cualitative results sample
                f, axarr = plt.subplots(2, 6)
                axarr[0,0].imshow( np.reshape(MFpreim_ci[idx_inc[8],:],(32,32)).T, cmap='Greys_r' )
                axarr[0,1].imshow( np.reshape(MFpreim_ci[idx_inc[36],:],(32,32)).T, cmap='Greys_r' )
                axarr[0,2].imshow( np.reshape(MFpreim_ci[idx_inc[100],:],(32,32)).T, cmap='Greys_r' )
                axarr[0,3].imshow( np.reshape(MFpreim_ci[idx_inc[600],:],(32,32)).T, cmap='Greys_r' )
                axarr[0,4].imshow( np.reshape(MFpreim_ci[idx_inc[2000],:],(32,32)).T, cmap='Greys_r' )
                axarr[0,5].imshow( np.reshape(MFpreim_ci[idx_inc[2100],:],(32,32)).T, cmap='Greys_r' )
                axarr[1,0].imshow( np.reshape(MF[idx_inc[8],:],(32,32)).T, cmap='Greys_r' )
                axarr[1,1].imshow( np.reshape(MF[idx_inc[36],:],(32,32)).T, cmap='Greys_r' )
                axarr[1,2].imshow( np.reshape(MF[idx_inc[100],:],(32,32)).T, cmap='Greys_r' )
                axarr[1,3].imshow( np.reshape(MF[idx_inc[600],:],(32,32)).T, cmap='Greys_r' )
                axarr[1,4].imshow( np.reshape(MF[idx_inc[2000],:],(32,32)).T, cmap='Greys_r' )
                axarr[1,5].imshow( np.reshape(MF[idx_inc[2100],:],(32,32)).T, cmap='Greys_r' )
                f.set_size_inches(15,8)
                figName = str(sigmal)+str(sigmap)+str(sigmai)
                f.savefig('/home/rajaquep/outcomes/'+figName +'.png')
                timec = toc - tic
                with open(filename, 'a') as text_file:
                    text_file.write("%s ; %f ; %f ; %f ; %f ; %f ; %f ; %f ; %f; %f; %f\n" % ('cmu_rbfx4', sigmas, sigmal, sigmap, sigmai, rt_obinc, timec, errorc, errora, errori, rse_fs))

                    
def main_experiments():
    import matplotlib
    import numpy as np
    import scipy.io as sio #to load mat files
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    print sio.whosmat('../datasets/CMU(30x11x21x1024).mat')
    matTF = sio.loadmat('../datasets/CMU(30x11x21x1024).mat')#loading tensor of formatted faces' images
    TF = matTF['FullTensor']
    print TF.shape
    MF = np.reshape(TF,(30*11*21,1024))
    print MF.shape
    
    rt_obinc = 0.4 # rate of objects incomplete
    rt_incpob = 0.5 # rate of minimun elements (attributes or variables) incomplete for each object
    idx_inc = np.sort(np.random.choice(MF.shape[0], MF.shape[0]*rt_obinc, replace=False))
    idx_v = np.ones(MF.shape[0])
    idx_v[idx_inc] = 0
    idx_s = np.vstack((idx_v,idx_v)).repeat([1,MF.shape[1]-1],0).T
    idx_com = np.arange(0,MF.shape[0]-1)
    idx_com = np.delete(idx_com, idx_inc)
    
    idx_s2 = np.copy(idx_s) # idx_s2 could have partially incomplete objects
    for i in idx_inc:
        idx_s2[i] = np.random.permutation(np.append(np.random.randint(2,size=MF.shape[1]*(1-rt_incpob)),np.zeros(MF.shape[1]*rt_incpob)))
    
    MF_ci = MF * idx_s2
    MF_i = MF_ci[idx_inc]#subset with incomplete elements
    MF_c = MF_ci[idx_com]#subset with complete elements
    
    #subject
    vdiff=[0.3, 0.7]
    #pose
    vsigmap=[0.01, 5, 50, 150.5, 350.5]
    #image
    vsigmai=[10.0, 100.0, 500.0, 1000.0, 1100.0, 1500.0, 2000.0]
    #mask
    vmask=[0,1]
    #normalize
    vnormalize = [0, 1, 2]

    training = np.arange(10,MF.shape[0],21)
    test = np.arange(15,MF.shape[0],21)#fixing illumination

    testTensorialKspi(MF, idx_s2, vdiff, vsigmap, vsigmai, vmask, vnormalize, training, test)

                    
def testTensorialKspi(MF, idx_s2, vdiff, vsigmal, vsigmai, vmask, vnormalize, training, test, countexp):
    filename = r'/home/rajaquep/outcomes/cmu_faces_13012017.csv'
    with open(filename, 'a') as text_file:
        text_file.write("experiment; diff; sigmap; sigmal; sigmai; mask; normalize; rse kcnmf; rse test\n")
    #kernels parameters (fixed)
    eq = 1
    
    #sigmal = 20.5 #100.0#1.6#
    sigmap = 350.5

    chooseks = 1
    choosekl = 1
    choosekp = 1
    chooseki = 1

    bases = 30
    #--------------------
    MF_ci = MF * idx_s2
    #sample
    #30x11x21x1024
    completes = training
    MF_sm = MF[completes]
    nobj = len(completes)
    #print '#objects:', nobj
    #countexp = 0
    #Compute kernels
    for diff in vdiff:
        for sigmal in vsigmal:
            for sigmai in vsigmai:
                for mask in vmask:
                    for normalize in vnormalize:
                        print countexp
                        #Kernels
                        poses = [175, 95, 93, 88, 10, 5, 180, 90, 80, 2, 170]
                        illuminations = np.arange(1,22)
                        Ksm_g = np.ones((nobj,nobj))

                        if chooseks == 1:
                            #Kernel for subject
                            #30x11x21x1024
                            TS = np.zeros((30,11,21))
                            for i in range(30):
                                for j in range(len(poses)):
                                    for k in range(len(illuminations)):
                                        TS[i,j,k] = i

                            MS = np.reshape(TS,(30*11*21,1))
                            #Ks = np.identity(nobj)*eq
                            #Ks[Ks==0] = diff
                            Ks = np.zeros((nobj,nobj))
                            for i in xrange(len(completes)):
                                for j in xrange(i, len(completes)):
                                    if MS[completes[i]]==MS[completes[j]]:
                                        Ks[i,j], Ks[j,i] = eq,eq
                                    else:
                                        Ks[i,j], Ks[j,i] = diff,diff
                            #print 'Ks size:', Ks.shape
                            Ksm_g *= Ks
                            #Ksm_g = Ks
                        if choosekl == 1:
                            #Illumination Kernel
                            TL_ = np.repeat(np.array(np.arange(1,22), ndmin = 2),11,axis=0)
                            TL = np.repeat(np.array(TL_,ndmin = 3),30,axis=0)
                            ML = np.reshape(TL,(30*11*21,1))
                            ML_c = ML[completes]
                            Kl = kernel._get_kernel(ML_c, ML_c,'rbf', gamma=(2*sigmal)**-2)    
                            #print 'Kl size:', Kl.shape
                            Ksm_g *= Kl
                        if choosekp == 1:
                            TP_ = np.repeat(np.array([175, 95, 93, 88, 10, 5, 180, 90, 80, 2, 170], ndmin = 2),21,axis=0)
                            TP = np.repeat(np.array(TP_.T,ndmin = 3),30,axis=0)
                            MP = np.reshape(TP,(30*11*21,1))
                            MP_c = MP[completes]
                            Kp = kernel._get_kernel(MP_c, MP_c,'rbf', gamma=(2*sigmap)**-2)
                            #print 'Kp size:', Kp.shape
                            Ksm_g *= Kp
                            #Ksm_g = np.dot(Ksm_g,Kp) 
                        if chooseki == 1:    
                            #image kernel    
                            Ki = kernel._get_kernel(MF_sm, MF_sm, 'rbf', gamma=(2*sigmai)**-2)
                            #print 'Ki size:', Ki.shape
                            Ksm_g *= Ki
                        #tensorial kernel
                        #Ksm_g

                        #Factorize
                        cnmf_ksmg = KCNMF(Ksm_g,num_bases=bases)
                        cnmf_ksmg.factorize(niter=10)
                        #print '(rbf kernel) rse:',np.linalg.norm(Ksm_g - np.dot(cnmf_ksmg.W,cnmf_ksmg.H))/np.linalg.norm(Ksm_g)
                        rse_kcnmf = np.linalg.norm(Ksm_g - np.dot(cnmf_ksmg.W,cnmf_ksmg.H))/np.linalg.norm(Ksm_g)
                        f, axarr = plt.subplots(1, 1+chooseks+choosekl+choosekp+chooseki)
                        i=0
                        if chooseks == 1:
                            axarr[i].pcolor(Ks)
                            axarr[i].set_title('Subject')
                            i+=1
                        if choosekl == 1:    
                            axarr[i].pcolor(Ki)
                            axarr[i].set_title('Image')
                            i+=1
                        if choosekp == 1:    
                            axarr[i].pcolor(Kp)
                            axarr[i].set_title('Pose')
                            i+=1
                        if chooseki == 1:    
                            axarr[i].pcolor(Ki)
                            axarr[i].set_title('Image')
                            i+=1
                        axarr[i].pcolor(Ksm_g)
                        axarr[i].set_title('Tensorial')

                        f.set_size_inches(19,3)
                        plt.pcolor(Ksm_g)
                        plt.colorbar()
                        figName = str(countexp)+str('_13012017_')+'kernel'
                        f.savefig('/home/rajaquep/outcomes/'+figName +'.png')
                        #Faces for Training
                        fig = plt.figure(figsize=(19,3))
                        plt.imshow( np.reshape(MF_sm[range(30)],(30*32,32)).T, cmap='Greys_r' )
                        plt.title('Training')
                        figName = str(countexp)+str('_13012017_')+'training'
                        fig.savefig('/home/rajaquep/outcomes/'+figName +'.png')
                        #Choosing 30 objects for testing
                        incompletes = test
                        incompletes = incompletes[range(30)]
                        #Reconstruction
                        lamb = 1e-10
                        logFile = '/home/rajaquep/outcomes/'+str(countexp)+str('_13012017_')+'log.txt'
                        with open(logFile, 'a') as log_file:
                            log_file.write("Computing Hs...")
                        H4_i = np.zeros((bases, len(incompletes)))
                        for i in range(len(incompletes)):
                            auxpi4 = np.linalg.inv(np.dot(cnmf_ksmg.W.T, cnmf_ksmg.G)+ lamb*np.identity(bases))
                            Ktx = np.ones((nobj,1))
                            gamma_t = 0 #preimage
                            if chooseks == 1:
                                Ksx = np.zeros((nobj,1))
                                for j in range(nobj):
                                    if MS[incompletes[i]]==MS[completes[j]]: Ksx[j] = eq
                                    else: Ksx[j] = diff    
                                Ktx *= Ksx
                                #gamma_t += (2*sigmas)**-2#preimage
                            if choosekl == 1:
                                Klx = kernel._get_kernel(ML_c, ML[i],'rbf', gamma=(2*sigmal)**-2)
                                Ktx *= Klx
                                gamma_t += (2*sigmal)**-2#preimage
                            if choosekp == 1:
                                Kpx = kernel._get_kernel(MP_c, MP[i],'rbf', gamma=(2*sigmap)**-2)
                                Ktx *= Kpx
                                gamma_t += (2*sigmap)**-2#preimage
                            if chooseki == 1:
                                if mask == 1:
                                    Kix = kernel._get_kernel(idx_s2[incompletes[i]]*MF_sm, MF_ci[incompletes[i]],'rbf', gamma=(2*sigmai)**-2)#complete image
                                else:
                                    Kix = kernel._get_kernel(MF_sm, MF_ci[incompletes[i]], 'rbf', gamma=(2*sigmai)**-2)
                                #print 'Ki:', Kix, Kix2
                                Ktx *= Kix
                                gamma_t += (2*sigmai)**-2#preimage
                            #Ktx = Ksx+Kix
                            H4_i[:,i] = np.squeeze(np.dot(auxpi4, np.dot(cnmf_ksmg.G.T, Ktx) ))
                            if normalize==1:
                                H4_i[:,i] = (H4_i[:,i]+abs(min(H4_i[:,i]))) / sum(H4_i[:,i]+abs(min(H4_i[:,i])))
                            if normalize ==2:
                                H4_i[:,i] = abs(H4_i[:,i])
                        with open(logFile, 'a') as log_file:
                            log_file.write("Hs computed \nComputing preimage...")
                        MFpreim_ci = np.copy(MF_ci)
                        rse_f = []
                        for i in range(len(incompletes)):                            
                            alpha = np.dot(cnmf_ksmg.G, H4_i[:,i]) #np.mean(cnmf_kg.H,0)        
                            try:        
                                MFpreim_ci[incompletes[i]], trainErrorskl = kernel.preimage(MF_sm, alpha, 'rbf', gamma=gamma_t)
                                rse_f.append(np.linalg.norm(MF[incompletes[i]] - MFpreim_ci[incompletes[i]])/np.linalg.norm(MF[incompletes[i]]))
                                with open(logFile, 'a') as log_file:                            
                                    log_file.write("Computed pre-image for object %i, %i\n" % (i, incompletes[i]))
                            except:
                                with open(logFile, 'a') as log_file:                            
                                    log_file.write("Problems computing preimage %i, %i\n" % (i, incompletes[i]))
                                continue
                        with open(logFile, 'a') as log_file:
                            log_file.write("...pre-images computed")                        
                        #plt.pcolor(H4_i)
                        #plt.colorbar()
                        #figName = str(countexp)+str('_13012017_')+'H'
                        #plt.savefig('/home/rajaquep/outcomes/'+figName +'.png')
                        #outcomes
                        f, axarr = plt.subplots(3, 1)
                        axarr[0].imshow( np.reshape(MF[incompletes],(len(incompletes)*32,32)).T, cmap='Greys_r' )
                        axarr[0].set_title('Original')
                        axarr[1].imshow( np.reshape(MF_ci[incompletes],(len(incompletes)*32,32)).T, cmap='Greys_r' )
                        axarr[1].set_title('Incomplete')
                        axarr[2].imshow( np.reshape(MFpreim_ci[incompletes],(len(incompletes)*32,32)).T, cmap='Greys_r' )
                        axarr[2].set_title('Reconstruction')
                        f.set_size_inches(19,3)
                        figName = str(countexp)+str('_13012017_')+'test'
                        f.savefig('/home/rajaquep/outcomes/'+figName +'.png')
                        with open(filename, 'a') as text_file:
                            text_file.write("%i, %f, %f, %f, %f, %f, %f, %f, %f\n" % (countexp, diff, sigmap, sigmal, sigmai, mask, normalize, rse_kcnmf, np.linalg.norm(MF[incompletes] - MFpreim_ci[incompletes])/np.linalg.norm(MF[incompletes])))                        
                        countexp+=1

def test_cmu3(MF, iterations, vrates, vsigmal, vsigmai, vnormalize, countexp):
    filename = r'/home/rajaquep/outcomes/cmu_faces_22012017.csv'
    with open(filename, 'a') as text_file:
        text_file.write("experiment; rate; sigmap; sigmal; sigmai; mask; normalize; rse kcnmf; rse test; rse test**\n")    
    #kernels parameters (fixed)
    eq = 1
    diff = .3
    
    sigmap = 350.5

    chooseks = 1
    choosekl = 1
    choosekp = 1
    chooseki = 1
    
    mask = 1

    bases = 30
    #--------------------
    
    for rt_obinc in vrates:
        MF_ci,incompletes,completes,idx_s2 = incomplete_data(MF, rt_obinc)        
        MF_sm = MF[completes]
        nobj = len(completes)    
        for sigmal in vsigmal:
            for sigmai in vsigmai:
                for normalize in vnormalize:
                    for it in range(iterations):                    
                        logFile = '/home/rajaquep/outcomes/'+str(countexp)+str('_22012017_')+'log.txt'
                        with open(logFile, 'a') as log_file:
                            log_file.write("Computing Experiment %i\n"%(countexp))
                        print countexp
                        #Kernels
                        poses = [175, 95, 93, 88, 10, 5, 180, 90, 80, 2, 170]
                        illuminations = np.arange(1,22)
                        Ksm_g = np.ones((nobj,nobj))

                        if chooseks == 1:
                            #Kernel for subject
                            #30x11x21x1024
                            TS = np.zeros((30,11,21))
                            for i in range(30):
                                for j in range(len(poses)):
                                    for k in range(len(illuminations)):
                                        TS[i,j,k] = i

                            MS = np.reshape(TS,(30*11*21,1))
                            #Ks = np.identity(nobj)*eq
                            #Ks[Ks==0] = diff
                            Ks = np.zeros((nobj,nobj))
                            for i in xrange(len(completes)):
                                for j in xrange(i, len(completes)):
                                    if MS[completes[i]]==MS[completes[j]]:
                                        Ks[i,j], Ks[j,i] = eq,eq
                                    else:
                                        Ks[i,j], Ks[j,i] = diff,diff
                            #print 'Ks size:', Ks.shape
                            Ksm_g *= Ks
                            #Ksm_g = Ks
                        if choosekl == 1:
                            #Illumination Kernel
                            TL_ = np.repeat(np.array(np.arange(1,22), ndmin = 2),11,axis=0)
                            TL = np.repeat(np.array(TL_,ndmin = 3),30,axis=0)
                            ML = np.reshape(TL,(30*11*21,1))
                            ML_c = ML[completes]
                            Kl = kernel._get_kernel(ML_c, ML_c,'rbf', gamma=(2*sigmal)**-2)    
                            #print 'Kl size:', Kl.shape
                            Ksm_g *= Kl
                        if choosekp == 1:
                            TP_ = np.repeat(np.array([175, 95, 93, 88, 10, 5, 180, 90, 80, 2, 170], ndmin = 2),21,axis=0)
                            TP = np.repeat(np.array(TP_.T,ndmin = 3),30,axis=0)
                            MP = np.reshape(TP,(30*11*21,1))
                            MP_c = MP[completes]
                            Kp = kernel._get_kernel(MP_c, MP_c,'rbf', gamma=(2*sigmap)**-2)
                            #print 'Kp size:', Kp.shape
                            Ksm_g *= Kp
                            #Ksm_g = np.dot(Ksm_g,Kp) 
                        if chooseki == 1:    
                            #image kernel    
                            Ki = kernel._get_kernel(MF_sm, MF_sm, 'rbf', gamma=(2*sigmai)**-2)
                            #print 'Ki size:', Ki.shape
                            Ksm_g *= Ki
                        #tensorial kernel
                        #Ksm_g

                        #Factorize
                        cnmf_ksmg = KCNMF(Ksm_g,num_bases=bases)
                        cnmf_ksmg.factorize(niter=10)
                        #print '(rbf kernel) rse:',np.linalg.norm(Ksm_g - np.dot(cnmf_ksmg.W,cnmf_ksmg.H))/np.linalg.norm(Ksm_g)
                        rse_kcnmf = np.linalg.norm(Ksm_g - np.dot(cnmf_ksmg.W,cnmf_ksmg.H))/np.linalg.norm(Ksm_g)
                        
                        #Reconstruction
                        lamb = 1e-10                        
                        with open(logFile, 'a') as log_file:
                            log_file.write("Computing Hs...\n")
                        H4_i = np.zeros((bases, len(incompletes)))
                        for i in range(len(incompletes)):
                            auxpi4 = np.linalg.inv(np.dot(cnmf_ksmg.W.T, cnmf_ksmg.G)+ lamb*np.identity(bases))
                            Ktx = np.ones((nobj,1))
                            gamma_t = 0 #preimage
                            if chooseks == 1:
                                Ksx = np.zeros((nobj,1))
                                for j in range(nobj):
                                    if MS[incompletes[i]]==MS[completes[j]]: Ksx[j] = eq
                                    else: Ksx[j] = diff    
                                Ktx *= Ksx
                                #gamma_t += (2*sigmas)**-2#preimage
                            if choosekl == 1:
                                Klx = kernel._get_kernel(ML_c, ML[i],'rbf', gamma=(2*sigmal)**-2)
                                Ktx *= Klx
                                gamma_t += (2*sigmal)**-2#preimage
                            if choosekp == 1:
                                Kpx = kernel._get_kernel(MP_c, MP[i],'rbf', gamma=(2*sigmap)**-2)
                                Ktx *= Kpx
                                gamma_t += (2*sigmap)**-2#preimage
                            if chooseki == 1:
                                if mask == 1:
                                    Kix = kernel._get_kernel(idx_s2[incompletes[i]]*MF_sm, MF_ci[incompletes[i]],'rbf', gamma=(2*sigmai)**-2)#complete image
                                else:
                                    Kix = kernel._get_kernel(MF_sm, MF_ci[incompletes[i]], 'rbf', gamma=(2*sigmai)**-2)
                                #print 'Ki:', Kix, Kix2
                                Ktx *= Kix
                                gamma_t += (2*sigmai)**-2#preimage
                            #Ktx = Ksx+Kix
                            H4_i[:,i] = np.squeeze(np.dot(auxpi4, np.dot(cnmf_ksmg.G.T, Ktx) ))
                            if normalize==1:
                                H4_i[:,i] = (H4_i[:,i]+abs(min(H4_i[:,i]))) / sum(H4_i[:,i]+abs(min(H4_i[:,i])))
                            if normalize ==2:
                                H4_i[:,i] = abs(H4_i[:,i])
                            if normalize == 3:
                                H4_i[:,i] = np.maximum(H4_i[:,i],0)
                        with open(logFile, 'a') as log_file:
                            log_file.write("Hs computed \nComputing preimage...\n")
                        MFpreim_ci = np.copy(MF_ci)
                        rse_f = []
                        for i in range(len(incompletes)):                            
                            alpha = np.dot(cnmf_ksmg.G, H4_i[:,i]) #np.mean(cnmf_kg.H,0)        
                            try:        
                                MFpreim_ci[incompletes[i]], trainErrorskl = kernel.preimage(MF_sm, alpha, 'rbf', gamma=gamma_t)
                                rse_f.append(np.linalg.norm(MF[incompletes[i]] - MFpreim_ci[incompletes[i]])/np.linalg.norm(MF[incompletes[i]]))
                                with open(logFile, 'a') as log_file:                            
                                    log_file.write("Computed pre-image for object %i, %i\n" % (i, incompletes[i]))
                            except:
                                with open(logFile, 'a') as log_file:                            
                                    log_file.write("Problems computing preimage %i, %i\n" % (i, incompletes[i]))
                                continue
                        with open(logFile, 'a') as log_file:
                            log_file.write("...pre-images computed")
                        rse_test = np.linalg.norm(MF[incompletes] - MFpreim_ci[incompletes])/np.linalg.norm(MF[incompletes])
                        inv_idx2 = idx_s2+1
                        inv_idx2[inv_idx2==2] = 0
                        rse_test2 = np.linalg.norm(MF[incompletes] - ((MFpreim_ci[incompletes]*inv_idx2[incompletes])+MF_ci[incompletes]))/np.linalg.norm(MF[incompletes])
                        with open(filename, 'a') as text_file:
                            text_file.write("%i, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % (countexp, rt_obinc, sigmap, sigmal, sigmai, mask, normalize, rse_kcnmf, rse_test, rse_test2))
                        countexp+=1