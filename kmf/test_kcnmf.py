import numpy as np
import kernel as kernel
from kernelFaces import KernelFaces as KF
import pymf
from pymf.cnmf import CNMF 
import time

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
    #idx_com = np.arange(0,MF.shape[0]-1)
    #idx_com = np.delete(idx_com, idx_inc)
    #data_i = MF[idx_inc]#subset with incomplete elements
    #data_c = MF[idx_com]#subset with complete elements
    idx_s2 = np.copy(idx_s) # idx_s2 could have partially incomplete objects
    for i in idx_inc:
        idx_s2[i] = np.random.permutation(np.append(np.random.randint(2,size=X.shape[1]*(1-rt_incpob)),np.zeros(X.shape[1]*rt_incpob)))
    MF_ci = X * idx_s2

    return MF_ci, idx_inc
    
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

                
def test_cmu2(MF, sigmal, sigmap, , sigmai, bases, type_h):
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
            Ki = kernel._get_kernel(kf.MF[idx_com], kf.MF[idx_com], 'rbf', gamma=(2*sigmai)**-2)
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
                    H_i[:,k] = np.dot(np.linalg.pinv(cnmf_kt_c.W),Ktx)
                elif type_h == 2:
                    H_i[:,k] = np.dot(np.dot( np.dot(cnmf_kt_c.W.T, cnmf_kt_c.W)+ lamb*np.identity(bases),cnmf_kt_c.W.T ),Ktx)
                elif type_h == 3:
                    H_i[:,k] = np.dot(np.dot(cnmf_kt_c.W.T, np.linalg.inv( np.dot(cnmf_kt_c.W,cnmf_kt_c.W.T)+ lamb*np.identity(cnmf_kt_c.W.shape[0]) ) ),Ktx)
                elif type_h == 4:
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
