# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 16:49:42 2014

@author: aepaezt
"""

import ParameterTuningFunctions as ptf
import h5py

def main():
    """
    Tuning OKMF for Jaffe
    """
    dataset = h5py.File('../Datasets/jaffe.h5')
    k = 10
    Budgets = [10,50,100,150,213]
    Gammas = [1.0,0.9,0.8,0.7]
    Lambdas = [0.0,0.01,0.1,0.2,0.3,0.4]
    Alphas = [0.4,0.5,0.6,0.7]
    Sigmas = range(-10,11,1)
    RL = ptf.LinearTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,10)
    RR = ptf.RBFTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,Sigmas,10)
    KL = ptf.LinearTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,10,K=True)
    KR = ptf.RBFTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,Sigmas,10,K=True)
    dataset.close()
    output = 'Jaffe_Random_Linear ' + str(RL) + '\n'
    output += 'Jaffe_Random_RBF '+ str(RR) + '\n'
    output += 'Jaffe_Kmeans_Linear ' + str(KL) + '\n'
    output += 'Jaffe_Kmeans_RBF '+ str(KR) + '\n'
    print output
    outputFile = open('Jaffe Tuning','w')
    outputFile.write(output)
    outputFile.close()
    del output,k,Budgets,Gammas,Lambdas,Alphas,Sigmas,RL,RR,KL,KR,outputFile
    """
    Tuning OKMF for ATT
    """
    dataset = h5py.File('../Datasets/ATT.h5')
    k = 40
    Budgets = [40,50,100,200,300,400]
    Gammas = [1.0,0.9,0.8,0.7]
    Lambdas = [0.0,0.01,0.1,0.2,0.3,0.4]
    Alphas = [0.4,0.5,0.6,0.7]
    Sigmas = range(-10,11,1)
    RL = ptf.LinearTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,10)
    RR = ptf.RBFTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,Sigmas,10)
    KL = ptf.LinearTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,10,K=True)
    KR = ptf.RBFTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,Sigmas,10,K=True)
    dataset.close()
    output = 'ATT_Random_Linear ' + str(RL) + '\n'
    output += 'ATT_Random_RBF '+ str(RR) + '\n'
    output += 'ATT_Kmeans_Linear ' + str(KL) + '\n'
    output += 'ATT_Kmeans_RBF '+ str(KR) + '\n'
    print output
    outputFile = open('ATT Tuning','w')
    outputFile.write(output)
    outputFile.close()
    del output,Budgets,Gammas,Lambdas,Alphas,Sigmas,RL,RR,KL,KR,outputFile
    """
    Tuning OKMF for AR
    """
    dataset = h5py.File('../Datasets/AR.h5')
    k = 40
    Budgets = [40,50,100,500,1000,2600]
    Gammas = [1.0,0.9,0.8,0.7]
    Lambdas = [0.0,0.01,0.1,0.2,0.3,0.4]
    Alphas = [0.4,0.5,0.6,0.7]
    Sigmas = range(-10,11,1)
    RL = ptf.LinearTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,10)
    RR = ptf.RBFTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,Sigmas,10)
    KL = ptf.LinearTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,10,K=True)
    KR = ptf.RBFTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,Sigmas,10,K=True)
    dataset.close()
    output = 'AR_Random_Linear ' + str(RL) + '\n'
    output += 'AR_Random_RBF '+ str(RR) + '\n'
    output += 'AR_Kmeans_Linear ' + str(KL) + '\n'
    output += 'AR_Kmeans_RBF '+ str(KR) + '\n'
    print output
    outputFile = open('AR Tuning','w')
    outputFile.write(output)
    outputFile.close()
    del output,Budgets,Gammas,Lambdas,Alphas,Sigmas,RL,RR,KL,KR,outputFile
    """
    Tuning OKMF for abalone dataset
    """
    dataset = h5py.File('../Datasets/abalone.h5')
    k = 3
    Budgets = [3,50,500,1000,2000,3000,4177]
    Gammas = [1.0,0.9,0.8,0.7]
    Lambdas = [0.0,0.01,0.1,0.2,0.3,0.4]
    Alphas = [0.4,0.5,0.6,0.7]
    Sigmas = range(-10,11,1)
    RL = ptf.LinearTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,10)
    RR = ptf.RBFTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,Sigmas,10)
    KL = ptf.LinearTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,10,K=True)
    KR = ptf.RBFTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,Sigmas,10,K=True)
    dataset.close()
    output = 'Abalone_Random_Linear ' + str(RL) + '\n'
    output += 'Abalone_Random_RBF '+ str(RR) + '\n'
    output += 'Abalone_Kmeans_Linear ' + str(KL) + '\n'
    output += 'Abalone_Kemans_RBF '+ str(KR) + '\n'
    print output
    outputFile = open('Abalone Tuning','w')
    outputFile.write(output)
    outputFile.close()
    del output,Budgets,Gammas,Lambdas,Alphas,Sigmas,RL,RR,KL,KR,outputFile

def main2():
    """
    Tuning OKMF for Jaffe
    """
    dataset = h5py.File('../Datasets/jaffe.h5')
    k = 10
    Budgets = [10,50,100,150,213]
    Gammas = [1.0,0.9,0.8,0.7]
    Lambdas = [0.0,0.01,0.1,0.2,0.3,0.4]
    Alphas = [0.4,0.5,0.6,0.7]
    Sigmas = range(-10,11,1)
    RL = ptf.LinearTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,10)
    RR = ptf.RBFTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,Sigmas,10)
    KL = ptf.LinearTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,10,K=True)
    KR = ptf.RBFTuning(dataset,k,Budgets,Gammas,Lambdas,Alphas,Sigmas,10,K=True)
    dataset.close()
    output = 'Jaffe_Random_Linear ' + str(RL) + '\n'
    output += 'Jaffe_Random_RBF '+ str(RR) + '\n'
    output += 'Jaffe_Kmeans_Linear ' + str(KL) + '\n'
    output += 'Jaffe_Kmeans_RBF '+ str(KR) + '\n'
    print output
    outputFile = open('Jaffe Tuning','w')
    outputFile.write(output)
    outputFile.close()
    del output,k,Budgets,Gammas,Lambdas,Alphas,Sigmas,RL,RR,KL,KR,outputFile

if __name__ == '__main__':
    main()
