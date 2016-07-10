# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 14:47:30 2015

@author: aepaezt
"""

from ast import literal_eval
from ExperimentFunctions import LinearExperiment,RBFExperiment
import h5py
import numpy as np

def main():
    """
    Jaffe experiment
    """
    # Load parameters
    parameterFile = open('Jaffe Tuning','r')
    randomLinearParameter = parameterFile.readline().replace('Jaffe_Random_Linear ','')
    randomRbfParameter = parameterFile.readline().replace('Jaffe_Random_RBF ','')
    kmeansLinearParameter = parameterFile.readline().replace('Jaffe_Kmeans_Linear ','')
    kmeansRbfParameter = parameterFile.readline().replace('Jaffe_Kmeans_RBF ','')
    parameterFile.close()
    randomLinearParameter = literal_eval(randomLinearParameter)
    randomRbfParameter = literal_eval(randomRbfParameter)
    kmeansLinearParameter = literal_eval(kmeansLinearParameter)
    kmeansRbfParameter = literal_eval(kmeansRbfParameter)
    # Load dataset
    dataset = h5py.File('../Datasets/jaffe.h5')
    k = 10
    # Run experiments
    budget,Gamma,Lambda,Alpha = loadParameters(randomLinearParameter)
    randomLinearResult = LinearExperiment(dataset,k,budget,Gamma,Lambda,Alpha,30)
    budget,Gamma,Lambda,Alpha,Sigma = loadParameters(randomRbfParameter)
    randomRbfResult = RBFExperiment(dataset,k,budget,Gamma,Lambda,Alpha,Sigma,30)
    budget,Gamma,Lambda,Alpha = loadParameters(kmeansLinearParameter)
    kmeansLinearResult = LinearExperiment(dataset,k,budget,Gamma,Lambda,Alpha,30,K=True)
    budget,Gamma,Lambda,Alpha,Sigma = loadParameters(kmeansRbfParameter)
    kmeansRbfResult = RBFExperiment(dataset,k,budget,Gamma,Lambda,Alpha,Sigma,30,K=True)
    dataset.close()
    # Format results
    RL = formatResults(randomLinearResult)
    RR = formatResults(randomRbfResult)
    KL = formatResults(kmeansLinearResult)
    KR = formatResults(kmeansRbfResult)
    output = 'Jaffe_Random_Linear ' + str(RL) + '\n'
    output += 'Jaffe_Random_RBF '+ str(RR) + '\n'
    output += 'Jaffe_Kmeans_Linear ' + str(KL) + '\n'
    output += 'Jaffe_Kmeans_RBF '+ str(KR) + '\n'
    print output
    # Write results
    outputFile = open('Jaffe Results','w')
    outputFile.write(output)
    outputFile.close()
    """
    ATT experiment
    """
    # Load parameters
    parameterFile = open('ATT Tuning','r')
    randomLinearParameter = parameterFile.readline().replace('ATT_Random_Linear ','')
    randomRbfParameter = parameterFile.readline().replace('ATT_Random_RBF ','')
    kmeansLinearParameter = parameterFile.readline().replace('ATT_Kmeans_Linear ','')
    kmeansRbfParameter = parameterFile.readline().replace('ATT_Kmeans_RBF ','')
    parameterFile.close()
    randomLinearParameter = literal_eval(randomLinearParameter)
    randomRbfParameter = literal_eval(randomRbfParameter)
    kmeansLinearParameter = literal_eval(kmeansLinearParameter)
    kmeansRbfParameter = literal_eval(kmeansRbfParameter)
    # Load dataset
    dataset = h5py.File('../Datasets/ATT.h5')
    k = 40
    # Run experiments
    budget,Gamma,Lambda,Alpha = loadParameters(randomLinearParameter)
    randomLinearResult = LinearExperiment(dataset,k,budget,Gamma,Lambda,Alpha,30)
    budget,Gamma,Lambda,Alpha,Sigma = loadParameters(randomRbfParameter)
    randomRbfResult = RBFExperiment(dataset,k,budget,Gamma,Lambda,Alpha,Sigma,30)
    budget,Gamma,Lambda,Alpha = loadParameters(kmeansLinearParameter)
    kmeansLinearResult = LinearExperiment(dataset,k,budget,Gamma,Lambda,Alpha,30,K=True)
    budget,Gamma,Lambda,Alpha,Sigma = loadParameters(kmeansRbfParameter)
    kmeansRbfResult = RBFExperiment(dataset,k,budget,Gamma,Lambda,Alpha,Sigma,30,K=True)
    dataset.close()
    # Format results
    RL = formatResults(randomLinearResult)
    RR = formatResults(randomRbfResult)
    KL = formatResults(kmeansLinearResult)
    KR = formatResults(kmeansRbfResult)
    output = 'ATT_Random_Linear ' + str(RL) + '\n'
    output += 'ATT_Random_RBF '+ str(RR) + '\n'
    output += 'ATT_Kmeans_Linear ' + str(KL) + '\n'
    output += 'ATT_Kmeans_RBF '+ str(KR) + '\n'
    print output
    # Write results
    outputFile = open('ATT Results','w')
    outputFile.write(output)
    outputFile.close()
    """
    AR experiment
    """
    # Load parameters
    parameterFile = open('AR Tuning','r')
    randomLinearParameter = parameterFile.readline().replace('AR_Random_Linear ','')
    randomRbfParameter = parameterFile.readline().replace('AR_Random_RBF ','')
    kmeansLinearParameter = parameterFile.readline().replace('AR_Kmeans_Linear ','')
    kmeansRbfParameter = parameterFile.readline().replace('AR_Kmeans_RBF ','')
    parameterFile.close()
    randomLinearParameter = literal_eval(randomLinearParameter)
    randomRbfParameter = literal_eval(randomRbfParameter)
    kmeansLinearParameter = literal_eval(kmeansLinearParameter)
    kmeansRbfParameter = literal_eval(kmeansRbfParameter)
    # Load dataset
    dataset = h5py.File('../Datasets/AR.h5')
    k = 40
    # Run experiments
    budget,Gamma,Lambda,Alpha = loadParameters(randomLinearParameter)
    randomLinearResult = LinearExperiment(dataset,k,budget,Gamma,Lambda,Alpha,30)
    budget,Gamma,Lambda,Alpha,Sigma = loadParameters(randomRbfParameter)
    randomRbfResult = RBFExperiment(dataset,k,budget,Gamma,Lambda,Alpha,Sigma,30)
    budget,Gamma,Lambda,Alpha = loadParameters(kmeansLinearParameter)
    kmeansLinearResult = LinearExperiment(dataset,k,budget,Gamma,Lambda,Alpha,30,K=True)
    budget,Gamma,Lambda,Alpha,Sigma = loadParameters(kmeansRbfParameter)
    kmeansRbfResult = RBFExperiment(dataset,k,budget,Gamma,Lambda,Alpha,Sigma,30,K=True)
    dataset.close()
    # Format results
    RL = formatResults(randomLinearResult)
    RR = formatResults(randomRbfResult)
    KL = formatResults(kmeansLinearResult)
    KR = formatResults(kmeansRbfResult)
    output = 'AR_Random_Linear ' + str(RL) + '\n'
    output += 'AR_Random_RBF '+ str(RR) + '\n'
    output += 'AR_Kmeans_Linear ' + str(KL) + '\n'
    output += 'AR_Kmeans_RBF '+ str(KR) + '\n'
    print output
    # Write results
    outputFile = open('AR Results','w')
    outputFile.write(output)
    outputFile.close()
    """
    Abalone experiment
    """
    # Load parameters
    parameterFile = open('Abalone Tuning','r')
    randomLinearParameter = parameterFile.readline().replace('Abalone_Random_Linear ','')
    randomRbfParameter = parameterFile.readline().replace('Abalone_Random_RBF ','')
    kmeansLinearParameter = parameterFile.readline().replace('Abalone_Kmeans_Linear ','')
    kmeansRbfParameter = parameterFile.readline().replace('Abalone_Kemans_RBF ','')
    parameterFile.close()
    randomLinearParameter = literal_eval(randomLinearParameter)
    randomRbfParameter = literal_eval(randomRbfParameter)
    kmeansLinearParameter = literal_eval(kmeansLinearParameter)
    kmeansRbfParameter = literal_eval(kmeansRbfParameter)
    # Load dataset
    dataset = h5py.File('../Datasets/abalone.h5')
    k = 3
    # Run experiments
    budget,Gamma,Lambda,Alpha = loadParameters(randomLinearParameter)
    randomLinearResult = LinearExperiment(dataset,k,budget,Gamma,Lambda,Alpha,30)
    budget,Gamma,Lambda,Alpha,Sigma = loadParameters(randomRbfParameter)
    randomRbfResult = RBFExperiment(dataset,k,budget,Gamma,Lambda,Alpha,Sigma,30)
    budget,Gamma,Lambda,Alpha = loadParameters(kmeansLinearParameter)
    kmeansLinearResult = LinearExperiment(dataset,k,budget,Gamma,Lambda,Alpha,30,K=True)
    budget,Gamma,Lambda,Alpha,Sigma = loadParameters(kmeansRbfParameter)
    kmeansRbfResult = RBFExperiment(dataset,k,budget,Gamma,Lambda,Alpha,Sigma,30,K=True)
    dataset.close()
    # Format results
    RL = formatResults(randomLinearResult)
    RR = formatResults(randomRbfResult)
    KL = formatResults(kmeansLinearResult)
    KR = formatResults(kmeansRbfResult)
    output = 'Abalone_Random_Linear ' + str(RL) + '\n'
    output += 'Abalone_Random_RBF '+ str(RR) + '\n'
    output += 'Abalone_Kmeans_Linear ' + str(KL) + '\n'
    output += 'Abalone_Kemans_RBF '+ str(KR) + '\n'
    print output
    # Write results
    outputFile = open('Abalone Results','w')
    outputFile.write(output)
    outputFile.close()

def main2():
    """
    Jaffe experiment
    """
    # Load parameters
    parameterFile = open('Jaffe Tuning','r')
    randomLinearParameter = parameterFile.readline().replace('Jaffe_Random_Linear ','')
    randomRbfParameter = parameterFile.readline().replace('Jaffe_Random_RBF ','')
    kmeansLinearParameter = parameterFile.readline().replace('Jaffe_Kmeans_Linear ','')
    kmeansRbfParameter = parameterFile.readline().replace('Jaffe_Kmeans_RBF ','')
    parameterFile.close()
    randomLinearParameter = literal_eval(randomLinearParameter)
    randomRbfParameter = literal_eval(randomRbfParameter)
    kmeansLinearParameter = literal_eval(kmeansLinearParameter)
    kmeansRbfParameter = literal_eval(kmeansRbfParameter)
    # Load dataset
    dataset = h5py.File('../Datasets/jaffe.h5')
    k = 10
    # Run experiments
    budget,Gamma,Lambda,Alpha = loadParameters(randomLinearParameter)
    randomLinearResult = LinearExperiment(dataset,k,budget,Gamma,Lambda,Alpha,30)
    budget,Gamma,Lambda,Alpha,Sigma = loadParameters(randomRbfParameter)
    randomRbfResult = RBFExperiment(dataset,k,budget,Gamma,Lambda,Alpha,Sigma,30)
    budget,Gamma,Lambda,Alpha = loadParameters(kmeansLinearParameter)
    kmeansLinearResult = LinearExperiment(dataset,k,budget,Gamma,Lambda,Alpha,30,K=True)
    budget,Gamma,Lambda,Alpha,Sigma = loadParameters(kmeansRbfParameter)
    kmeansRbfResult = RBFExperiment(dataset,k,budget,Gamma,Lambda,Alpha,Sigma,30,K=True)
    dataset.close()
    # Format results
    RL = formatResults(randomLinearResult)
    RR = formatResults(randomRbfResult)
    KL = formatResults(kmeansLinearResult)
    KR = formatResults(kmeansRbfResult)
    output = 'Jaffe_Random_Linear ' + str(RL) + '\n'
    output += 'Jaffe_Random_RBF '+ str(RR) + '\n'
    output += 'Jaffe_Kmeans_Linear ' + str(KL) + '\n'
    output += 'Jaffe_Kmeans_RBF '+ str(KR) + '\n'
    print output
    # Write results
    outputFile = open('Jaffe Results','w')
    outputFile.write(output)
    outputFile.close()

def loadParameters(Parameters):
    budget = Parameters['budget']
    Gamma = Parameters['Gamma']
    Lambda = Parameters['Lambda']
    Alpha = Parameters['Alpha']
    if Parameters.has_key('Sigma'):
        Sigma = Parameters['Sigma']
        return budget,Gamma,Lambda,Alpha,Sigma
    else:
        return budget,Gamma,Lambda,Alpha

def formatResults(results):
    result = dict()
    result['avg'] = np.average(results)
    result['std'] = np.std(results)
    result['min'] = np.min(results)
    result['max'] = np.max(results)
    return result

if __name__ == '__main__':
    main()