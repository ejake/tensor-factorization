# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 17:08:22 2014

@author: aepaezt

Datasets experimentation
"""

import numpy as np

def L1Normalization(X):
    """
    Calculate the L1 normalization of a matrix
    
    Parameters
    ----------
    X : ndarray
        A matrix to be normalized
    
    Returns
    -------
    Y : ndarray
        The L1 normalization of matrix X
    """
    result = X / np.dot(np.ones(X.shape),np.diag(np.sum(X,0)))
    return result