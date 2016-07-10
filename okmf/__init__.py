# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:57:30 2015

@author: aepaezt
"""

from accuracy import accuracy
from L1Normalization import L1Normalization
import KKmeans
from OKMF import OKMF
import CNMF
try:
    from cudaOKMF import cudaOKMF
except:
    print 'No CUDA support'