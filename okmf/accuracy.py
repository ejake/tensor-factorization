# -*- coding: utf-8 -*-
"""
@author: Esteban Paez Torres

Confusion matrix construction can generate rows or columns of zeros.
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.linear_assignment_ import linear_assignment

def accuracy(l,lg):
    profitMatrix = confusion_matrix(lg,l)
    costMatrix = np.iinfo(np.int64).max - profitMatrix
    ind = linear_assignment(costMatrix)
    total = 0.0
    for i in ind:
        total += profitMatrix[tuple(i)]
    return total / lg.shape[0]

