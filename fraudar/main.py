import numpy as np
from scipy import sparse
from . import greedy

'''
Fraudar code is from (Hooi et al., 2016)
'''

def run_fraudar(G):
    # G should be size (nrev * npap)
    M = (sparse.coo_matrix(G) > 0).astype('int')
    (finalRowSet, finalColSet), bestAveScore = greedy.logWeightedAveDegree(M)
    return list(finalRowSet), ''
