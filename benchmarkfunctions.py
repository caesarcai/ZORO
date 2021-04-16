'''This module contains the following:

Sparse Quadric

Max-k-sum-squared

TODO:
    - Add more test functions.
    - Do ew want to consider multiplicative noise?

'''
import numpy as np
import sys

class SparseQuadric(object):
    '''An implementation of the sparse quadric function.'''
    def __init__(self, n, s, noiseamp):
        self.noiseamp = noiseamp/np.sqrt(n)
        self.s = s
        self.dim = n
        self.rng = np.random.RandomState()
        
    def __call__(self,x):
        f_no_noise = np.dot(x[0:self.s],x[0:self.s])
        return f_no_noise + self.noiseamp*self.rng.randn()
    
class MaxK(object):
    '''An implementation of the max-k-squared-sum function.'''
    def __init__(self, n, s, noiseamp):
        self.noiseamp = noiseamp/np.sqrt(n)
        self.dim = n
        self.s = s
        self.rng = np.random.RandomState()

    def __call__(self, x):

        idx = np.argsort(np.abs(x))
        idx2 = idx[self.dim-self.s:self.dim]
        f_no_noise = np.dot(x[idx2], x[idx2])/2
        return f_no_noise + self.noiseamp*self.rng.randn()
 

