#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:29:52 2021

@author: danielmckenzie


% Cosamp algorithm
%   Input
%       K : sparsity of Sest
%       Phi : measurement matrix
%       u: measured vector
%       tol : tolerance for approximation between successive solutions. 
%   Output
%       Sest: Solution found by the algorithm
%
% Algorithm as described in "CoSaMP: Iterative signal recovery from 
% incomplete and inaccurate samples" by Deanna Needell and Joel Tropp.
% 


% This implementation was written by David Mary, 
% but modified 20110707 by Bob L. Sturm to make it much clearer,
% and corrected multiple times again and again.
% To begin with, see: http://media.aau.dk/null_space_pursuits/2011/07/ ...
% algorithm-power-hour-compressive-sampling-matching-pursuit-cosamp.html
% Modified slightly for speed by Daniel McKenzie and HanQin Cai in 2020-2021
% This script/program is released under the Commons Creative Licence
% with Attribution Non-commercial Share Alike (by-nc-sa)
% http://creativecommons.org/licenses/by-nc-sa/3.0/
% Short Disclaimer: this script is for educational purpose only.
% Longer Disclaimer see  http://igorcarron.googlepages.com/disclaimer

"""

import numpy as np
import numpy.linalg as la

def cosamp(Phi, u, K, tol, maxiterations):
    
    # Initialization
    
    Sest = np.zeros((np.shape(Phi)[1],1))
    v = u;
    t = 0;
    halt = False
    num_precision = 1e-12
    T = np.array([])
    
    while t<= maxiterations and not halt:
        y = np.abs(np.dot(Phi.T,v))
        
#        target = np.sort(y)[::-1][2*K]
#        Omega = [i for (i, val) in enumerate(y)
#                 if val > target and val > num_precision]
        Omega = np.argpartition(-abs(y), 2*K)
        Omega = Omega[:2*K]
        Omega = Omega[abs(y[Omega]) > num_precision]
        
        T = np.union1d(Omega, T)
        T = T.astype(int)
        b ,_ ,_ ,_ = la.lstsq(Phi[:,T], u, rcond=None)
        
        Kgoodindices = np.argpartition(-abs(b), K)
        Kgoodindices = Kgoodindices[:K]
        Kgoodindices = Kgoodindices[abs(b[Kgoodindices]) > num_precision]
        
        T = T[Kgoodindices]
        Sest = np.zeros(np.shape(Phi)[1])
        b = b[Kgoodindices]
        Sest[T] = b;
        v = u - np.dot(Phi[:,T],b)
        t = t+1
        
        halt = la.norm(v)/la.norm(u) < tol
        
    return Sest
        
        