#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:46:35 2021

Code for ZORO, by Cai, McKenzie Yin and Zhang

"""

import numpy as np
from base import BaseOptimizer
from Cosamp import cosamp


class ZORO(BaseOptimizer):
    '''
    ZORO for black box optimization. 
    TODO: 
         - Implement opportunistic sampling
    '''
    
    def __init__(self, x0, f, params, function_budget=10000, prox=None,
                 function_target=None):
        
        super().__init__()
        
        self.function_evals = 0
        self.function_budget = function_budget
        self.function_target = function_target
        self.f = f
        self.x = x0
        self.n = len(x0)
        self.t = 0
        self.delta = params["delta"]
        self.sparsity = params["sparsity"]
        self.step_size = params["step_size"]
        self.num_samples = params["num_samples"]
        self.prox = prox
        # Define sampling matrix
        # TODO (?): add support for other types of random sampling directions
        Z = 2*(np.random.rand(self.num_samples, self.n) > 0.5) - 1

        cosamp_params = {"Z": Z, "delta": self.delta, "maxiterations": 10,
                         "tol": 0.5, "sparsity": self.sparsity}
        self.cosamp_params = cosamp_params

    # Handle the (potential) proximal operator
    def Prox(self, x):
        if self.prox is None:
            return x
        else:
            return self.prox.prox(x, self.step_size)
       
    def CosampGradEstimate(self):
        '''
        Gradient estimation sub-routine.
        '''
      
        maxiterations = self.cosamp_params["maxiterations"]
        Z = self.cosamp_params["Z"]
        delta = self.cosamp_params["delta"]
        sparsity = self.cosamp_params["sparsity"]
        tol = self.cosamp_params["tol"]
        num_samples = np.size(Z, 0)
        x = self.x
        f = self.f
        y = np.zeros(num_samples)
        function_estimate = 0
        
        for i in range(num_samples):
            y_temp = f(x + delta*np.transpose(Z[i,:]))
            y_temp2 = f(x)
            function_estimate += y_temp2
            y[i] = (y_temp - y_temp2)/(np.sqrt(num_samples)*delta)
            self.function_evals += 2
        
        Z = Z/np.sqrt(num_samples)
        grad_estimate = cosamp(Z, y, sparsity, tol, maxiterations)
        function_estimate = function_estimate/num_samples
    
        return grad_estimate, function_estimate

    def step(self):
        '''
        Take step of optimizer
        '''
   
        grad_est, f_est = self.CosampGradEstimate()
        self.fd = f_est
        # Note that if no prox operator was specified then self.prox is the
        # identity mapping.
        self.x = self.Prox(self.x -self.step_size*grad_est)

        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return current iterate
            return self.function_evals, self.x, 'B'

        if self.function_target is not None:
            if self.reachedFunctionTarget(self.function_target, f_est):
                # if function target is reached terminate
                return self.function_evals, self.x, 'T'
 
        self.t += 1

        return self.function_evals, False, False