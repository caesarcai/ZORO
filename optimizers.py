#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:46:35 2021

Code for ZORO, by Cai, McKenzie Yin and Zhang

"""

import numpy as np
from multiprocessing.dummy import Pool
from base import BaseOptimizer
from scipy.linalg import circulant
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
    def Prox(self,x):
        if self.prox == None:
            return x
        else:
            return self.prox.prox(x, self.step_size)
            
        
        
    def CosampGradEstimate(self):
    # Gradient estimation
        
        maxiterations = self.cosamp_params["maxiterations"]
        Z = self.cosamp_params["Z"]
        delta = self.cosamp_params["delta"]
        sparsity = self.cosamp_params["sparsity"]
        tol = self.cosamp_params["tol"]
        num_samples = np.size(Z,0)
        x = self.x
        f = self.f
        dim = len(x)
    
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
        # Take step of optimizer
        
        grad_est, f_est = self.CosampGradEstimate()
        self.fd = f_est
        # Note that if no prox operator was specified then self.prox is the
        # identity mapping.
        self.x = self.Prox(self.x -self.step_size*grad_est)
        
        
        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return parent
            return self.function_evals, self.x, 'B'

        if self.function_target != None:
            if self.reachedFunctionTarget(self.function_target, f_est):
                # if function target is reached terminate
                return self.function_evals, self.x, 'T'
            
        self.t += 1
       
        return self.function_evals, False, False
    
        

class ZOBCD(BaseOptimizer):
    ''' ZOBCD for black box optimization. A sparsity-aware, block coordinate 
    descent method. See "A Zeroth Order Block Coordinate Descent Algorithm for 
    Black-Box Optimization" by Cai, Lou, McKenzie and Yin.
    
    March 23rd 2021
    
    '''

    def __init__(self, y0, sigma, f, params, function_budget=10000,
                 function_target=None):
        
        super().__init__()
        
        self.function_evals = 0
        self.function_budget = function_budget
        self.function_target = function_target
        self.f = f
        self.x = y0
        self.n = len(y0)
        self.t = 0
        self.Type = params["Type"]
        self.sparsity = params["sparsity"]
        self.delta1 = params["delta1"]
        self.step_size = sigma
        self.shuffle = np.random.permutation(self.n)
        
        # block stuff
        oversampling_param = 1.05
        self.J = params["J"]
        self.block_size = int(np.ceil(self.n/self.J))
        self.sparsity = int(np.ceil(oversampling_param*self.sparsity/self.J))
        print(self.sparsity)
        self.samples_per_block = int(np.ceil(oversampling_param*self.sparsity*np.log(self.block_size)))
        
        # Define cosamp_params
        if self.Type == "BCD":
            Z = 2*(np.random.rand(self.samples_per_block,self.block_size) > 0.5) - 1
        elif self.Type == "BCCD":
            z1 = 2*(np.random.rand(1,self.block_size) > 0.5) - 1
            Z1 = circulant(z1)
            SSet = np.random.choice(self.block_size,self.samples_per_block, replace=False)
            Z = Z1[SSet,:]
        else:
            raise Exception("Need to choose a type, either BCD or BCCD")
        
        cosamp_params = {"Z": Z, "delta": self.delta1, "maxiterations": 10,
                         "tol": 0.5, "sparsity": self.sparsity, "block": []} 
        self.cosamp_params = cosamp_params
        
    def CosampGradEstimate(self):
        # Gradient estimation
        
        maxiterations = self.cosamp_params["maxiterations"]
        Z = self.cosamp_params["Z"]
        delta = self.cosamp_params["delta"]
        sparsity = self.cosamp_params["sparsity"]
        tol = self.cosamp_params["tol"]
        block = self.cosamp_params["block"]
        num_samples = np.size(Z,0)
        x = self.x
        f = self.f
        dim = len(x)

        Z_padded = np.zeros((num_samples,dim))
        Z_padded[:,block] = Z
    
        y = np.zeros(num_samples)
        print(num_samples)
        function_estimate = 0
        
        for i in range(num_samples):
            y_temp = f(x + delta*np.transpose(Z_padded[i,:]))
            y_temp2 = f(x)
            function_estimate += y_temp2
            y[i] = (y_temp - y_temp2)/(np.sqrt(num_samples)*delta)
            self.function_evals += 2
        
        Z = Z/np.sqrt(num_samples)
        block_grad_estimate = cosamp(Z, y, sparsity, tol, maxiterations)
        grad_estimate = np.zeros(dim)
        grad_estimate[block] = block_grad_estimate
        function_estimate = function_estimate/num_samples
        
        return grad_estimate, function_estimate
    
    def step(self):
        # Take step of optimizer
        
        if self.t % self.J == 0:
            self.shuffle = np.random.permutation(self.n)
            print('Reshuffled!')
        
        coord_index = np.random.randint(self.J)
        block = np.arange((coord_index-1)*self.block_size, min(coord_index*self.block_size,self.n))
        block = self.shuffle[block]
        self.cosamp_params["block"] = block
        grad_est, f_est = self.CosampGradEstimate()
        self.fd = f_est
        self.x += -self.step_size*grad_est
        
        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return parent
            return self.function_evals, self.x, 'B'

        if self.function_target != None:
            if self.reachedFunctionTarget(self.function_target, f_est):
                # if function target is reach return population expected value
                return self.function_evals, self.x, 'T'
            
        self.t += 1
       
        return self.function_evals, False, False
