#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:27:29 2021

@author: danielmckenzie

TODO:
     - There is a slight mismatch between function value and number of queries 
"""
from optimizers import *
from benchmarkfunctions import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
mpl.style.use('seaborn')

# problem set up
n = 2000
decay_factor = 0.5
noiseamp = 0.0001
obj_func = CompressibleQuadric(n, decay_factor, noiseamp)

# Choose initialization
#x0    = np.random.randn(n)
#x0    = 100*x0/np.linalg.norm(x0)

x0    = 10*np.ones(n)

xx0   = copy.deepcopy(x0)

#sparsity = s
#sparsity = int(0.1*len(x0)) # This is a decent default, if no better estimate is known. 

init_sparsity = 2   # initial sparisty guess for adaptive sampling, i.e. AdaZORO

# Parameters for ZORO. Defaults are fine in most cases
params = {"step_size":1.0, "delta": 0.0001, "max_cosamp_iter": 10, 
          "cosamp_tol": 0.1,"sparsity": init_sparsity,
          "num_samples_constant": 1, "phi_cosamp": 0.3,"phi_lstsq": 0.1, 
          "compessible_constant": 1.1}

performance_log_ZORO = [[0, obj_func(x0)]]


# initialize optimizer object
opt  = AdaZORO(x0, obj_func, params, function_budget= int(2e5))

# the actual optimization routine
termination = False
while termination is False:
    # optimization step
    # solution_ZORO = False until a termination criterion is met, in which 
    # case solution_ZORO = the solution found.
    # termination = False until a termination criterion is met.
    # If ZORO terminates because function evaluation budget is met, 
    # termination = B
    # If ZORO terminated because the target accuracy is met,
    # termination= T.
    
    evals_ZORO, solution_ZORO, termination = opt.step()

    # save some useful values
    performance_log_ZORO.append( [evals_ZORO,np.mean(opt.fd)] )
    # print some useful values
    opt.report( 'Estimated f(x_k): %f  function evals: %d\n' %
        (np.mean(opt.fd), evals_ZORO) )
   
fig, ax = plt.subplots()

ax.plot(np.array(performance_log_ZORO)[:,0],
 np.log10(np.array(performance_log_ZORO)[:,1]), linewidth=1, label = "ZORO")
plt.xlabel('function evaluations')
plt.ylabel('$log($f(x)$)$')
leg = ax.legend()
plt.show()