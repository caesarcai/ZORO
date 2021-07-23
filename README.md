# ZORO

A Python implementation of the ZORO algorithm, as introduced in *Zeroth-order regularized optimization (ZORO): Approximately sparse gradients and adaptive sampling* by Cai, McKenzie, Yin and Zhang. Preprint available <a href=https://arxiv.org/abs/2003.13001> here </a>.

ZORO, and its adaptive variant AdaZORO, are implemented as classes in `optimizers.py`. We use the `BaseOptimizer` class as in [this](https://github.com/NiMlr/High-Dim-ES-RL) Repo.

## Requirements
Python 3.5+. For proximal operators the `pyproximal` package is required.

## Examples
See `Test.py`, `Test_Prox.py` and `Test_Ada.py` for examples of using ZORO.

## Questions?
Feel free to contact us at mckenzie@math.ucla.edu or hqcai@math.ucla.edu.

## Recommended citation
If you find this code useful please cite the following work:

H.Q. Cai, D. Mckenzie, W. Yin, and Z. Zhang. Zeroth-Order Regularized Optimization (ZORO): Approximately Sparse Gradients and Adaptive Sampling. arXiv preprint arXiv: 2003.13001.

Bibtex:  
@article{cai2020zeroth,  
title={Zeroth-order regularized optimization (zoro): Approximately sparse gradients and adaptive sampling},  
author={Cai, HanQin and Mckenzie, Daniel and Yin, Wotao and Zhang, Zhenliang},  
journal={arXiv preprint arXiv:2003.13001},  
year={2020}  
}
