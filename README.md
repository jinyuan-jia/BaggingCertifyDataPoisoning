# BaggingCertifyDataPoisoning
This repository contains code for our AAAI 2021 paper "[Intrinsic Certified Robustness of Bagging against Data Poisoning Attacks](https://arxiv.org/pdf/2008.04495.pdf)".

Required python tool: keras 2.3.1 (tensorflow 1.14.0 backend), numpy, scipy, statsmodels, argparse

# Code usage: 

training_mnist_bagging.py is used to train models, where each of them is learnt on a subset of k training examples sampled from the training dataset uniformly at random with replacement. 

compute_certified_radius.py is used to compute the certified poisoning size (refer to our Theorem 1). 

You can directly run the command in the following file: ```run.py ``` 

# Citation 

If you use this code, please cite the following paper: 

```
@inproceedings{
jia2021intrinsic,
title={Intrinsic Certified Robustness of Bagging against Data Poisoning Attacks},
author={Jinyuan Jia and Xiaoyu Cao and Neil Zhenqiang Gong},
booktitle={AAAI},
year={2021}
}
```
