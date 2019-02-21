# DNN's Sharpest Directions

Repository for [On the Relation Between the Sharpest Directions of DNN Loss and the SGD Step Length](https://arxiv.org/abs/1807.05031).

Disclaimer: this repository contains code for running all  experiments in the paper. Code in `experiments` folder is
not release quality, but is provided for maximum reproducibility.

## Requirements

* keras==2.2.5
* tensorflow-gpu==1.8.0

Rest of the requirements can be found in `requirements.txt`.

## Example commands

Few exemplary commands for running experiments are given here

### SimpleCNN - measure Hessian

To measure the largest eigenvalues of the Hessian along the trajectory run: 

``python bin/train_simple_cnn_cifar.py medium example --lanczos_top_K=5 --lanczos_top_K_N=2250``

This will run training of SimpleCNN (medium size) while evaluating top $K=5$ eigenvalues (using $2250$ random samples
from the training set). All outputs are saved to folder `example`.

You can visualize results by using Tensorboard, or by examining history.csv saved in the results dirs. 

### SimpleCNN + NSGD

To reproduce an exemplary run of NSGD on CIFAR10 and SimpleCNN run the following:

``python bin/train_simple_cnn_cifar.py medium baseline --n_epochs=1000 --opt_kwargs=\"{'overshoot':1.0,'KK':5}\" --optim=nsgd  --lr=0.01 --m=0 --seed=800 --data_seed=800  --reduce_callback --reduce_callback_kwargs=\"{'monitor': 'val_loss', 'patience':200}\" --early_stopping --early_stopping_patience=500  --lanczos_top_K=5 --lanczos_top_K_N=2250``

``python bin/train_simple_cnn_cifar.py medium gamma=01 --n_epochs=1000 --opt_kwargs=\"{'overshoot':0.1,'KK':5}\" --optim=nsgd  --lr=0.01 --m=0 --seed=800 --data_seed=800  --reduce_callback --reduce_callback_kwargs=\"{'monitor': 'val_loss', 'patience':200}\" --early_stopping --early_stopping_patience=500  --lanczos_top_K=5 --lanczos_top_K_N=2250``

There is a lot of options, but they are simply configuring:

* NSGD (``--opt_kwargs=\"{'overshoot':1.0,'KK':5}\" --optim=nsgd  --lr=0.01 --m=0``)
* Early stopping (``--early_stopping --early_stopping_patience=500``)
* Automatic LR schedule (``--reduce_callback --reduce_callback_kwargs=\"{'monitor': 'val_loss', 'patience':200}\"``)
* Lanczos computation (``--lanczos_top_K=5 --lanczos_top_K_N=2250``)

### Resnet32 + NSGD

To reproduce an exemplary run of NSGD on CIFAR10 and Resnet32 run the following (warning: takes few hours):

``python bin/train_resnet_cifar.py cifar10_resnet32_nobn_nodrop baseline --n_epochs=1000 --opt_kwargs=\"{'overshoot':1.0,'KK':5}\" --optim=nsgd  --lr=0.01 --m=0 --seed=800 --data_seed=800  --reduce_callback --reduce_callback_kwargs=\"{'monitor': 'val_loss', 'patience':200}\" --early_stopping --early_stopping_patience=500  --lanczos_top_K=5 --lanczos_top_K_N=2250``

``python bin/train_resnet_cifar.py cifar10_resnet32_nobn_nodrop gamma=001 --n_epochs=1000 --opt_kwargs=\"{'overshoot':0.01,'KK':5}\" --optim=nsgd  --lr=0.01 --m=0 --seed=800 --data_seed=800  --reduce_callback --reduce_callback_kwargs=\"{'monitor': 'val_loss', 'patience':200}\" --early_stopping --early_stopping_patience=500  --lanczos_top_K=5 --lanczos_top_K_N=2250``

Note that Resnet32 uses L2 regularization.

### FashionMNIST

To run experiments on FashionMNIST pass ``--dataset=fmnist``, e.g.:

``slurm_dnnsharpestp "python bin/train_simple_cnn_cifar.py medium_fmnist baseline_fmnist --augmentation --n_epochs=1000 --opt_kwargs=\"{'overshoot':1,'KK':5}\" --optim=nsgd  --lr=0.01 --m=0 --lr_schedule=\"\" --seed=800 --data_seed=800  --reduce_callback --reduce_callback_kwargs=\"{'monitor': 'val_loss', 'patience':200}\" --early_stopping --early_stopping_patience=500  --lanczos_top_K=5 --lanczos_top_K_N=2250"``

Note: this includes the same data augmentation as for CIFAR10.
 
## Experiments

To reproduce experiments in the paper see `experiments` folder. Each folder includes bash scripts and plotting code.

## Warning

Due to issues with tensorflow, computing Hvp over a stochastic graph (e.g. using dropout) can be very slow. In particular, 
in the Appendix experiments we used Batch Normalization in inference mode during Hessian computation.

