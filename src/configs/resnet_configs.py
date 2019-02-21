# -*- coding: utf-8 -*-
"""
Configs used in the project for resnet experiments
"""

from src.utils.vegab import ConfigRegistry

resnet_configs = ConfigRegistry()

resnet_configs.set_root_config({
    ### Data ###
    "dataset": "cifar",
    "use_valid": True,
    "seed": 777,
    "data_seed": 777,
    "random_Y_fraction": 0.0,
    "which": "cifar",
    "preprocessing": "center",
    "n_examples": -1,
    "augmentation": False,
    ##############

    ### Model ###
    "model": "resnet",
    "l2": 0.0,
    "pool_size": 7,
    "k": 4,
    "n_stages": 1,
    "n": 2,
    "block_type": "identity", # identity or original
    "normalization": "bn",
    "dropout": 0.0,
    "activation_main_path": "id",
    "init_scale": 1.0,
    "load_weights_from": "",
    "load_opt_weights": True,
    "split_softmax": False, # Just for compability
    ##############


    ### Optimization ###
    "lr": 0.1,
    "m": 0.9,
    "reload_before_drop": False,
    "n_epochs": 150,
    "optim": "sgd",
    "batch_size": 128,
    "lr_schedule": "",
    "lr_schedule_type": "list_of_lists",
    "samples_per_epoch": -1,
    "epoch_size": -1,
    "opt_kwargs": "{}",
    ##############

    ### Callbacks configs ###
    "fbg_analysis": False,

    "decompose_analysis": False,
    "decompose_analysis_kw": "{}",

    "alex_random": False, # For unittest

    "reduce_callback": False,
    "reduce_callback_kwargs": "{}",
    "early_stopping": True,
    "early_stopping_patience": 200,
    "eigen_loss": False,
    "save_eigendirections": -1,
    "eigen_loss_N": 1280,
    "eigen_loss_training_mode": False,
    "eigen_loss_scaling": "by_grad_norm_lr",
    "eigen_loss_ids": "", # Which ids to calculate
    "track_dists_from_init": True,
    "measure_train_loss": False,
    "reload": False,
    "variable_schedule": "{}",
    "save_pred_freq": 0,
    "save_freq": 10,
    "lanczos_kwargs": "{'frequency':-1,'impl':'scipy'}",
    "lanczos_fullgrad_approx": False,
    "lanczos_top_K": -1,
    "lanczos_aug": True,
    "lanczos_inference_mode": True,
    "lanczos_top_K_N": 100000000,  # Data used to compute
    "lanczos_top_K_N_sample": 2250,  # Data used to compute
    "lanczos_top_K_bs": 128,  # Batch used to compute
    ##############

})

## MNIST ##

c = resnet_configs['root']
resnet_configs['mnist_root'] = c

## CIFAR ##

c = resnet_configs['root']
c['dataset'] = "cifar"
c['augmentation'] = True
c['n'] = 5
c['pool_size'] = 8
c['k'] = 1
c['n_stages'] = 3
c['shared_ids'] = "[[],[],[]]"
c['l2'] = 0.0001
c['n_epochs'] = 300
c['dropout'] = 0.0
c['lr_schedule'] = "[[80,0.1],[120,0.01],[160,0.001],[200,0.0001],[240,0.00001],[1000000000000,0.000001]]"
resnet_configs['cifar_root'] = c

c = resnet_configs['cifar_root']
c['which'] = "10"
c['n'] = 3
resnet_configs['cifar10_resnet20'] = c

c = resnet_configs['cifar_root']
c['which'] = "10"
c['n'] = 5
resnet_configs['cifar10_resnet32'] = c

c = resnet_configs['cifar10_resnet32']
c['normalization'] = 'none'
c['dropout'] = 0
c['init_scale'] = 1/32. # This is sloppy
resnet_configs['cifar10_resnet32_nobn_nodrop'] = c

c = resnet_configs['cifar10_resnet32_nobn_nodrop']
c['init_scale'] = 1/64.  # This is sloppy
resnet_configs['cifar10_resnet32_nobn_nodrop_2'] = c

c = resnet_configs['cifar10_resnet32']
c['normalization'] = 'none'
c['dropout'] = 0
c['init_scale'] = 1./(5*3) # Number of residual blocks as in https://arxiv.org/pdf/1709.02956.pdf
resnet_configs['cifar10_resnet32_nobn_nodrop_3'] = c

c = resnet_configs['cifar10_resnet32_nobn_nodrop_2']
c['which'] = '100'
resnet_configs['cifar100_resnet32_nobn_nodrop_2'] = c

c = resnet_configs['cifar_root']
c['which'] = "10"
c['n'] = 6
resnet_configs['cifar10_resnet38'] = c

c = resnet_configs['cifar_root']
c['which'] = "10"
c['n'] = 7
resnet_configs['cifar10_resnet44'] = c

c = resnet_configs['cifar_root']
c['which'] = "10"
c['n'] = 9
resnet_configs['cifar10_resnet56'] = c

c = resnet_configs['cifar10_resnet56']
c['lr_schedule_type'] = 'batch_mapping'
c['lr_schedule'] = 'super_convergence_resnet56()'
c['batch_size'] = 1024 # Needs more stability
c['block_type'] = 'original' # Don't ask..
c['dropout'] = 0.0 # ?
c['activation_main_path'] = 'relu'
resnet_configs['cifar10_resnet56_super_convergence'] = c


c = resnet_configs['cifar10_resnet56']
c['batch_size'] = 1024 # Needs more stability
c['block_type'] = 'original' # Don't ask..
c['dropout'] = 0.0
resnet_configs['cifar10_resnet56_super_convergence_baseline'] = c

c = resnet_configs['cifar10_resnet56']
c['normalization'] = 'none'
c['dropout'] = 0
c['init_scale'] = 1/56.
resnet_configs['cifar10_resnet56_nobn_nodrop'] = c

c = resnet_configs['cifar10_resnet56_nobn_nodrop']
c['init_scale'] = 1/(56*2.)
resnet_configs['cifar10_resnet56_nobn_nodrop_2'] = c


c = resnet_configs['cifar10_resnet56_nobn_nodrop'] # On chyba zabija..
c['init_scale'] = 1/(28.)
resnet_configs['cifar10_resnet56_nobn_nodrop_3'] = c

c = resnet_configs['cifar10_resnet56_nobn_nodrop_2']
c['which'] = '100'
resnet_configs['cifar100_resnet56_nobn_nodrop_2'] = c

c = resnet_configs['cifar_root']
c['which'] = "10"
c['n'] = 12
resnet_configs['cifar10_resnet74'] = c

c = resnet_configs['cifar_root']
c['which'] = "10"
c['n'] = 18
resnet_configs['cifar10_resnet110'] = c

c = resnet_configs['cifar_root']
c['which'] = "10"
c['n'] = 18
c['shared_ids'] = str([[i for i in range(c['n']) if i >= 5] for _ in range(3)])
resnet_configs['cifar10_resnet110_shared'] = c

c = resnet_configs['cifar_root']
c['which'] = "100"
c['n'] = 3
resnet_configs['cifar100_resnet20'] = c

c = resnet_configs['cifar_root']
c['which'] = "100"
c['n'] = 5
resnet_configs['cifar100_resnet32'] = c

c = resnet_configs['cifar_root']
c['which'] = "100"
c['n'] = 6
resnet_configs['cifar100_resnet38'] = c

c = resnet_configs['cifar_root']
c['which'] = "100"
c['n'] = 7
resnet_configs['cifar100_resnet44'] = c

c = resnet_configs['cifar_root']
c['which'] = "100"
c['n'] = 9
resnet_configs['cifar100_resnet56'] = c

c = resnet_configs['cifar_root']
c['which'] = "100"
c['n'] = 12
resnet_configs['cifar100_resnet74'] = c

c = resnet_configs['cifar_root']
c['which'] = "100"
c['n'] = 18
resnet_configs['cifar100_resnet110'] = c
