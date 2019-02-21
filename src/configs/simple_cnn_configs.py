# -*- coding: utf-8 -*-
"""
Configs used in the project for resnet experiments
"""

from src.utils.vegab import ConfigRegistry

simple_cnn_configs = ConfigRegistry()

simple_cnn_configs.set_root_config({
    ### Data ###
    "dataset": "cifar",
    "n_examples": -1,
    "use_valid": True,
    "seed": 777,
    "data_seed": 777,
    "augmentation": True,
    "preprocessing": "center",
    "random_Y_fraction": 0.0,
    "which": "10",
    ##############

    ### Model ###
    "load_weights_from": "",
    "load_opt_weights": False,
    "dropout": 0.0,
    "l2": 0.,
    "bn": False,
    "n_filters": 32,
    "use_bias": True,
    "n_dense": 128,
    "kernel_size": 3,
    "n1": 1,
    "activation":"relu",
    "n2": 1,
    "init": "glorot_uniform",
    ##############

    ### Optimization ###
    "lr": 0.01,
    "m": 0.9,
    "loss": "categorical_crossentropy",
    "reload_before_drop": False,
    "n_epochs": 150,
    "opt_kwargs": "{}",
    "optim": "sgd",
    "batch_size": 128,
    "lr_schedule": "",
    "lr_schedule_type": "list_of_lists",
    "samples_per_epoch": -1,
    "epoch_size": -1,
    "reduce_callback": False,
    "reduce_callback_kwargs": "{'monitor': 'val_acc'}",
    ##############

    ### Callbacks configs ###,
    "fbg_analysis": False,

    "decompose_analysis": False,
    "decompose_analysis_kw": "{}",

    "alex_adapt_lr": "",
    "eigen_loss": False,
    "save_eigendirections": -1,
    "eigen_loss_training_mode": False,
    "eigen_loss_N": 1280,
    "eigen_loss_ids": "", # Which ids to calculate
    "measure_train_loss": False,
    "reload": False,
    "variable_schedule": "{}",
    "early_stopping": True,
    "early_stopping_patience": 200,
    "save_freq": 100,
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

c = simple_cnn_configs['root']
c['kernel_size'] = 5
c['n_filters'] = 100
simple_cnn_configs['medium'] = c

c = simple_cnn_configs['medium']
c['dataset'] = 'fmnist'
c['augmentation'] = True
simple_cnn_configs['medium_fmnist'] = c


c = simple_cnn_configs['root']
c['which'] = '100'
simple_cnn_configs['root_100'] = c