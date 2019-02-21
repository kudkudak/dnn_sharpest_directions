# -*- coding: utf-8 -*-
"""
Configs used in the project for resnet experiments
"""

from src.utils.vegab import ConfigRegistry

vgg_configs = ConfigRegistry()

vgg_configs.set_root_config({
    ### Data ###
    "which": "10",
    "use_valid": True,
    "seed": 777,
    "data_seed": 777,
    "random_Y_fraction": 0.0,
    "augmentation": True,
    ##############


    ### Model ###
    "l2": 5e-4,
    "load_weights_from": "",
    "dropout_1": 0.0,
    "dropout_2": 0.5,
    "dim_clf": 512,
    "activ": "relu",
    "bn": True,
    "features": "vgg11",
    "load_opt_weights": True,
    "bn_use_beta": True,
    ##############

    ### Optimization ###
    "lr": 0.1,
    "sgdw_wd": 0.0,
    "m": 0.9,
    "opt_kwargs": "{}",
    "n_epochs": 300,
    "optim": "sgd",
    "reload_before_drop": False,
    "batch_size": 128,
    "lr_schedule": "divide_every_k_schedule(n_epochs=300, freq=25, lr0=0.1, mult=2)",
    "lr_schedule_type": "list_of_lists",
    "samples_per_epoch": -1,
    ##############

    ### Callbacks configs ###
    "fbg_analysis": False,

    "decompose_analysis": False,
    "decompose_analysis_kw": "{}",

    "early_stopping": True,
    "early_stopping_patience": 200,
    "eigen_loss": False,
    "eigen_loss_N": 1280,
    "reduce_callback": False,
    "reduce_callback_kwargs": "{}",
    "eigen_loss_ids": "",  # Which ids to calculate
    "save_freq": -1,
    "measure_train_loss": False,
    "reload": False,
    "epoch_size": -1,
    "save_eigendirections": -1,
    "variable_schedule": "{}",
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

c = vgg_configs['root']
vgg_configs['vgg11'] = c

c = vgg_configs['root']
c['droput_2'] = 0.0 # TODO: Not usre..
vgg_configs['vgg11_no_dropout'] = c

