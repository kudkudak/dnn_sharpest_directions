# -*- coding: utf-8 -*-
"""
Configs used in the project for resnet experiments
"""

from src.utils.vegab import ConfigRegistry

imdb_configs = ConfigRegistry()

imdb_configs.set_root_config({
    ### Data ###
    "dataset": "imdb",
    "n_examples": -1,
    "seed": 777,
    "use_valid": True,
    "data_seed": 777,
    "random_Y_fraction": 0.0,
    ##############

    ### Model ###
    "model": "cnn",
    "load_weights_from": "",
    "load_opt_weights": False,
    "dropout": 0.2,
    "filters": 250,
    "hidden_dims": 250,
    ##############

    ### Optimization ###
    "lr": 0.01,
    "m": 0.0,
    "batch_size": 32,
    "loss": "categorical_crossentropy",
    "reload_before_drop": False,
    "n_epochs": 10,
    "opt_kwargs": "{}",
    "optim": "sgd",
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

    "eigen_loss": False,
    "save_eigendirections": -1,
    "measure_train_loss": False,
    "reload": False,
    "variable_schedule": "{}",
    "early_stopping": True,
    "early_stopping_patience": 200,
    "save_freq": 100,
    "lanczos_kwargs": "{'frequency':-1,'impl':'scipy'}",
    "lanczos_fullgrad_approx": False,
    "lanczos_top_K": -1,
    "lanczos_aug": False,
    "lanczos_inference_mode": True,
    "lanczos_top_K_N": 100000000,  # Data used to compute
    "lanczos_top_K_N_sample": 2250,  # Data used to compute
    "lanczos_top_K_bs": 128,  # Batch used to compute
    ##############
})

c = imdb_configs['root']
c['model'] = 'cnn'
imdb_configs['cnn'] = c

c = imdb_configs['cnn']
c['model'] = 'cnn'
c['filters'] = 500
c['hidden_dims'] = 500
imdb_configs['cnn_big'] = c

c = imdb_configs['cnn']
c['model'] = 'cnn'
c['filters'] = 1000
c['hidden_dims'] = 1000
imdb_configs['cnn_big2'] = c