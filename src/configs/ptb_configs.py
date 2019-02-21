# -*- coding: utf-8 -*-
"""
Configs used for PTB experiments. Should be similar to

https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
https://gist.github.com/p-baleine/e6f57dd6f89c932eccadec5d78fed0b5
https://openreview.net/pdf?id=ByJHuTgA-
"""

from src.utils.vegab import ConfigRegistry

ptb_configs = ConfigRegistry()

ptb_configs.set_root_config({
    "seed": 777,

    # Lanczos
    "lanczos_kwargs": "{'frequency':-1,'impl':'scipy'}",
    "lanczos_top_K": -1,
    "lanczos_aug": True,
    "lanczos_inference_mode": True,
    "lanczos_fullgrad_approx": False,
    "lanczos_top_K_N": 100000000,  # Data used to compute
    "lanczos_top_K_N_sample": 2250,  # Data used to compute, ~5%
    "lanczos_top_K_bs": 20,  # Batch used to compute.
    "save_eigendirections": False,

    # Optimization
    "epoch_size": -1, # (In num_steps units, because one example is num_steps words)
    "load_weights_from": "",
    "load_opt_weights": True,
    "save_freq": 10,
    "n_epochs": 15,
    "m": 0.0,
    "lr_decay": 0.5,
    "batch_size": 20,
    "opt_kwargs": "{'clipnorm': 5.0}", # Warning : Might be subotpimal due to a different init!
    "max_grad_norm": 5.0,

    "lr_schedule": "[[5, 5.0]] + [[5+i,5.0/(2**i)] for i in range(1,1000)]",
    "opt": "sgd",

    "reload": False,

    # Model
    "lr": 5.0, # This is different!
    "num_layers": 2,
    "num_steps": 20,
    "hidden_size": 200,
    "max_epoch": 4,
    "keep_prob":  1.0, # This is dropout
    "vocab_size": 10000,

    # Callbacks
    "early_stopping": False,
    "early_stopping_patience": 200,
    "measure_train_loss": False,
    "decompose_analysis": False,
    "decompose_analysis_kw": "{}",
})

# Should get approx 121/60 (valid/test)
c = ptb_configs['root']
ptb_configs['small'] = c

# Warning: perplexity is not comparable between small and tiny
c = ptb_configs['root']
c['vocab_size'] = 1000
ptb_configs['tiny'] = c