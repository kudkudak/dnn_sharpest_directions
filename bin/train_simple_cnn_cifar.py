#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains a simple CNN on cifar

Run as:

python bin/train_simple_cnn_cifar.py root test
python bin/train_simple_cnn_cifar.py medium test

To see TB:

tensorboard --logdir=.
"""
import h5py
import matplotlib
import logging
import numpy as np

logger = logging.getLogger(__name__)

matplotlib.use('Agg')
import tensorflow as tf

from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam

from src.callback_constructors import add_eigenloss_callback, \
    add_lanczos_callbacks, add_common_callbacks, config_alex
from src.callbacks import PickableReduceLROnPlateau
from src.callbacks_analysis import DecomposeStepAnalysis
from src.optimizers import NSGD
from src.training_loop import training_loop
from src.utils.vegab import wrap
from src.models.simple_cnn import build_simple_cnn
from src.configs.simple_cnn_configs import simple_cnn_configs as configs
from src import DATA_FORMAT, vegab_plugins
from bin.train_resnet_cifar import init_data


def init_data_and_model(config):
    # Load data
    train, valid, test, meta_data = init_data(config)
    w, h = meta_data['x_train'].shape[1:3]
    n_classes = len(set(meta_data['y_train'].reshape(-1, )))
    assert DATA_FORMAT == "channels_last"

    build_kwargs = {
        "input_shape": (w, h, meta_data['x_train'].shape[3]),
        "dropout": config['dropout'],
        "init": config.get("init", "glorot_uniform"),
        "l2": config.get("l2", 0.0),
        "bn": config.get("bn", False),
        "nb_classes": n_classes,
        "n_filters": config.get("n_filters", 32),
        "n_dense": config.get("n_dense", 128),
        "kernel_size": config.get("kernel_size", 3),
        "n1": config.get("n1", 1),
        "n2": config.get("n2", 1),
        "use_bias": config.get("use_bias", True),
    }

    # Load model
    model = build_simple_cnn(**build_kwargs)
    model_inference = build_simple_cnn(training=False, **build_kwargs)

    return (train, valid, test, meta_data), (model, model_inference, None)


def train(config, save_path):
    tf.set_random_seed(config['seed'])
    np.random.seed(config['seed'])

    (train, valid, test, meta_data), (model, model_inference, _) = init_data_and_model(config)
    n_classes = len(set(meta_data['y_train'].reshape(-1, )))

    model.summary()
    model.steerable_variables['bs'] = meta_data['batch_size_np']

    if config['optim'] == "sgd":
        optimizer = SGD(lr=config['lr'], momentum=config['m'])
    elif config['optim'] == "nsgd":
        opt_kwargs = eval(config['opt_kwargs'])
        optimizer = NSGD(lr=config['lr'], momentum=config['m'], **opt_kwargs)
        # That was a stupid bug..
    elif config['optim'] == "adam":
        opt_kwargs = eval(config['opt_kwargs'])
        optimizer = Adam(lr=config['lr'], **opt_kwargs)
    else:
        raise NotImplementedError()

    model.compile(optimizer=optimizer,
        loss=config['loss'],
        metrics=['accuracy', config['loss']])

    if model_inference is not None:
        model_inference.summary()
        model_inference.compile(optimizer=optimizer,
            loss=config['loss'],
            metrics=['accuracy', config['loss']])

    model.metrics_names.append("lr")
    model.metrics_tensors.append(optimizer.lr)

    # Config Lanczos
    if config['lanczos_top_K_N'] == -1:
        config['lanczos_top_K_N'] = len(meta_data['x_train'])

    if config['measure_train_loss']:
        train_eval = [meta_data['x_train'][0:10000],
            np_utils.to_categorical(meta_data['y_train'][0:10000], int(n_classes))]
    else:
        train_eval = None

    callbacks, lanczos_clbk = add_lanczos_callbacks(config=config, save_path=save_path,
        meta_data=meta_data, model=model, model_inference=model_inference, n_classes=int(n_classes))

    callbacks += add_eigenloss_callback(config=config, save_path=save_path,
        top_eigenvalues_clbk=lanczos_clbk, model_inference=model_inference,
        meta_data=meta_data, n_classes=int(n_classes))

    callbacks += add_common_callbacks(config=config, save_path=save_path,
        meta_data=meta_data, model=model, model_inference=model_inference, train=train_eval,
        n_classes=int(n_classes))

    if isinstance(optimizer, NSGD):
        callbacks += config_alex(config=config, optimizer=optimizer, top_eigenvalues_clbk=lanczos_clbk,
            model=model)

    if config['epoch_size'] != -1:
        epoch_size = config['epoch_size']
    else:
        epoch_size = len(meta_data['x_train'])
    steps_per_epoch = epoch_size / config['batch_size']

    if config['decompose_analysis']:
        kw = eval(config['decompose_analysis_kw'])
        X, y = meta_data['x_train'], np_utils.to_categorical(meta_data['y_train'], 10)
        callbacks.append(DecomposeStepAnalysis(X=X, y=y, batch_size=config['batch_size'],
            save_path=save_path, sharpest_clbk=lanczos_clbk, **kw))

    if config['reduce_callback']:
        kwargs = eval(config['reduce_callback_kwargs'])
        callbacks.append(PickableReduceLROnPlateau(**kwargs))

    if config['load_weights_from']:
        logger.info("Loading weights from " + config['load_weights_from'])
        logger.info("Loading weights")
        model.load_weights(config['load_weights_from'])
        logger.info("Loaded weights")

        ## Don't load opt
        if config['load_opt_weights']:
            with h5py.File(config['load_weights_from']) as f:
                if 'optimizer_weights' in f:
                    # build train function (to get weight updates)
                    model._make_train_function()  # Note: might need call to model
                    optimizer_weights_group = f['optimizer_weights']
                    optimizer_weight_names = [n.decode('utf8') for n in optimizer_weights_group.attrs['weight_names']]
                    optimizer_weight_values = [optimizer_weights_group[n] for n in optimizer_weight_names]
                    model.optimizer.set_weights(optimizer_weight_values)
                else:
                    logger.error("No optimizer weights in wieghts file!")
                    raise Exception()

    training_loop(model=model, train=train, steps_per_epoch=steps_per_epoch, save_freq=config['save_freq'],
        checkpoint_monitor="val_acc", epochs=config['n_epochs'], save_path=save_path,
        reload=config['reload'],
        valid=valid, custom_callbacks=callbacks, verbose=2)


if __name__ == "__main__":
    wrap(configs, train, vegab_plugins)
