#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains a simple CNN or a simple MLP (TODO) on IMDB

Run as:

python bin/train_imdb.py cnn test

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
from src.optimizers import NSGD
from src.training_loop import training_loop
from src.utils.vegab import wrap
from src.models.imdb import build_cnn
from src.configs.imdb_configs import imdb_configs as configs
from src import vegab_plugins
from src.data import get_imdb, to_stream


def init_data_and_model(config):
    # Load data
    train, valid, test, meta_data = get_imdb(use_valid=config['use_valid'], seed=config['data_seed'])
    n_classes = len(set(meta_data['y_train'].reshape(-1, )))
    logging.info(valid[1].shape)
    logging.info(test[1].shape)
    valid[1] = np_utils.to_categorical(valid[1], n_classes)
    test[1] = np_utils.to_categorical(test[1], n_classes)
    train, batch_size_np = to_stream(train[0], train[1], n_classes=n_classes, batch_size=config['batch_size'],
        seed=config['data_seed'])
    if config['model'] == 'cnn':
        model = build_cnn(dropout=config['dropout'], filters=config['filters'], hidden_dims=config['hidden_dims'])
        model_inference = build_cnn(training=False, dropout=config['dropout'],
            filters=config['filters'], hidden_dims=config['hidden_dims'])
    else:
        raise NotImplementedError()

    meta_data['batch_size_np'] = batch_size_np

    return (train, valid, test, meta_data), (model, model_inference, None)


def train(config, save_path):
    tf.set_random_seed(config['seed'])
    np.random.seed(config['seed'])

    (train, valid, test, meta_data), (model, model_inference, _) = init_data_and_model(config)
    n_classes = len(set(meta_data['y_train'].reshape(-1, )))

    model.summary()

    if config['optim'] == "sgd":
        optimizer = SGD(lr=config['lr'], momentum=config['m'])
    elif config['optim'] == "nsgd":
        opt_kwargs = eval(config['opt_kwargs'])
        optimizer = NSGD(lr=config['lr'], momentum=config['m'], **opt_kwargs)
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
