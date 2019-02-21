#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains a resnet on Cifar

Run as:

python bin/train_resnet_cifar.py cifar10_resnet32 test

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

from src.callback_constructors import add_eigenloss_callback, add_random_labels_evaluation, \
    add_lanczos_callbacks, add_common_callbacks, config_alex
from src.callbacks import PickableReduceLROnPlateau
from src.callbacks_analysis import DecomposeStepAnalysis
from src.optimizers import NSGD
from src.training_loop import training_loop
from src.utils.vegab import wrap
from src.models.resnet import build_resnet
from src.configs.resnet_configs import resnet_configs as configs
from src.data import get_cifar, to_image_stream_cifar10, get_mnist, random_label_corrupt
from src import DATA_FORMAT, vegab_plugins


def init_data(config):
    if config.get('dataset', 'cifar') == "fmnist":
        return init_data_fmnist(config)
    elif config.get('dataset', 'cifar') == "cifar":
        return init_data_cifar(config)
    else:
        raise NotImplementedError("Not implemented dataset" + config['dataset'])


def init_data_fmnist(config):
    train, valid, test, meta_data = get_mnist(which=config['dataset'], preprocessing=config['preprocessing'],
        seed=config['data_seed'], use_valid=True)

    n_classes = len(set(meta_data['y_train']))
    assert n_classes in {10, 100}

    logging.info(valid[1].shape)
    logging.info(test[1].shape)

    valid[1] = np_utils.to_categorical(valid[1], n_classes)
    test[1] = np_utils.to_categorical(test[1], n_classes)

    if "n_examples" in config and config['n_examples'] > 0:
        assert len(train[0]) >= config['n_examples']
        meta_data['x_train'] = meta_data['x_train'][0:config['n_examples']]
        meta_data['y_train'] = meta_data['y_train'][0:config['n_examples']]

    # Should work
    train, batch_size_np = to_image_stream_cifar10(meta_data['x_train'], meta_data['y_train'], n_classes=n_classes,
        batch_size=config['batch_size'], augmented=config['augmentation'], seed=config['data_seed'])
    train_2, _ = to_image_stream_cifar10(meta_data['x_train'], meta_data['y_train'], n_classes=n_classes,
        batch_size=128, augmented=config['augmentation'], seed=config['data_seed'])  # config['batch_size']

    meta_data['batch_size_np'] = batch_size_np
    meta_data['train_2'] = train_2

    return train, valid, test, meta_data


def init_data_cifar(config):
    train, valid, test, meta_data = get_cifar(which=config['which'], preprocessing="center",
        seed=config['data_seed'], use_valid=True)

    n_classes = len(set(meta_data['y_train'].reshape(-1, )))
    assert n_classes in {10, 100}

    logging.info(valid[1].shape)
    logging.info(test[1].shape)

    valid[1] = np_utils.to_categorical(valid[1], n_classes)
    test[1] = np_utils.to_categorical(test[1], n_classes)

    if "n_examples" in config and config['n_examples'] > 0:
        assert len(train[0]) >= config['n_examples']
        train = [train[0][0:config['n_examples']], train[1][0:config['n_examples']]]
        meta_data['x_train'] = meta_data['x_train'][0:config['n_examples']]
        meta_data['y_train'] = meta_data['y_train'][0:config['n_examples']]

    if config.get('random_Y_fraction', 0) != 0.:
        logger.info("Adding random labels {}".format(config['random_Y_fraction']))
        X_train, y_train, ids_random_train = random_label_corrupt(train[0], train[1], n_classes,
            random_Y_fraction=config['random_Y_fraction'],
            seed=config['data_seed'])
        train = [X_train, y_train]
        meta_data['x_train'] = X_train
        meta_data['y_train'] = y_train
        meta_data['ids_random_train'] = ids_random_train

    train_1, batch_size_np = to_image_stream_cifar10(train[0], train[1], n_classes=n_classes,
        batch_size=config['batch_size'], augmented=config['augmentation'], seed=config['data_seed'])

    train_2, _ = to_image_stream_cifar10(train[0], train[1], n_classes=n_classes,
        batch_size=128, augmented=config['augmentation'], seed=config['data_seed'])  # config['batch_size']

    meta_data['batch_size_np'] = batch_size_np
    meta_data['train_2'] = train_2

    return train_1, valid, test, meta_data


def init_data_and_model(config):
    # Load data
    train, valid, test, meta_data = init_data(config)
    w, h = meta_data['x_train'].shape[1:3]
    n_channels = meta_data['x_train'].shape[3]
    logger.info((w, h, n_channels))
    n_classes = len(set(meta_data['y_train'].reshape(-1, )))
    assert n_classes in {10, 100}
    assert DATA_FORMAT == "channels_last"

    build_kwargs = {
        "pool_size": config['pool_size'],
        "init_scale": config['init_scale'],
        "n": config['n'],
        "l2": config['l2'],
        "nb_classes": n_classes,
        "normalization": config['normalization'],
        "k": config['k'],
        "resnet_dropout": config['dropout'],
        "n_stages": config['n_stages'],
        "seed": config['seed'],
        "input_dim": (w, h, n_channels)}

    # Load model
    model, meta = build_resnet(**build_kwargs)
    model_inference, _ = build_resnet(training=False, **build_kwargs)

    return (train, valid, test, meta_data), (model, model_inference, meta)


def train(config, save_path):
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    tf.set_random_seed(config['seed'])
    np.random.seed(config['seed'])

    (train, valid, test, meta_data), (model, model_inference, meta_model) = init_data_and_model(config)
    n_classes = len(set(meta_data['y_train'].reshape(-1, )))

    model.summary()
    model.steerable_variables['bs'] = meta_data['batch_size_np']

    if config['optim'] == "sgd":
        optimizer = SGD(lr=config['lr'], momentum=config['m'])
    elif config['optim'] == "nsgd":
        opt_kwargs = eval(config['opt_kwargs'])
        optimizer = NSGD(lr=config['lr'], momentum=config['m'], **opt_kwargs)
    elif config['optim'] == 'rmsprop':
        optimizer = RMSprop(lr=config['lr'])
    elif config['optim'] == 'adam':
        optimizer = Adam(lr=config['lr'])
    elif config['optim'] == "nsgd":
        opt_kwargs = eval(config['opt_kwargs'])
        optimizer = NSGD(lr=config['lr'], momentum=config['m'], **opt_kwargs)
    else:
        raise NotImplementedError()

    model.compile(optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'categorical_crossentropy'])

    if model_inference is not None:
        model_inference.summary()
        model_inference.compile(optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'categorical_crossentropy'])

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

    callbacks += add_eigenloss_callback(config=config, save_path=save_path, top_eigenvalues_clbk=lanczos_clbk,
        meta_data=meta_data, n_classes=int(n_classes), model_inference=model_inference)

    callbacks += add_common_callbacks(config=config, save_path=save_path,
        meta_data=meta_data, model=model, model_inference=model_inference, train=train_eval,
        n_classes=int(n_classes))

    if isinstance(optimizer, NSGD):
        callbacks += config_alex(config=config, optimizer=optimizer, top_eigenvalues_clbk=lanczos_clbk,
            model=model)

    if config['decompose_analysis']:
        kw = eval(config['decompose_analysis_kw'])
        X, y = meta_data['x_train'], np_utils.to_categorical(meta_data['y_train'], 10)
        callbacks.append(DecomposeStepAnalysis(X=X, y=y, batch_size=config['batch_size'],
            save_path=save_path, sharpest_clbk=lanczos_clbk, **kw))

    if config['epoch_size'] != -1:
        epoch_size = config['epoch_size']
    else:
        epoch_size = len(meta_data['x_train'])
    steps_per_epoch = epoch_size / config['batch_size']

    if config['reduce_callback']:
        kwargs = eval(config['reduce_callback_kwargs'])
        callbacks.append(PickableReduceLROnPlateau(**kwargs))

    if config.get("random_Y_fraction", 0.0) > 0:
        logger.info("Addding random label evaluation")
        x_train = meta_data['x_train']
        y_train = np_utils.to_categorical(meta_data['y_train'], int(n_classes))
        callbacks += add_random_labels_evaluation(config=config, model=model, x_train=x_train,
            y_train=y_train, ids_random_train=meta_data['ids_random_train'])

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
