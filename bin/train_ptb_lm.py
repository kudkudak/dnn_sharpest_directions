#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Simple script to train LSTM on PTB

Run as:

python bin/train_ptb_lm.py small tst

Achieves 61 train, 124 valid perplexity after 20 epochs, roughly matching
TF small config (train perplexity is 2x higher, but valid matches). Takes around 30m on TitanX.

To see TB:

tensorboard --logdir=.

Should roughly match log in :
https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
Stolen from:
https://gist.github.com/p-baleine/e6f57dd6f89c932eccadec5d78fed0b5
"""
import os
import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)
import h5py
import tensorflow as tf

import keras.backend as K
from keras.optimizers import Optimizer, interfaces, SGD

from src.callbacks import LambdaCallbackPickable
from src.callback_constructors import add_lanczos_callbacks, add_common_callbacks, config_alex
from src.configs.ptb_configs import ptb_configs as configs
from src.data_ptb import ptb_iterator, ptb_raw_data
from src.models.ptb import ptb_lstm
from src.training_loop import training_loop
from src.utils.vegab import wrap
from src import DATA_FORMAT, vegab_plugins
# TODO: Incude PTBSGD necessary features into NSGD?
from src.optimizers import PtbSGD, NSGD


def init_data_and_model(config):
    # Init model
    model = ptb_lstm(config)
    model_inference = ptb_lstm(config, training=False)

    # Init data
    raw_data = ptb_raw_data(vocab_size=config['vocab_size'])
    word_to_id, id_to_word, train_data, valid_data, test_data = raw_data
    meta_data = {"word_to_id": word_to_id, "id_to_word": id_to_word, "train_data": train_data,
        "valid_data": valid_data, "test_data": test_data}

    meta_data['train'] = train_data
    meta_data['valid'] = valid_data
    meta_data['test'] = test_data

    def train():
        while True:
            it = ptb_iterator(train_data, config['batch_size'], config['num_steps'], config['vocab_size'])
            for xy in it:
                yield xy

    def train_2():
        while True:
            # Small trick - doesn't change cls
            it = ptb_iterator(train_data, config['batch_size'], config['num_steps'], None)
            for xy in it:
                yield xy

    def valid():
        while True:
            it = ptb_iterator(valid_data, config['batch_size'], config['num_steps'], config['vocab_size'])
            for xy in it:
                yield xy

    def test():
        while True:
            it = ptb_iterator(test_data, config['batch_size'], config['num_steps'], config['vocab_size'])
            for xy in it:
                yield xy

    train = train()
    valid = valid()
    test = test()

    # Stupid meta_data
    meta_data['batch_size_np'] = np.array([config['batch_size']])
    meta_data['num_steps_valid'] = sum([1 for _ in
        ptb_iterator(valid_data, config['batch_size'], config['num_steps'], config['vocab_size'])])

    x_train, y_train = [], []
    for step, (x, y) in enumerate(ptb_iterator(train_data, config['batch_size'],
            config['num_steps'], vocab_size=None)):

        if step == 0:
            logger.info(x.shape)
            logger.info(y.shape)

        x_train.append(x)
        y_train.append(y)
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    meta_data['x_train'] = x_train
    meta_data['y_train'] = y_train

    meta_data['train_2'] = train_2()

    # (n_examples, n_steps)
    logger.info("len(x_train)=" + str((x_train.shape)))
    logger.info("len(y_train)=" + str((y_train.shape)))

    return (train, valid, test, meta_data), (model, model_inference)


def perplexity(y_true, y_pred):
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    assert K.ndim(cross_entropy) == 2
    return K.exp(K.mean(cross_entropy))


def train(config, save_path):
    tf.set_random_seed(config['seed'])
    np.random.seed(config['seed'])

    (train, valid, test, meta_data), (model, model_inference) = init_data_and_model(config)

    logger.info("len(train_data)=" + str(len(meta_data['train_data'])))

    vocab_size = config['vocab_size']
    if config['epoch_size'] == -1:
        steps_per_epoch = ((len(meta_data['train_data']) // config['batch_size']) - 1) // config['num_steps']
    else:
        steps_per_epoch = config['epoch_size'] / config['batch_size']
        # steps_per_epoch = (((config['num_steps'] * config['epoch_size']) // config['batch_size']) - 1) // config[
        #     'num_steps']

    logger.info("steps_per_epoch=" + str(steps_per_epoch))

    model.summary()

    logger.info("Running!")
    logger.info(next(meta_data['train_2'])[0].shape)
    logger.info(next(meta_data['train_2'])[1].shape)

    if config['opt'] == 'ptbsgd':
        optimizer = PtbSGD(lr=config['lr'], decay=config['lr_decay'],
            clipnorm=config['max_grad_norm'],
            epoch_size=steps_per_epoch,
            max_epoch=config['max_epoch'])
    elif config['opt'] == 'sgd':
        opt_kw = eval(config['opt_kwargs'])
        optimizer = SGD(momentum=config['m'], lr=config['lr'], **opt_kw)
    elif config['opt'] == "nsgd":
        opt_kwargs = eval(config['opt_kwargs'])
        optimizer = NSGD(lr=config['lr'], momentum=config['m'], **opt_kwargs)
    else:
        raise NotImplementedError()

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy", perplexity])
    model_inference.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy", perplexity])

    model.metrics_names.append("lr")
    model.metrics_tensors.append(optimizer.lr)

    with open(os.path.join(save_path, 'vocab.bin'), 'wb') as f:
        pickle.dump(meta_data['word_to_id'], f)

    logger.info('Training with {} size'.format(config['hidden_size']))

    # IDEA: Change to cls inside lanczos if neeed, simple
    callbacks, lanczos_clbk = add_lanczos_callbacks(config=config, save_path=save_path,
        meta_data=meta_data, model=model, model_inference=model_inference, n_classes=config['vocab_size'])

    if isinstance(optimizer, NSGD):
        callbacks += config_alex(config=config, optimizer=optimizer, top_eigenvalues_clbk=lanczos_clbk,
            model=model)

    callbacks += add_common_callbacks(config=config, save_path=save_path,
        meta_data=meta_data, model=model, model_inference=model_inference, train=None,
        n_classes=vocab_size)

    # We use stateful LSTM and model PTB as seq-2-seq
    def reset_model(epoch, logs):
        model.reset_states()

    callbacks += [LambdaCallbackPickable(on_epoch_end=reset_model)]

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

    training_loop(model=model, train=train, steps_per_epoch=steps_per_epoch,
        save_freq=config['save_freq'],
        checkpoint_monitor="val_acc", epochs=config['n_epochs'], save_path=save_path,
        reload=config['reload'], validation_steps=meta_data['num_steps_valid'],
        valid=valid, custom_callbacks=callbacks, verbose=2)


if __name__ == "__main__":
    wrap(configs, train, vegab_plugins)
