#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
python bin/evaluate/evaluate.py -p test
"""

import argh
import json
from os.path import join, exists
import glob
import os
import numpy as np
import tqdm

from src.callbacks import compute_svd

from collections import defaultdict

import h5py

import keras.backend as K
from keras.utils import np_utils

from src.utils.vegab import configure_logger
from bin.train_resnet_cifar import init_data_and_model as resnet_cifar_init_data_and_model
from bin.train_vgg_cifar import init_data_and_model as vgg_cifar_init_data_and_model
from bin.train_simple_cnn_cifar import init_data_and_model as simple_cnn_cifar_init_data_and_model
from bin.train_imdb import init_data_and_model as imdb_init_data_and_model
from src.lanczos import TopKEigenvaluesBatched

import logging

logger = logging.getLogger(__name__)


def init_data_and_model(path):
    # A bit of a mouthful, but serves the purposs
    C = json.load(open(join(path, "config.json")))
    if exists(join(path, "train_resnet_cifar.py")):
        return resnet_cifar_init_data_and_model(C)
    elif exists(join(path, "train_vgg_cifar.py")):
        return vgg_cifar_init_data_and_model(C)
    elif exists(join(path, "train_simple_cnn_cifar.py")):
        return simple_cnn_cifar_init_data_and_model(C)
    elif exists(join(path, "train_imdb.py")):
        return imdb_init_data_and_model(C)
    else:
        raise NotImplementedError()


def get_loss(model, X, y, N=1000):
    vals = model.evaluate(X[0:N], y[0:N], verbose=0)
    return vals[0]


def add_direction_to_weights(v, eps, mapping, model):
    idx = [0] + list(np.cumsum([np.prod(c) for c in mapping['shapes']]))
    setters = []
    for w in model.trainable_weights:
        idw = mapping['names'].index(w.name)
        setters.append((w, K.get_value(w) + eps * v[idx[idw]: idx[idw + 1]].reshape(mapping['shapes'][idw])))
    K.batch_set_value(setters)


def dist_weights(model, f_B):
    WB = h5py.File(f_B)
    if "model_weights" in WB:
        WB = WB['model_weights']

    logger.info("Adding dist travelled")
    dist2 = 0
    for p in model.trainable_weights:
        layer_name, weight_name = p.name.split("/")
        dist2 += np.sum((WB[layer_name][layer_name][weight_name] - K.get_value(p)) ** 2)

    return np.sqrt(dist2)


def construct_lanczos(model_inference, C, dataset, K, N=5120, id_sample=0):
    lanczos_aug = C.get("lanczos_aug", False)

    if not lanczos_aug:
        logger.info("Not using augmentation in Lanczos computation")
        if isinstance(dataset, list):
            ids = np.random.RandomState(id_sample).choice(len(dataset[0]), len(dataset[0]), replace=False)
            d = [dataset[0][ids][0:N], np_utils.to_categorical(dataset[1][ids])[:N]]
        else:
            L = N
            L_sampled = 0
            d = {"x_train": [], "y_train": []}
            while L_sampled < L:
                x, y = next(dataset)
                L_sampled += len(x)
                d['x_train'].append(x)
                d['y_train'].append(y)
            logger.info("Sampled {} wanted {}".format(L_sampled, N))
            d = [np.concatenate(d['x_train']), np.concatenate(d['y_train'])]
            logger.info("Sampled {}".format(len(d[0])))
    else:
        # This samples a constant batc
        logger.info("Using data augmentation in Lanczos computation")
        L_sampled = 0
        d = {"x_train": [], "y_train": []}
        # This first loop makes sure the data we evaluate on a different subsample
        # Note: id_sample + 1 to make sure this is independent from a sample used in training in a
        # special case when no resample was used
        while L_sampled < N * (id_sample + 1):
            logger.info(L_sampled)
            x, y = next(dataset)
            L_sampled += len(x)
            pass
        L_sampled = 0
        while L_sampled < N:
            logger.info(L_sampled)
            x, y = next(dataset)
            L_sampled += len(x)
            d['x_train'].append(x)
            d['y_train'].append(y)
        d['x_train'] = np.concatenate(d['x_train'])
        d['y_train'] = np.concatenate(d['y_train'])
        logger.info("Sampled {}".format(len(d['x_train'])))
        logger.info("Sampled {}".format(len(d['y_train'])))
        logger.info("Sampled {}".format(d['x_train'].shape))
        logger.info("Sampled {}".format(d['y_train'].shape))
        d = [d['x_train'], d['y_train']]

    top_eigenvalues_clbk = TopKEigenvaluesBatched(
        data=d,
        impl="scipy",
        save_path=".",
        n_classes=d[1].shape[1],
        K=K, batch_size=C['lanczos_top_K_bs'])

    top_eigenvalues_clbk.set_my_model(model_inference)
    top_eigenvalues_clbk._compile()

    return top_eigenvalues_clbk


def evaluate_eigenvalues(model_inference, C, dataset, K, N=5120, id_sample=0):
    top_eigenvalues_clbk = construct_lanczos(model_inference, C, dataset, K, N=N, id_sample=id_sample)
    E, Ev = top_eigenvalues_clbk._compute_top_K()
    return list([float(v) for v in E])


def evaluate_FN(model_inference, C, dataset, K, N=5120, id_sample=0, M=50):
    # M=50 seems to converge pretty well. M=20 is already pretty good if time is an issue
    top_eigenvalues_clbk = construct_lanczos(model_inference, C, dataset, K, N=N, id_sample=id_sample)
    fn = 0
    rng = np.random.RandomState(777)
    logger.info("Evaluating the FN")
    for _ in tqdm.tqdm(range(M), total=M):
        logger.info(_)
        fn += np.linalg.norm(top_eigenvalues_clbk._compute_Hv(rng.normal(0, 1, size=(top_eigenvalues_clbk.dim,))))
    return np.sqrt(fn / float(M))


def stream_create(X, y, rng=np.random.RandomState(777), batch_size=100):
    while True:
        ids = rng.choice(len(X), len(X), replace=False)
        assert len(set(ids)) == len(ids) == len(X)

        # Discards last batch of non-equal size
        batch = [[], []]

        for id in ids:
            batch[0].append(X[id])
            batch[1].append(y[id])

            if len(batch[0]) == batch_size:
                batch = [np.array(bb) for bb in batch]

                yield batch

                batch = [[], []]


def main(path="", checkpoint="best_val", L=5, N=2560, eval_hessian=0, M=1, config="{}"):
    """
    Params
    ------
    K: Number of eigenvecors, inint
    N: Numer of samples to use
    """
    eval_results = {}

    (train, valid, test, meta_data), (model, model_inference, meta) = init_data_and_model(path)

    n_classes = len(set(meta_data['y_train'].reshape(-1, )))
    print(n_classes)

    C = json.load(open(join(path, "config.json")))
    C_update = eval(config)
    logger.info("Updating config with " + str(C_update))
    C.update(C_update)

    model.compile(optimizer="sgd",
        loss='categorical_crossentropy',
        metrics=['accuracy', 'categorical_crossentropy'])

    model_inference.compile(optimizer="sgd",
        loss='categorical_crossentropy',
        metrics=['accuracy', 'categorical_crossentropy'])

    # TODO: model_inference loading weights might be faulty? Not sure.
    if checkpoint == "best_val":
        mwp = join(path, "model_best_val.h5")
    elif checkpoint == "last_epoch":
        mwp = join(path, "model_last_epoch.h5")
    else:
        raise NotImplementedError()
    logger.info("Loading weights from " + mwp)
    model_inference.load_weights(mwp)
    model.load_weights(mwp)

    if eval_hessian:
        Es = []
        for i in range(M):
            E = evaluate_eigenvalues(model_inference, C=C, dataset=train, K=L, id_sample=i)
            logger.info(E)
            Es.append(E)
        eval_results['{}_top_K_e'.format(checkpoint)] = Es
        Es = []
        for i in range(M):
            E = evaluate_FN(model_inference, C=C, dataset=train, K=L, id_sample=i, M=50)
            logger.info(E)
            Es.append(E)
        eval_results['{}_FN'.format(checkpoint)] = Es

    # Distance from init #
    eval_results['{}_dist_init'.format(checkpoint)] = dist_weights(model, os.path.join(path, "init_weights.h5"))

    # Acc / log loss #
    print(meta_data['y_test'].shape)
    y_test = meta_data['y_test']
    if y_test.shape[-1] != n_classes:
        y_test = np_utils.to_categorical(y_test, n_classes)
    y_valid = meta_data['y_valid']
    if y_valid.shape[-1] != n_classes:
        y_valid = np_utils.to_categorical(y_valid, n_classes)
    y_train = meta_data['y_train']
    if y_train.shape[-1] != n_classes:
        y_train = np_utils.to_categorical(y_train, n_classes)

    result = model.evaluate(meta_data['x_test'], y_test, batch_size=100)
    eval_results['{}_test_acc'.format(checkpoint)] = result[1]
    eval_results['{}_test_total_loss'.format(checkpoint)] = result[0]
    eval_results['{}_test_loss'.format(checkpoint)] = result[2]

    result = model.evaluate(meta_data['x_train'][0:len(y_test)], y_train[0:len(y_test)], batch_size=100)

    eval_results['{}_train_acc'.format(checkpoint)] = result[1]
    eval_results['{}_train_loss'.format(checkpoint)] = result[2]
    eval_results['{}_train_total_loss'.format(checkpoint)] = result[0]

    result = model.evaluate(meta_data['x_valid'][0:len(y_valid)], y_valid[0:len(y_test)], batch_size=100)
    eval_results['{}_val_acc'.format(checkpoint)] = result[1]
    eval_results['{}_val_loss'.format(checkpoint)] = result[2]
    eval_results['{}_val_total_loss'.format(checkpoint)] = result[0]

    logger.info(eval_results)

    json.dump(eval_results, open(join(path, "eval_results_{}.json".format(checkpoint)), "w"))


if __name__ == "__main__":
    configure_logger('', log_file=None)
    argh.dispatch_command(main)
