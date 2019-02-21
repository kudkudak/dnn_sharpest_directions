# -*- coding: utf-8 -*-
"""
Simple data getters. Each returns iterator for train and dataset for test/valid
"""
import logging
import os

import numpy as np
from keras.datasets import cifar10, cifar100, mnist, fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from keras.preprocessing import sequence
from keras.datasets import imdb

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
FMNIST_DIR = os.path.join(ROOT_DIR, "data/fmnist")


def get_imdb(max_features=5000, maxlen=400, use_valid=True, seed=777):
    rng = np.random.RandomState(seed)

    logger.info('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    logger.info((len(x_train), 'train sequences'))
    logger.info((len(x_test), 'test sequences'))

    logger.info(('Pad sequences (samples x time)'))
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    logger.info(('x_train shape:', x_train.shape))
    logger.info(('x_test shape:', x_test.shape))


    if use_valid:
        # Some randomization to make sure
        N = int(0.9*len(x_train))
        ids = rng.choice(len(x_train), len(x_train), replace=False)
        assert len(set(ids)) == len(ids) == len(x_train)
        x_train = x_train[ids]
        y_train = y_train[ids]
        (x_train, y_train), (x_valid, y_valid) = (x_train[0:N], y_train[0:N]), \
            (x_train[N:], y_train[N:])
        valid = [x_valid, y_valid]

    train = [x_train, y_train]
    test = [x_test, y_test]

    if use_valid:
        return train, valid, test, {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
            "x_valid": x_valid, "y_valid": y_valid}
    else:
        # Using as valid test
        return train, test, test, {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

def _batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def random_label_corrupt(X, y, n_classes, random_Y_fraction=0.0, seed=777):
    """
    Params
    ------
    arrays: list of tuples
        List of (X, y) pairs to which add random labels
    """
    rng = np.random.RandomState(seed)
    n_random = int(random_Y_fraction * len(X))
    ids_random = rng.choice(len(X), n_random, replace=False)
    logger.info(y.shape)
    y[ids_random] = rng.randint(0, n_classes, size=(y[ids_random].shape[0],)).reshape(-1, 1)
    return X, y, ids_random


def to_stream(X, y, batch_size=128, n_classes=10, seed=777):
    """

    Uses typical augmentation for CIFAR10/CIFAR100 if augmented=True

    Returns
    -------
    stream: iterator
        iterator over data
    batch_size_np: np.array
        object holding used batch size. Can be used to dynamically adapt batch size

    Notes
    -----
    Augmentation works well only for CIFAR10/CIFAR100
    """
    rng = np.random.RandomState(seed)

    X = X.copy()
    y = y.copy()
    assert set(y.reshape(-1, )) == set(range(n_classes)), "Please transform y to accepted format"
    y = np_utils.to_categorical(y, n_classes)

    batch_size_np = np.array([batch_size])

    def stream_create():
        while True:
            ids = rng.choice(len(X), len(X), replace=False)
            assert len(set(ids)) == len(ids) == len(X)

            # Discards last batch of non-equal size
            batch = [[], []]

            for id in ids:
                batch[0].append(X[id])
                batch[1].append(y[id])

                if len(batch[0]) == batch_size_np[0]:
                    batch = [np.array(bb) for bb in batch]

                    yield batch

                    batch = [[], []]

    stream = stream_create()

    return stream, batch_size_np


def to_image_stream_cifar10(X, y, batch_size=128, n_classes=10, augmented=True, seed=777, dim_ordering="channels_last"):
    """

    Uses typical augmentation for CIFAR10/CIFAR100 if augmented=True

    Returns
    -------
    stream: iterator
        iterator over data
    batch_size_np: np.array
        object holding used batch size. Can be used to dynamically adapt batch size

    Notes
    -----
    Augmentation works well only for CIFAR10/CIFAR100
    """
    rng = np.random.RandomState(seed)

    X = X.copy()
    y = y.copy()
    assert X.ndim == 4, "Please tranform x to supported image format"
    assert set(y.reshape(-1, )) == set(range(n_classes)), "Please transform y to accepted format"
    y = np_utils.to_categorical(y, n_classes)

    batch_size_np = np.array([batch_size])

    # Prepare train
    if augmented:
        # Uses default CIFAR10 augmentation
        datagen_train = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            data_format=dim_ordering,
            width_shift_range=0.125,  # 4 px
            height_shift_range=0.125,  # 4 px
            horizontal_flip=True,
            vertical_flip=False)

        datagen_train.fit(X)

        # Hack, concatenate y with additional column that will be used to pass

        train_sub = datagen_train.flow(X, y, batch_size=1, shuffle=True, seed=seed)

        def stream_create():
            batch = [[], []]
            while True:
                x, y = next(train_sub)
                batch[0].append(x)
                batch[1].append(y)

                if len(batch[0]) == batch_size_np[0]:
                    batch = [np.concatenate(bb, axis=0) for bb in batch]
                    assert len(batch[0]) == batch_size_np[0]

                    yield batch

                    batch = [[], []]

        stream = stream_create()
    else:
        def stream_create():
            while True:
                ids = rng.choice(len(X), len(X), replace=False)
                assert len(set(ids)) == len(ids) == len(X)

                # Discards last batch of non-equal size
                batch = [[], []]

                for id in ids:
                    batch[0].append(X[id])
                    batch[1].append(y[id])

                    if len(batch[0]) == batch_size_np[0]:
                        batch = [np.array(bb) for bb in batch]

                        yield batch

                        batch = [[], []]

        stream = stream_create()

    return stream, batch_size_np


def get_cifar(which="10", preprocessing="center", seed=777, use_valid=True):
    rng = np.random.RandomState(seed)

    if which == '10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif which == '100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        raise NotImplementedError(which)

    # Always outputs channels last
    if x_train.shape[1] == 3:
        logging.info("Transposing")
        x_train = x_train.transpose((0, 2, 3, 1))
        x_test = x_test.transpose((0, 2, 3, 1))
    assert x_train.shape[3] == 3 or x_train.shape[3] == 1

    if use_valid:
        # Some randomization to make sure
        ids = rng.choice(len(x_train), len(x_train), replace=False)
        assert len(set(ids)) == len(ids) == len(x_train)
        x_train = x_train[ids]
        y_train = y_train[ids]
        assert len(x_train) == 50000, len(x_train)
        (x_train, y_train), (x_valid, y_valid) = (x_train[0:45000], y_train[0:45000]), \
            (x_train[-5000:], y_train[-5000:])

    meta_preprocessing = {"type": preprocessing}
    if preprocessing == "center":
        mean = np.mean(x_train, axis=0, keepdims=True)  # Pixel mean
        # Complete std as in https://github.com/gcr/torch-residual-networks/blob/master/data/cifar-dataset.lua#L3
        std = np.std(x_train)
        meta_preprocessing['mean'] = mean
        meta_preprocessing['std'] = std
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        if use_valid:
            x_valid = (x_valid - mean) / std
    elif preprocessing == "01":  # Required by scatnet
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        if use_valid:
            x_valid = x_valid / 255.0
    else:
        raise NotImplementedError("Not implemented preprocessing " + preprocessing)

    logging.info('x_train shape:' + str(x_train.shape))
    logging.info(str(x_train.shape[0]) + 'train samples')
    logging.info(str(x_test.shape[0]) + 'test samples')
    if use_valid:
        logging.info(str(x_valid.shape[0]) + 'valid samples')
    logging.info('y_train shape:' + str(y_train.shape))

    # Prepare test
    train = [x_train, y_train]
    test = [x_test, y_test]
    if use_valid:
        valid = [x_valid, y_valid]

    if use_valid:
        return train, valid, test, {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
            "x_valid": x_valid, "y_valid": y_valid, "preprocessing": meta_preprocessing}
    else:
        # Using as valid test
        return train, test, test, {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
            "preprocessing": meta_preprocessing}


def get_mnist(which="fmnist", preprocessing="01", seed=777, select_classes=None, use_valid=True):
    """
    Returns
    -------
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
    """
    rng = np.random.RandomState(seed)

    if use_valid:
        logger.info("Using valid")
    else:
        logger.info("Using as valid test")

    if which == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
    elif which == "fmnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train, y_train = np.array(X_train).astype("float32"), np.array(y_train)
        X_test, y_test = np.array(X_test).astype("float32"), np.array(y_test)
        x_train = X_train.reshape(-1, 28, 28, 1)
        x_test = X_test.reshape(-1, 28, 28, 1)
    else:
        raise NotImplementedError()

    if select_classes is not None:
        logging.info("Selecting classes in MNIST {}".format(select_classes))
        pass

    # Permute
    ids_train = rng.choice(len(x_train), len(x_train), replace=False)
    ids_test = rng.choice(len(x_test), len(x_test), replace=False)
    x_train, y_train = x_train[ids_train], y_train[ids_train]
    x_test, y_test = x_test[ids_test], y_test[ids_test]

    logger.info("Loaded dataset using eval")

    if use_valid:
        assert len(x_train) == 60000, len(x_train)
        (x_train, y_train), (x_valid, y_valid) = (x_train[0:50000], y_train[0:50000]), \
            (x_train[-10000:], y_train[-10000:])

    if preprocessing == "center":
        mean = np.mean(x_train, axis=0, keepdims=True)  # Pixel mean
        # Complete std as in https://github.com/gcr/torch-residual-networks/blob/master/data/cifar-dataset.lua#L3
        std = np.std(x_train)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        if use_valid:
            x_valid = (x_valid - mean) / std
    elif preprocessing == "01":  # Required by scatnet
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        if use_valid:
            x_valid = x_valid / 255.0
    else:
        raise NotImplementedError("Not implemented preprocessing " + preprocessing)

    logger.info('x_train shape:' + str(x_train.shape))
    logger.info(str(x_train.shape[0]) + 'train samples')
    logger.info(str(x_test.shape[0]) + 'test samples')
    if use_valid:
        logger.info(str(x_valid.shape[0]) + 'valid samples')
    logger.info('y_train shape:' + str(y_train.shape))

    # Prepare test
    train = [x_train, y_train]
    test = [x_test, y_test]
    if use_valid:
        valid = [x_valid, y_valid]

    y_valid = np_utils.to_categorical(y_valid, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    # TODO: Add y_train

    if use_valid:
        return train, valid, test, {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
            "x_valid": x_valid, "y_valid": y_valid}
    else:
        # Using as valid test
        return train, test, test, {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
