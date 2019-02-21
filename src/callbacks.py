# -*- coding: utf-8 -*-
"""
Callbacks used in the project
"""
from keras.callbacks import *

try:
    import tensorflow
    import tensorflow as tf
except:
    pass

import pandas as pd
import os

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
from collections import defaultdict
from os.path import join
import logging
logger = logging.getLogger(__name__)

import h5py

class DistanceTravelled(Callback):
    """
    Callback computing distance travelled
    """

    def __init__(self, compare_path, prefix=""):
        self.compare_path = compare_path # typically init
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        W = h5py.File(self.compare_path)
        if "model_weights" in W:
            W = W['model_weights']

        logger.info("Adding dist travelled")
        all_dist, all_r = [], []
        for p in self.model.trainable_weights:
            layer_name, weight_name = p.name.split("/")
            dist = np.linalg.norm((W[layer_name][layer_name][weight_name] - K.get_value(p)).reshape(-1,))
            r = np.linalg.norm(K.get_value(p).reshape(-1,))
            all_dist.append(dist)
            all_r.append(r)
            logs[self.prefix + 'dist/' + p.name] = dist
            logs[self.prefix + 'r/' + p.name] = r
        logs[self.prefix + 'dist/' + 'all'] = sum(all_dist)
        logs[self.prefix + 'r/' + 'all'] = sum(all_r)



def add_dead_neurons_metric_tensors(model, tensors, names, eps=1e-5, data_format="channels_last", prefix=""):
    """
    Adds trackers of dead neurons and mean activation
    """
    output_at = defaultdict(int)  # In case of shared layers
    if data_format != "channels_last":
        raise NotImplementedError()

    for eps in [1e-5, 1e-4, 1e-3]:
        for h, name in zip(tensors, names):
            model.metrics_names.append(prefix + "h_dead_{}/{}".format(eps, name))
            if K.ndim(h) == 4:
                model.metrics_tensors.append(K.mean(K.greater_equal(eps, K.mean(K.abs(h), axis=0)), axis=(0, 1, 2)))
            elif K.ndim(h) == 3:
                model.metrics_tensors.append(K.mean(K.greater_equal(eps, K.mean(K.abs(h), axis=0)), axis=(0, 1)))
            elif K.ndim(h) == 2:
                model.metrics_tensors.append(K.mean(K.greater_equal(eps, K.mean(K.abs(h), axis=0)), axis=(0,)))
            else:
                raise NotImplementedError()


logger = logging.getLogger(__name__)


def _batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class PickableReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    # Example

    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```

    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
            verbose=0, mode='auto', epsilon=1e-4, cooldown=0, min_lr=0):
        super(PickableReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
            (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)

                        if hasattr(self.model, "steerable_variables"):
                            if "overshoot" in self.model.steerable_variables:
                                old_val = float(K.get_value(self.model.steerable_variables['overshoot']))
                                print("Setting overshoot to " + str(old_val * self.factor))
                                K.set_value(self.model.steerable_variables['overshoot'], old_val * self.factor)

                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                  'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0

    def __getstate__(self):
        state = self.__dict__.copy()
        if "monitor_op" in state:
            del state['monitor_op']
        return state


class DumpTensorflowSummaries(Callback):
    def __init__(self, save_path):
        self._save_path = save_path
        self._examples_since_save = 0
        super(DumpTensorflowSummaries, self).__init__()

    @property
    def file_writer(self):
        from keras.backend.tensorflow_backend import _SESSION
        if not hasattr(self, '_file_writer'):
            if _SESSION is not None:
                self._file_writer = tensorflow.summary.FileWriter(
                    self._save_path, flush_secs=10., graph=_SESSION.graph)
            else:
                self._file_writer = tensorflow.summary.FileWriter(
                    self._save_path, flush_secs=10.)
        return self._file_writer

    def on_epoch_end(self, epoch, logs=None):
        summary = tensorflow.Summary()
        for key, value in logs.items():
            try:
                float_value = float(value)
                value = summary.value.add()
                value.tag = key
                value.simple_value = float_value
            except:
                pass
        self.file_writer.add_summary(
            summary, epoch)


class LambdaCallbackPickable(LambdaCallback):
    def set_callback_state(self, callback_state={}):
        self.callback_state = callback_state

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['on_epoch_end']
        del state['on_epoch_begin']
        del state['on_batch_end']
        del state['on_train_end']
        del state['on_train_begin']
        del state['on_batch_begin']
        return state

    def __setstate__(self, newstate):
        newstate['on_epoch_end'] = self.on_epoch_end
        newstate['on_train_end'] = self.on_train_end
        newstate['on_epoch_begin'] = self.on_epoch_begin
        newstate['on_train_begin'] = self.on_train_begin
        newstate['on_batch_end'] = self.on_batch_end
        newstate['on_batch_begin'] = self.on_batch_begin
        self.__dict__.update(newstate)


class LambdaCallbackPickableEveryKExamples(LambdaCallback):
    """
    Runs lambda every K examples.

    Assumes batch_size in batch logs
    """

    def __init__(self,
            on_k_examples=None,
            k=45000,  # Epoch of mnist ;)
            call_on_batch_0=True,
            **kwargs):
        super(LambdaCallback, self).__init__()
        self.__dict__.update(kwargs)
        self.examples_seen = 0
        self.call_on_batch_0 = call_on_batch_0
        self.examples_seen_since_last_call = 0
        self.k = k
        self.on_k_examples = on_k_examples
        self.calls = 0

    def on_batch_end(self, batch, logs=None):
        assert "batch_size" in logs
        self.examples_seen += logs['batch_size']
        self.examples_seen_since_last_call += logs['batch_size']

        if (self.call_on_batch_0 and batch == 0) \
                or self.examples_seen_since_last_call > self.k:
            logger.info("Batch " + str(batch))
            logger.info("Firing on K examples, ex seen = " + str(self.examples_seen))
            logger.info("Firing on K examples, ex seen last call = " + str(self.examples_seen_since_last_call))
            self.on_k_examples(self.calls, self.examples_seen, logs)
            self.examples_seen_since_last_call = 0
            self.calls += 1

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['on_k_examples']
        return state

    def __setstate__(self, newstate):
        if hasattr(self, "on_k_examples"):
            newstate['on_k_examples'] = self.on_k_examples
        self.__dict__.update(newstate)



class MeasureTime(Callback):
    def __init__(self):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_train_begin(self, logs=None):
        self.train_start = time.time()

    def on_batch_begin(self, batch, logs=None):
        self.batch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs['t_epoch'] = np.float64(time.time() - self.epoch_start)

    def on_batch_end(self, batch, logs=None):
        logs['t_batch'] = np.float64(time.time() - self.batch_start)

    def on_train_end(self, logs=None):
        logs['t_train'] = np.float64(time.time() - self.train_start)


from scipy.sparse.linalg import svds


def compute_svd(model, k=2):
    logs = {}
    for l in model.layers:
        if hasattr(l, "kernel"):
            W = K.get_value(l.kernel)
            if W.ndim == 4:
                # Slightly arbitrary, but consistent, way of estimating this
                # Would need to think more what is a more appropriate way?
                # This is same as in https://arxiv.org/pdf/1705.10941.pdf
                W = W.transpose(3, 0, 1, 2).reshape(W.shape[0], -1)
            if W.shape[0] == 1:
                logger.warning(("Skipping " + str(W.shape)))
            else:
                _, S, _ = svds(W, k=k)
            for id in range(k):
                logs["SVD/" + l.name + "_k=" + str(id)] = S[id]

    return logs


class SVDWeights(Callback):
    def __init__(self, k=2):
        self.k = k

    def on_epoch_end(self, epoch, logs=None):
        logger.info("Computing SVD of weight matrices of the network")
        logs.update(compute_svd(self.model, self.k))
        logger.info("Done computing SVD of weight matrices of the network")


class StoreBatchLogs(Callback):
    """Callback that stores every-so-often history from batch

    Adds its own key, "example_seen".

    Notes
    =====
    Requires specyfing batch size every epoch
    """

    def __init__(self, save_path=None, frequency=10000):
        self.frequency = frequency
        self.examples_seen = 0
        self.save_path = save_path
        self.examples_seen_since_last_population = 0
        self.history = defaultdict(list)  # OK to się save'uje
        super(StoreBatchLogs, self).__init__()

    def on_batch_end(self, batch, logs=None):
        assert "batch_size" in logs

        self.examples_seen += logs['batch_size']
        logs['examples_seen'] = self.examples_seen
        self.examples_seen_since_last_population += logs['batch_size']

        # Used by other callback
        setattr(self.model, "history_batch", self.history)

        if self.examples_seen_since_last_population > self.frequency:
            self.examples_seen_since_last_population = 0
            for k in logs:
                self.history[k].append(logs[k])

    def on_epoch_end(self, epoch, logs):
        # Do IO only once each epoch
        if self.save_path:
            pd.DataFrame(self.history).to_csv(self.save_path)
