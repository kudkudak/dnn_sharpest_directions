# -*- coding: utf-8 -*-
"""
Callbacks used in the project that run some sort of analysis related to curvature
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

import tqdm


def _add_direction_to_weights(model, v, eps, mapping):
    idx = [0] + list(np.cumsum([np.prod(c) for c in mapping['shapes']]))
    setters = []
    for w in model.trainable_weights:
        if "inference_" + w.name in mapping['names']:
            idw = mapping['names'].index("inference_" + w.name)
        else:
            idw = mapping['names'].index(w.name)
        setters.append((w, K.get_value(w) + eps * v[idx[idw]: idx[idw + 1]].reshape(mapping['shapes'][idw])))
    K.batch_set_value(setters)


def _make_loss_fnc(model, mode):
    if not hasattr(model, 'loss_fnc_' + str(mode)):
        inputs = (model._feed_inputs +
                  model._feed_targets +
                  model._feed_sample_weights)
        test_function = K.function(
            inputs,
            [model.total_loss],
            name='loss_fnc_' + str(mode),
            feed_dict={K.learning_phase(): 1})
        setattr(model, 'loss_fnc_' + str(mode), test_function)


def _get_loss(model, X, y, bs, mode=0):
    fnc_name = 'loss_fnc_' + str(mode)
    _make_loss_fnc(model, mode)
    fnc = getattr(model, fnc_name)

    tmp_fnc = model.test_function
    model.test_function = fnc

    vals = model.evaluate(X, y, verbose=0, batch_size=bs)

    model.test_function = tmp_fnc

    return vals


def _plot_loss_surface(model, v, Xb, yb, mapping, epses, save_path, bs, mode):
    # TODO: No scale?
    weight_path = os.path.join(save_path, "tmp.h5")
    model.save_weights(weight_path)
    loss_curve = []
    model.load_weights(weight_path)
    for eps in epses:
        _add_direction_to_weights(model, v, eps, mapping=mapping)
        # TODO: This uses testing mode. Ridiculousl
        # loss = model.evaluate(Xb, yb)[0] # TODO: Can be sped up
        loss = _get_loss(model, Xb, yb, bs, mode)
        loss_curve.append(loss)
        # TODO: Reload?
        _add_direction_to_weights(model, v, -eps, mapping=mapping)  # Erase
    # Reload jst to be sure
    model.load_weights(weight_path)
    return loss_curve


def _make_test_function_in_train_mode(model):
    if not hasattr(model, 'test_function_train_model') or model.test_function_train_model is None:
        inputs = (model._feed_inputs +
                  model._feed_targets +
                  model._feed_sample_weights)
        test_function = K.function(
            inputs,
            [model.total_loss] + model.metrics_tensors,
            updates=model.state_updates,
            name='test_function_train_model',
            feed_dict={K.learning_phase(): 1})

        setattr(model, "test_function_train_model", test_function)


def _evaluate(model, X, y, bs, mode=0):
    if mode == 0:
        pass
    elif mode == 1:
        _make_test_function_in_train_mode(model)
        tmp_fnc = model.test_function
        model.test_function = model.test_function_train_model
    else:
        raise NotImplementedError()

    vals = model.evaluate(X, y, verbose=0, batch_size=bs)
    metric_values = dict(zip(model.metrics_names, vals))

    if mode == 0:
        pass
    elif mode == 1:
        model.test_function = tmp_fnc
    else:
        raise NotImplementedError()

    return metric_values


def _compute_delta_metrics(lr, bs, model, X, y, get_random_grad, M, mapping_g, learning_phase, v=None):
    delta_metrics = []  # List of recorded deltas for every trial
    for _ in tqdm.tqdm(range(M), total=M):
        g = get_random_grad(batch_size=bs)

        # TODO: Add random dir
        if v is not None:
            dir = v
            eps = -lr * np.dot(g, v)
        else:
            dir = g
            eps = -lr

        _add_direction_to_weights(v=dir, eps=eps, mapping=mapping_g, model=model)

        metrics_after = _evaluate(model=model, X=X, y=y, mode=learning_phase, bs=bs)

        _add_direction_to_weights(v=dir, eps=-eps, mapping=mapping_g, model=model)
        delta_metrics.append(metrics_after)

    return delta_metrics


class DecomposeStepAnalysis(Callback):
    # Tries different LRs along sharpest dirs

    def __init__(self, X, y, batch_size, save_path,
            gammas=[0.25, 0.33, 0.5, 1., 2., 3., 4.],
            frequency=1, M=10, learning_phase=1, N=2560,
            sharpest_clbk=None):
        # 1280 is typical
        self.X = X
        self.N = N
        self.batch_size = batch_size
        self.y = y
        self.M = M
        self.frequency = frequency
        self.save_path = save_path
        self.learning_phase = learning_phase
        self.get_grad = None
        self.lanczos_clbk = sharpest_clbk
        self.gammas = list(gammas)

    # We need up to date evs!
    def on_epoch_begin(self, epoch, logs):
        X, y = self.X[0:self.N], self.y[0:self.N] # Not sure if this is optimal..
        cur_lr = K.get_value(self.model.optimizer.lr)
        LRS = [cur_lr * g for g in self.gammas]

        if epoch % self.frequency != 0:
            logger.info("Skipping computing decompose in epoch " + str(epoch))
            return

        result = {}
        result_path = os.path.join(self.save_path, "decompose_{}.json".format(int(epoch)))

        logger.info("Running decompose step analysis")

        if self.get_grad is None:
            self._compile()

        E, Ev = self.lanczos_clbk._this_epoch_e.copy(), self.lanczos_clbk._this_epoch_ev.copy()

        # Add baseline: random eigenvector.
        randomev = np.random.uniform(-1, 1, size=Ev[:,0:1].shape)
        randomev /= np.linalg.norm(randomev)
        Ev = np.concatenate([Ev, randomev], axis=1)

        metrics_before = _evaluate(model=self.model, X=X, y=y, mode=self.learning_phase, bs=self.batch_size)
        metrics_before['E'] = [float(vv) for vv in E]  # Just so it is serializable

        logger.info(E)
        mapping_E = {"names": self.lanczos_clbk.parameter_names, "shapes": self.lanczos_clbk.parameter_shapes}
        logger.info("Done computing Evs")

        result['before'] = metrics_before

        self.model.save_weights(os.path.join(self.save_path, "model_backup.h5"))
        for LR in tqdm.tqdm(LRS, total=len(self.gammas)):
            self.model.load_weights(os.path.join(self.save_path, "model_backup.h5"))  # Just reset, probably safer

            key = '{}_{}'.format(LR, self.batch_size)
            if key in result:
                continue

            # Compute for each eigenvector + for none (as a baseline - indicating this eigenvector is something nontrivial)
            dir_to_delta_metrics = {}
            for dir in tqdm.tqdm(range(len(E) + 1), total=len(E) + 1):
                key2 = dir
                if dir == len(E):
                    key2 = "random_ev"
                dir_to_delta_metrics[key2] = _compute_delta_metrics(v=Ev[:, dir], lr=LR, bs=self.batch_size,
                    X=X, y=y, learning_phase=self.learning_phase,
                    model=self.model, mapping_g=mapping_E, get_random_grad=self.get_random_grad, M=self.M)
            self.model.load_weights(os.path.join(self.save_path, "model_backup.h5"))
            baseline_metrics = _compute_delta_metrics(v=None, lr=LR, bs=self.batch_size,
                X=X, y=y, learning_phase=self.learning_phase,
                model=self.model, mapping_g=mapping_E, get_random_grad=self.get_random_grad, M=self.M)
            dir_to_delta_metrics['none'] = baseline_metrics

            result[key] = dir_to_delta_metrics

            logger.info("Dumping result. Btw, this should be close to 1:")
            logger.info(np.linalg.norm(Ev[:, dir]))
            json.dump(result, open(result_path, "w"))

        self.model.load_weights(os.path.join(self.save_path, "model_backup.h5"))

    def _compile(self):
        logger.info("Compiling")
        import keras.backend as K
        grad = K.gradients(self.model.total_loss, self.model.trainable_weights)
        calc_grad = K.function(self.model.inputs + self.model.targets + [
            K.learning_phase()] + self.model.sample_weights, grad)

        def get_grad(xx, yy):
            z = calc_grad([xx, yy, 1, np.ones_like(yy[:, 0]).astype("float32")])
            return np.concatenate([zz.reshape(-1, ) for zz in z], axis=0)

        def get_random_grad(batch_size):
            ids = np.random.choice(len(self.X), batch_size, replace=False)
            return get_grad(self.X[ids], self.y[ids])

        self.get_grad = get_grad
        self.get_random_grad = get_random_grad

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['X']
        del state['model']
        del state['y']
        del state['get_grad']
        del state['get_random_grad']
        del state['lanczos_clbk']
        logger.info(state)
        return state

# TODO: Important if less alig along other dirs. Would show unique
class FullBatchGradientAnalysis(Callback):
    # Computes alignment with sharpest dirs

    def __init__(self, X, y, batch_size, sharpest_clbk=None):
        self.X = X
        self.batch_size = batch_size
        self.y = y
        self.get_grad = None
        self.lanczos_clbk = sharpest_clbk

    def _compile(self):
        logger.info("Compiling")
        import keras.backend as K
        grad = K.gradients(self.model.total_loss, self.model.trainable_weights)
        calc_grad = K.function(self.model.inputs + self.model.targets + [
            K.learning_phase()] + self.model.sample_weights, grad)

        def get_grad(xx, yy):
            z = calc_grad([xx, yy, 1, np.ones_like(yy[:, 0]).astype("float32")])
            return np.concatenate([zz.reshape(-1, ) for zz in z], axis=0)

        self.get_grad = get_grad

    def on_epoch_end(self, epoch, logs):
        logger.info("Running full batch gradient analysis")

        if self.get_grad is None:
            self._compile()

        if self.lanczos_clbk is not None:
            eigv = self.lanczos_clbk._this_epoch_ev.copy()
        else:
            eigv = None

        nbatches = int((len(self.X) + self.batch_size - 1) / self.batch_size)

        # Computing the full batch gradient
        logger.info("Computing FBG")
        fbg = 0
        for b in range(nbatches):
            fbg += self.get_grad(self.X[b * self.batch_size:(b + 1) * self.batch_size],
                self.y[b * self.batch_size:(b + 1) * self.batch_size])
        fbg /= nbatches

        logger.info("Computing alignments")

        def cos(a, b):
            return float(a.dot(b) / (1e-10 + np.linalg.norm(a) * np.linalg.norm(b)))

        # Suggests a form of bug. But hard to understand, as code is simple and this culd be any vector..
        aligs = defaultdict(list)
        for b in range(nbatches):
            g = self.get_grad(self.X[b * self.batch_size:(b + 1) * self.batch_size],
                self.y[b * self.batch_size:(b + 1) * self.batch_size])
            aligs['fbg'].append(cos(g, fbg))
            if eigv is not None:
                for id in range(len(eigv.T)):
                    aligs[str(id)].append(cos(g, eigv[:, id]))
                    aligs['fbg_e' + str(id)].append(cos(fbg, eigv[:, id]))

        for k in aligs:
            logs['alig/' + k + "_mean"] = float(np.mean(aligs[k]))
            logs['alig/' + k + "_abs_mean"] = float(np.mean(np.abs(aligs[k])))
            logs['alig/' + k + "_std"] = float(np.std(aligs[k]))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['X']
        del state['y']
        del state['get_grad']
        del state['lanczos_clbk']
        return state


class CrossSectionLossSurfacePlotter(Callback):
    # Plots loss surface along sharpest dirs

    def __init__(self, X, y, ids, save_path, batch_size, training_mode=0, scaling="by_grad_norm_lr"):
        self.X = X
        self.save_path = save_path
        self.batch_size = batch_size
        self.training_mode = training_mode
        self.scaling = scaling
        self.y = y
        self._my_model = None
        self.ids = ids

        self._get_random_grad = None

    def set_mapping(self, mapping):
        self.mapping = mapping

    def set_directions(self, directions):
        self.directions = directions

    def set_my_model(self, model):
        self._my_model = model

    def get_my_model(self):
        # Defaults to model set by keras, but can be overwritten
        if self._my_model is not None:
            return self._my_model
        else:
            return self.model

    def _compile(self):
        assert self._my_model is not None
        grad = K.gradients(self.get_my_model().total_loss, self.get_my_model().trainable_weights)
        calc_grad = K.function(self.get_my_model().inputs + self.get_my_model().targets + [
            K.learning_phase()] + self.get_my_model().sample_weights, grad)

        def get_grad(xx, yy):
            z = calc_grad([xx, yy, self.training_mode, np.ones_like(yy[:, 0]).astype("float32")])
            return np.concatenate([zz.reshape(-1, ) for zz in z], axis=0)

        def get_random_grad():
            ids = np.random.choice(len(self.X), self.batch_size, replace=False)
            return get_grad(self.X[ids], self.y[ids])

        self._get_random_grad = get_random_grad

    def on_epoch_begin(self, epoch, logs):
        if self._get_random_grad is None:
            self._compile()

        logger.info("scaling=" + self.scaling)

        epses = [0.0, 0.25, 0.5, 1, 2, 4, 8, 16]
        epses += [-e for e in epses if e > 0]
        epses = sorted(epses)

        lr = K.get_value(self.model.optimizer.lr)

        logger.info("Computing cross-sections of the loss surface")

        # Sample 10 random gradients
        if self.scaling.startswith("by_grad_norm") or self.scaling == "by_grad_norm_lr":
            gs = [self._get_random_grad() for _ in range(10)]

        for id in self.ids:
            if self.scaling == "by_grad_norm_lr":
                scale = np.mean([lr * np.abs(g.dot(self.directions[:, id])) for g in gs])
            elif self.scaling.startswith("by_grad_norm"):
                prescale = float(self.scaling.split("_")[-1])
                scale = prescale * np.mean([np.abs(g.dot(self.directions[:, id])) for g in gs])
            else:
                scale = float(self.scaling)

            logger.info("Using scale " + str(scale) + " for dir " + str(id))
            loss_curve = _plot_loss_surface(self.get_my_model(), self.directions[:, id],
                self.X, self.y, self.mapping, epses=[e * scale for e in epses], save_path=self.save_path,
                bs=self.batch_size, mode=self.training_mode)
            logs['loss_curve_{}'.format(id)] = loss_curve
            logs['scale_{}'.format(id)] = scale

        logger.info("Done computing cross-sections of the loss surface")
        self._epoch_begin_logs = logs
        pickle.dump(epses, open(join(self.save_path, "loss_surfaces_eps.pkl"), "wb"))

    def on_epoch_end(self, epoch, logs):
        # HACK!!! Seems like it is necessary, weirdly
        for k in self._epoch_begin_logs:
            logs[k] = self._epoch_begin_logs[k]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['X']
        del state['y']
        del state['_get_random_grad']
        if "_my_model" in state:
            del state['_my_model']
        if "mapping" in state:
            del state['mapping']
        if "directions" in state:
            del state['directions']
        return state
