"""
Keras callback using Lanczos iteration to compute top K eigenvalues based on Tensorflow routine lanczos_bidiag
"""

import logging

logger = logging.getLogger(__name__)
import os
import tqdm
import gc

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np

import keras.backend as K
from keras.callbacks import Callback
from keras.utils import np_utils

import tensorflow as tf

try:
    from tensorflow.python.ops.gradients import _hessian_vector_product
except ImportError:
    from tensorflow.python.ops.gradients_impl import \
        _hessian_vector_product
try:
    from tensorflow_forward_ad import forward_gradients
except ImportError:
    logger.info("Failed to import jacobian_vector_product")

from scipy.sparse import linalg


def _construct_linear_operator_batched_scipy(L, ws, data, bs, X, y, sample_weights, n_classes, dtype=np.float32):
    # A bit of painful configuration
    shapes = [K.int_shape(w) for w in ws]
    dim = np.sum([np.prod(s) for s in shapes])
    shape = (dim, dim)
    v_vect = tf.placeholder(tf.float32, [dim, ])
    v_reshaped = []
    cur = 0

    # TODO: Learning phase?
    logger.info("Calculating shapes")
    for s in shapes:
        v_reshaped.append(K.reshape(v_vect[cur:np.prod(s) + cur], s))
        cur += np.prod(s)
    logger.info("Consturcting hvp op")
    vector_product = _hessian_vector_product(L, ws, v_reshaped)
    logger.info("Done constructing hvp op")

    # Apply
    def apply_cpu(v):
        sess = K.get_session()
        if isinstance(data, list):
            res = [np.zeros(K.int_shape(vv), dtype=np.float32) for vv in ws]
            nb = int((bs - 1 + data[0].shape[0]) / bs)
            n = 0
            for id in tqdm.tqdm(range(nb), total=nb):
                x_batch = data[0][id * bs:(id + 1) * bs].astype(dtype)
                y_batch = data[1][id * bs:(id + 1) * bs].astype(dtype)
                n += len(x_batch)
                if y_batch.shape[-1] != n_classes:
                    y_batch = np_utils.to_categorical(y_batch, n_classes)
                fd = {v_vect: v,
                    X: x_batch,
                    y: y_batch,
                    K.learning_phase(): 1.0,
                    sample_weights: np.ones(shape=(len(x_batch),)),
                }
                ress = sess.run(vector_product, feed_dict=fd)
                for id2 in range(len(ws)):
                    # res[id2] += bs * ress[id2].reshape(-1, 1)
                    g = ress[id2]
                    # NOTE: Could be optimized
                    if isinstance(vector_product[id2], tf.IndexedSlices):
                        vals, indices = g.values, g.indices
                        res[id2][indices] += len(x_batch) * vals
                    else:
                        res[id2] += len(x_batch) * g

            return np.concatenate([g.reshape(-1, 1) for g in res], axis=0) / n
            # return np.concatenate(res, axis=0) / data[0].shape[0]
        else:
            raise NotImplementedError()

    A = linalg.LinearOperator(shape, matvec=apply_cpu)

    return A, apply_cpu, dim


class TopKEigenvaluesBatched(Callback):
    """
    Computes top K eigenvalues using Lanczos algorithm and supports batching internally

    See e.g. http://people.bath.ac.uk/mamamf/talks/lanczos.pdf
    """

    def __init__(self, data, K, batch_size,
            save_path="", sample_N=-1, learning_phase=0, save_eigenv=False, frequency=-1,
            debug=False, data_steps=None, prefix="", which="hessian", maxiter=None, n_classes=10, impl="scipy"):
        self.prefix = prefix
        self.frequency = frequency
        self.sample_N = sample_N
        self.n_classes = n_classes
        self.maxiter = maxiter
        self.K = K
        self.learning_phase = learning_phase
        self.which = which
        self.debug = debug
        self.original_data = data
        if isinstance(data, list):
            self.data = list(data)
        else:
            self.data = [[], []]
        self.counter = 0
        self.data_steps = data_steps
        self.batch_size = batch_size
        self.save_path = save_path
        self.save_eigenv = save_eigenv
        self._lanczos_tensor = None
        self._this_epoch_ev = None
        self._last_epoch_ev = None
        self._my_model = None
        self._op = None

        if impl != "scipy":
            raise NotImplementedError("Not implemented " + impl)

    def _compute_Hv(self, v):
        return self._op_python(v)

    def _compute_eigenvalue(self, v, M=10):
        alpha = np.inf
        v = v.reshape(-1, 1)
        for _ in range(M):
            v_new = self._op_python(v)
            alpha = np.linalg.norm(v_new)
            v = v_new / np.linalg.norm(v_new)
        return alpha

    def _compile(self):
        logger.info("Compiling")
        X, y, sample_weights = self.get_my_model()._feed_inputs[0], \
            self.get_my_model()._feed_targets[0], self.get_my_model()._feed_sample_weights[0]
        _op, _op_python, dim = _construct_linear_operator_batched_scipy(
            L=self.get_my_model().total_loss,
            n_classes=self.n_classes,
            ws=self.get_my_model().trainable_weights, data=self.data,
            X=X, y=y, sample_weights=sample_weights, bs=self.batch_size
        )
        self.dim = dim
        self._op = _op
        self._op_python = _op_python
        self.parameters = self.get_my_model().trainable_weights
        self.parameter_names = [p.name for p in self.get_my_model().trainable_weights]
        self.parameter_shapes = [K.int_shape(p) for p in self.get_my_model().trainable_weights]

    def set_my_model(self, model):
        # model.summary()
        self._my_model = model

    def get_my_model(self):
        # Defaults to model set by keras, but can be overwritten
        if self._my_model is not None:
            return self._my_model
        else:
            return self.model

    def _compute_top_K(self, v0=None):
        # Get data
        if self.sample_N != -1:
            logger.info("Resampling")
            if isinstance(self.original_data, list):
                ids = np.random.choice(len(self.data[0]), self.sample_N, replace=False)
                assert len(ids) == self.sample_N
                self.data[0] = self.data[0][ids]
                self.data[1] = self.data[1][ids]
            else:
                L = self.sample_N
                L_sampled = 0
                d = {"x_train": [], "y_train": []}
                while L_sampled < L:
                    x, y = next(self.original_data)
                    L_sampled += len(x)
                    d['x_train'].append(x)
                    d['y_train'].append(y)
                logger.info("Sampled {} wanted {}".format(L_sampled, self.sample_N))
                self.data[0] = np.concatenate(d['x_train'])
                self.data[1] = np.concatenate(d['y_train'])
                logger.info("Sampled {}".format(len(d['x_train'])))

        if self._op is None:
            self._compile()

        v0 = None
        eigenvalues, eigenvectors = linalg.eigsh(self._op, k=self.K, v0=v0, maxiter=self.maxiter)
        ids = np.argsort(-eigenvalues)  # TODO: Lanczos returns negative, sometimes.Not sure
        eigenvalues = eigenvalues[ids]
        eigenvectors = eigenvectors[:, ids]

        return eigenvalues, eigenvectors

    def get_current_eigenvalues(self):
        return self._this_epoch_e

    def get_current_eigenvectors(self):
        return self._this_epoch_ev

    def on_batch_begin(self, batch, logs):
        if self.frequency == -1:
            return
        else:
            if self.frequency % self.counter == 0:
                logger.info("Recomputing. Freq={}. Counter={}".format(self.frequency, self.counter))
                E, Ev = self._compute_top_K()
                logger.info(E)
                self._last_epoch_ev = self._this_epoch_ev
                self._this_epoch_ev = Ev
                self._this_epoch_e = E
        self.counter += 1

    def on_epoch_begin(self, epoch, logs):
        E, Ev = self._compute_top_K()
        logs[self.prefix + 'top_K_e'] = E
        logs[self.prefix + 'SN'] = E[0]
        if self.save_eigenv > 0:
            if epoch % self.save_eigenv == 0:
                logger.info("Saving eigenvectors")
                np.savez(os.path.join(self.save_path, self.prefix + "top_K_ev_{}.npz".format(epoch)), Ev=Ev)
                pickle.dump(
                    {"shapes": self.parameter_shapes, "names": self.parameter_names},
                    open(os.path.join(self.save_path, self.prefix + "top_K_ev_mapping.pkl".format(epoch)), "wb"))
        self._last_epoch_ev = self._this_epoch_ev
        self._this_epoch_ev = Ev
        self._this_epoch_e = E
        self._epoch_begin_logs = logs
        logger.info(logs[self.prefix + 'top_K_e'])

        logger.info("gc.collect()")
        gc.collect()  # Let go of previous Ev
        logger.info("gc.collect() done")

    def on_epoch_end(self, epoch, logs):
        # HACK!!! Seems like it is necessary, weirdly
        for k in self._epoch_begin_logs:
            logs[k] = self._epoch_begin_logs[k]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['data']
        del state['original_data']
        del state['parameters']
        del state['_lanczos_tensor']
        del state['_op_python']
        if "_op" in state:
            del state['_op']
        del state['_this_epoch_ev']
        if "_last_epoch_ev" in state:
            del state['_last_epoch_ev']
        del state['_this_epoch_e']
        del state['_my_model']
        return state
