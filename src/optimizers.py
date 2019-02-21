"""
Optimizers used in the project
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

from keras.optimizers import *
from keras.legacy import interfaces

class NSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0., decay=0., rho=0.9, overshoot=3,
            KK=0, burnin=0, type="correct_v",
            nesterov=False, **kwargs):
        super(NSGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.overshoot = K.variable(overshoot, name="overshoot")
        self.initial_decay = decay
        self.nesterov = nesterov
        self.rho = rho
        self.type = type

        # Measurements
        self.alpha = K.zeros((1, KK))
        self.alpha_normalized = K.zeros((1, KK))
        self.alpha_v = K.zeros((1, KK))
        self.alpha_v_normalized = K.zeros((1, KK))

    def set_projections(self, projections):
        self.projections = projections

    def set_model(self, model):
        self.model = model

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                K.dtype(self.decay))))

        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        accumulators = [K.zeros(shape) for shape in shapes]
        self.accumulators = accumulators

        # add metrics to the model
        for p, a in zip(params, accumulators):
            self.model.metrics_names.append("var2/{}".format(p.name))
            self.model.metrics_tensors.append(K.mean(K.reshape(a, (-1,))))

        self.weights = [self.iterations] + moments

        alpha, gnorm, vnorm = 0, 0, 0

        # How could I *increase* LR along flatter dirs?
        # TODO: Add a unit test.

        # Collect alpha
        for p, g, m in zip(params, grads, moments):
            dim_p = np.prod(K.int_shape(p))
            alpha += K.reshape(K.dot(K.reshape(g, (1, dim_p)), self.projections[p.name]), (1, -1))

        vs = []
        gs = []
        alpha_v = 0

        # Correct v
        for p, g, m in zip(params, grads, moments):
            g_p = K.shape(g)
            if self.type == "correct_g" or self.type == "correct_both":
                # Leaves unaffected for gamma=1
                g -= K.reshape(K.dot(alpha, K.transpose(self.projections[p.name])), g_p)
                g += K.get_value(self.overshoot) * K.reshape(K.dot(alpha, K.transpose(self.projections[p.name])), g_p)
            else:
                pass
            # Standard
            gs.append(g)
            v = self.momentum * m - lr * g  # velocity
            dim_v = np.prod(K.int_shape(v))
            alpha_v += K.reshape(K.dot(K.reshape(v, (1, dim_v)), self.projections[p.name]), (1, -1))
            self.updates.append(K.update(m, v))
            vs.append(v)

        # Correct g
        new_vs = []
        for p, g, v, a in zip(params, gs, vs, accumulators):
            g_p = K.shape(g)

            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))

            # Measurements
            gnorm = gnorm + K.sum(K.reshape(K.pow(g, 2.0), (-1,)))

            # This might bias momentum.
            if self.type == "correct_v" or self.type == "correct_both":
                new_v = v - K.reshape(K.dot(alpha_v, K.transpose(self.projections[p.name])), g_p)
                new_v += K.get_value(self.overshoot) * K.reshape(K.dot(alpha_v, K.transpose(self.projections[p.name])), g_p)
                new_p = p + new_v
                vnorm = vnorm + K.sum(K.reshape(K.pow(new_v, 2.0), (-1,)))
                new_vs.append(v)
            elif self.type == "correct_g":
                new_p = p + v
                vnorm = vnorm + K.sum(K.reshape(K.pow(v, 2.0), (-1,)))
                new_vs.append(v)
            else:
                raise NotImplementedError()

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

        self.updates.append(K.update(self.alpha, K.abs(alpha)))
        self.updates.append(K.update(self.alpha_normalized, K.abs(alpha) / K.sqrt(gnorm)))

        self.updates.append(K.update(self.alpha_v, K.abs(alpha_v)))
        self.updates.append(K.update(self.alpha_v_normalized, K.abs(alpha_v) / K.sqrt(vnorm)))

        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
            'momentum': float(K.get_value(self.momentum)),
            'decay': float(K.get_value(self.decay)),
            'nesterov': self.nesterov}
        base_config = super(NSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PtbSGD(Optimizer):
    def __init__(self, lr=1.0, decay=.5, epoch_size=1000,
            max_epoch=4, **kwargs):
        super(PtbSGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0.)
        self.base_lr = K.variable(lr)
        self.lr = K.variable(lr)
        self.epoch = K.variable(0)
        self.decay = K.variable(decay)
        self.decay_lr = K.variable(1.0)
        self.epoch_size = K.variable(epoch_size)
        self.max_epoch = K.variable(max_epoch)

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)

        logger.info(self.epoch_size)
        epoch = self.iterations // self.epoch_size

        decay_lr = K.pow(self.decay, K.switch(epoch - self.max_epoch > 0.,
            epoch - self.max_epoch,
            K.variable(0.)))
        epoch_lr = self.base_lr * decay_lr

        self.updates = [(self.iterations, self.iterations + 1.)]
        self.updates.append((self.lr, epoch_lr))
        self.updates.append((self.epoch, epoch))
        self.updates.append((self.decay_lr, decay_lr))

        for p, g in zip(params, grads):
            self.updates.append((p, p - epoch_lr * g))
        return self.updates

    def get_config(self):
        config = {'base_lr': self.base_lr,  # float(K.get_value(self.base_lr)),
            'decay': float(K.get_value(self.decay)),
            'epoch_size': float(K.get_value(self.epoch_size)),
            'max_epoch': float(K.get_value(self.max_epoch))}
        base_config = super(PtbSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_lr(self):
        return self.lr.eval()
