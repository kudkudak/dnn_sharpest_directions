"""
Residual network definitions
"""

import logging

from keras.initializers import VarianceScaling
from keras.layers import *
from keras.layers import AveragePooling2D
from keras.layers import Dense, Conv2D, Add
from keras.models import Model
from keras.regularizers import l2 as l2_reg

from src import DATA_FORMAT

logger = logging.getLogger(__name__)

from keras.activations import *


def he_normal_scaled(scale):
    """He normal variance scaling initializer.

    # References
        He et al., http://arxiv.org/abs/1502.01852
    """
    return VarianceScaling(scale=2. * scale,
        mode='fan_in',
        distribution='normal')


class _residual_block():
    # Full pre-activation from https://arxiv.org/pdf/1603.05027.pdf
    # different than stochastic depth (original)

    def __init__(self, nb_filters=16, subsample=1, l2=0.0,
            id=0, activation="relu", scale_init=1.0, dropout=0.5, normalization="bn", identity=True, prefix_name=""):
        self.layers = []
        self.prefix_name = prefix_name
        self.id = id

        if normalization == "bn":
            y = BatchNormalization(axis=3 if DATA_FORMAT == "channels_last" else 1,
                name=prefix_name + "block_bn_1_id_" + str(id))
            self.layers.append(y)
        elif normalization == "none":
            pass
        else:
            raise NotImplementedError()

        if activation == "relu":
            y = Activation('relu', name=prefix_name + 'act_0_id_' + str(id))
        elif activation == "tanh":
            y = Activation("tanh", name=prefix_name + 'act_0_id_' + str(id))
        else:
            raise NotImplementedError()
        self.layers.append(y)

        # TODO: Is seed avoided correctly?
        init_fnc = he_normal_scaled(scale_init)
        y = Conv2D(nb_filters, 3, strides=(subsample, subsample), kernel_regularizer=l2_reg(l2),
            name=prefix_name + "conv_1_id_" + str(id), padding='same', data_format=DATA_FORMAT,
            kernel_initializer=init_fnc)
        self.layers.append(y)

        if normalization == "bn":
            y = BatchNormalization(axis=3 if DATA_FORMAT == "channels_last" else 1,
                name=prefix_name + "block_bn_2_id_" + str(id))
            self.layers.append(y)
        elif normalization == "none":
            pass
        else:
            raise NotImplementedError()

        if activation == "relu":
            y = Activation('relu', name=prefix_name + 'act_1_id_' + str(id))
        elif activation == "tanh":
            y = Activation('tanh', name=prefix_name + 'act_1_id_' + str(id))
        else:
            raise NotImplementedError()
        self.layers.append(y)

        y = Dropout(dropout)
        self.layers.append(y)

        y = Conv2D(nb_filters, 3, strides=(1, 1), kernel_regularizer=l2_reg(l2),
            name=prefix_name + "conv_2_id_" + str(id),
            padding='same', data_format=DATA_FORMAT, kernel_initializer=init_fnc)
        self.layers.append(y)

        if subsample > 1:
            self.shortcut = Conv2D(nb_filters, (1, 1), strides=(subsample, subsample),
                name=prefix_name + "shortcut_id_" + str(id),
                kernel_regularizer=l2_reg(l2),
                kernel_initializer=init_fnc, padding='same', data_format=DATA_FORMAT)

        self.identity = identity

        self.id_bn = 0
        self.id_scaler = 0

    def call(self, x, training=None):
        y = x
        for layer in self.layers:
            if isinstance(layer, Dropout) or isinstance(layer, BatchNormalization):
                y = layer(y, training=training)
            else:
                y = layer(y)

        if 'shortcut' in self.__dict__:
            x = self.shortcut(x)

        out = Add()([y,x])
        # out = merge([y, x], mode='sum')
        return out, y, x


def build_resnet(n, l2, nb_classes, pool_size=8, n_stages=3, init_scale=1.0,
        input_dim=[3, 32, 32], k=1, normalization="bn", resnet_dropout=0.2,
        resnet_activation='relu', training=None, seed=777):
    """
    Builds "original" resnet.
    """

    if training is False:
        prefix_name = "inference_"
    else:
        prefix_name = ""

    F_N = 0
    F_blocks, F_states, states = [], [], []
    init_fnc = he_normal_scaled(init_scale)
    inputs = Input(shape=input_dim)

    x = Conv2D(k * 16, (3, 3), padding='same', name=prefix_name + "first_conv",
        data_format=DATA_FORMAT, kernel_regularizer=l2_reg(l2), kernel_initializer=init_fnc)(inputs)

    logging.info(DATA_FORMAT)
    logging.info("First x shape " + str(K.int_shape(x)))

    id_layer = 0
    for stage in range(n_stages):

        # Configure stage
        n_filters = k * ((2 ** stage) * 16)
        n_layers_stage = n
        logging.info("stage {} n_filters {} n_layers {}".format(stage,
            n_filters, n_layers_stage
        ))

        for i in range(n_layers_stage):
            F_block = _residual_block(nb_filters=n_filters,
                scale_init=init_scale,
                normalization=normalization,
                prefix_name=prefix_name,
                id="{}_{}".format(i, stage),
                activation=resnet_activation,
                subsample=1 if (i > 0 or stage == 0) else 2,
                l2=l2,
                dropout=resnet_dropout)

            x_next, residue, x = F_block.call(x, training=training)

            # Book-keeping
            F_N += 1
            F_blocks.append(F_block)
            states.append(x)
            F_states.append(residue)
            x = x_next
            logging.info(K.int_shape(x))
            id_layer += 1

    # Last state (so there are F_N + 1 states)
    states.append(x)

    if normalization == "bn":
        post_bn = BatchNormalization(axis=3 if DATA_FORMAT == "channels_last" else 1,
            name=prefix_name + "post_bn")
    elif normalization == "none":
        post_bn = lambda x, training: x
    else:
        raise NotImplementedError()
    post_act = Activation('relu', name=prefix_name + "post_act")
    logging.info(K.int_shape(x))
    post_pool = AveragePooling2D(pool_size=(pool_size, pool_size),
        strides=None, padding='valid',
        data_format=DATA_FORMAT, name=prefix_name + "post_avg")
    post_flatten = Flatten()

    pre_softmax = Dense(nb_classes, activation='linear', kernel_regularizer=l2_reg(l2),
        name=prefix_name + "pre_softmax")
    post_softmax = Activation(activation='softmax', name=prefix_name + "post_softmax")
    prediction_layer = lambda xx: post_softmax(pre_softmax(xx))
    predictions = prediction_layer(post_flatten(post_pool(post_act(post_bn(x, training=training)))))

    model = Model(input=inputs, output=predictions)

    meta = {"F_states": F_states, "states": states, "F_blocks": F_blocks, "F_N": F_N,
        "postnet": lambda xx: prediction_layer(post_flatten(post_pool(post_act(post_bn(xx)))))}

    setattr(model, "steerable_variables", {})

    return model, meta
