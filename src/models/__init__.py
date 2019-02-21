"""
Model definitions

TODO: BUG! All evaluations are in train mode.. Damn..!
"""

import keras
import keras.backend as K

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Conv2D, merge
from keras.layers import Input, Lambda, InputSpec
from keras.regularizers import l2 as l2_reg
from keras.layers import AveragePooling2D, Layer
from keras.backend.common import _EPSILON
from keras.initializers import he_normal, normal, VarianceScaling, constant, RandomNormal
from keras.layers import Dense, Conv2D
from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.models import Model
from keras.initializers import truncated_normal, glorot_uniform, VarianceScaling

import os
import logging
logger = logging.getLogger(__name__)

def linear_regression(input_dim, config):
    inp = Input((input_dim,))
    if config['dropout'] > 0:
        x = Dropout(config['dropout'])(inp)
    else:
        x = inp

    if config['zero_init']:
        kernel_initializer = 'zeros'
    else:
        kernel_initializer = 'glorot_uniform'


    dense = Dense(1, input_dim=input_dim, use_bias=config['use_bias'], name="dense",
        activation='linear', kernel_initializer=kernel_initializer)
    y = dense(x)
    model = Model([inp], y)
    return model


def logreg(input_dim, nb_classes, config):
    inp = Input((input_dim,))
    if config['dropout'] > 0:
        x = Dropout(config['dropout'])(inp)
    else:
        x = inp
    dense = Dense(nb_classes, input_dim=input_dim, use_bias=config['use_bias'], name="dense",
        activation='softmax')
    y = dense(x)
    model = Model([inp], y)
    return model


def binary_logreg(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(2, activation='softmax'))
    return model

def simple_cnn_cifar10(config, training=True):
    inp = Input((32, 32, 3,))
    h = inp
    h = Conv2D(config['n_filters'], (3, 3),
        input_shape=(32, 32, 3), padding='same', activation=config['act'], name="conv_1")(h)
    h = Activation(activation=config['act'], name="conv_act_1")(h)
    if config['dropout_1'] > 0:
        h = Dropout(config['dropout_1'])(h, training=training)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Conv2D(config['n_filters'], (3, 3), name="conv_2", padding='same')(h)
    h = Activation(activation=config['act'], name="conv_act_2")(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Flatten()(h)
    h = Dense(config['dim'], name="final_dense", activation='relu')(h)
    if config['dropout_2'] > 0:
        h = Dropout(config['dropout_2'])(h, training=training)
    h = Dense(10, name="final_softmax", activation='softmax')(h)
    model = Model([inp], [h])
    setattr(model, "steerable_variables", {})
    return model, {}

def simple_multimodal_mlp(config, training=None):
    """
    Builds simple MLP

    Follows https://arxiv.org/pdf/1501.00102.pdf
    """

    if training is False:
        prefix_name = "inference_"
    else:
        prefix_name = ""

    img_rows, img_cols = 28, 28
    num_classes = 10
    input_shape = (img_rows, img_cols, 1)
    segment_shape = (img_rows / 2, img_cols / 2, 1)

    seed_init = config['seed']
    freeze_modalities = eval(config.get("freeze_modalities", "[]"))

    # (28, 28, 1)
    x = Input(input_shape)

    def get_segment(x, id):
        """
        0 1
        2 3
        """
        if id == 0:
            return x[:, 0:(img_rows / 2), :][:, :, 0:(img_cols / 2)]
        elif id == 1:
            return x[:, 0:(img_rows / 2), :][:, :, (img_cols / 2):]
        elif id == 2:
            return x[:, (img_rows / 2):, :][:, :, 0:(img_cols / 2)]
        elif id == 3:
            return x[:, (img_rows / 2):, :][:, :, (img_cols / 2):]
        else:
            raise NotImplementedError()

    segments = [Flatten()(
        Lambda(get_segment, arguments={"id": id}, output_shape=segment_shape)
            (x)) for id in range(4)]

    steerable_variables = {}

    segments_h = []
    for h_id, h in enumerate(segments):
        for id in range(config['k']):
            seed_init += 1

            dense = Dense(config['dim'],
                activation='linear',
                name=prefix_name + "dense_segment" + str(h_id) + "_" + str(id),
                kernel_regularizer=l2_reg(config['l2']))

            if h_id in freeze_modalities:
                logging.warning("Freezing modality " + str(h_id))
                dense.trainable = False

            h = dense(h)
            if config['bn']:
                h = BatchNormalization(axis=1, name=prefix_name+"bn_" + str(h_id) + "_" + str(id))(h, training=training)

            h = Activation(config['activation'], name=prefix_name+"act_" + str(h_id) + "_" + str(id))(h)
            h = Dropout(config['dropout'])(h, training=training)


        segments_h.append(h)

    h = merge(segments_h, mode="concat", concat_axis=1, name="merger_modalities")

    h = Dense(config['dim2'],
        activation='linear',
        name=prefix_name + "final_dense",
        kernel_regularizer=l2_reg(config['l2']))(h)
    if config['bn']:
        h = BatchNormalization(axis=1, name=prefix_name + "final_bn")(h, training=training)
    h = Activation(activation=config['activation'], name=prefix_name + "final_dense_act")(h)
    h = Dropout(config['dropout'])(h, training=training)

    h = Dense(num_classes, name=prefix_name + "pre_softmax", activation="linear")(h)
    out = Activation("softmax",  name=prefix_name + "final_softmax")(h)

    model = Model([x], out)

    setattr(model, "steerable_variables", steerable_variables)

    logger.info(steerable_variables)

    return model