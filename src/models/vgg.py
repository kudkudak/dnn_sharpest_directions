"""
Simple implementation of vgg. Includes currently only VGG11 customized for cifar10.
"""

import logging

import keras.backend as K
from keras.engine import Model
from keras.layers import *

logger = logging.getLogger(__name__)

vgg_cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}


def vgg_cifar(nb_classes, data_format, config, training=None):
    """
    Build VGG. Follows roughly https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py, differs
    in BN. BN is more freely applied here.
    """

    if training is False:
        prefix_name = "inference_"
    else:
        prefix_name = ""

    id_bn = 0
    bn_use_beta = config.get("bn_use_beta", True)

    if data_format != "channels_last":
        raise NotImplementedError()

    x = Input(shape=(32, 32, 3))
    h = x
    if config['bn']:
        id_bn += 1
        h = BatchNormalization(axis=3, name=prefix_name + "bn_" + str(id_bn),
            center=bn_use_beta)(h, training=training)

    # Config 'A'
    layers = vgg_cfg[config['features']]

    steerable_variables = {}

    id = 0
    id_conv = 0
    for l in layers:

        logger.info(K.int_shape(h))

        if l == 'M':
            h = MaxPooling2D((2, 2), strides=(2, 2), border_mode='valid', data_format=data_format)(h)
        else:
            id += 1
            id_conv += 1

            # TODO: L2 - rozkminic
            name = "feature_conv_" + str(id_conv)
            h = Convolution2D(l, 3, strides=(1, 1), border_mode='same',
                name=prefix_name + name, data_format=data_format)(h)

            if config['bn']:
                id_bn += 1
                h = BatchNormalization(axis=3, name=prefix_name + "bn_" + str(id_bn), center=bn_use_beta)(h,
                    training=training)

            if config['activ'] == 'leakyrelu':
                h = LeakyReLU(name=prefix_name + "feature_conv_act_" + str(id))(h)
            else:
                h = Activation(config['activ'], name=prefix_name + "feature_conv_act_" + str(id))(h)

            if config['dropout_1']:
                h = Dropout(config['dropout_1'])(h, training=training)

    # 2. Time to classify
    h = Flatten()(h)
    for i in range(2):
        name = "dense_" + str(i)

        if config['dropout_2']:
            h = Dropout(config['dropout_2'])(h, training=training)

        h = Dense(config['dim_clf'], name=prefix_name + name)(h)  # 512
        if config['bn']:
            id_bn += 1
            h = BatchNormalization(name=prefix_name + "bn_" + str(id_bn), center=bn_use_beta)(h, training=training)

        if config['activ'] == 'leakyrelu':
            h = LeakyReLU(name=prefix_name + "act_dense_" + str(i))(h)
        else:
            h = Activation(name=prefix_name + "act_dense_" + str(i), activation=config['activ'])(h)

    if config.get('split_softmax', False):
        h = Dense(nb_classes, activation='linear', name=prefix_name + "pre_softmax")(h)
        h = Activation("softmax", name=prefix_name + "final_softmax")(h)
    else:
        h = Dense(nb_classes, activation='softmax', name=prefix_name + "final_softmax")(h)

    model = Model([x], [h])

    steerable_variables["l2"] = [steerable_variables[k] for k in steerable_variables if k.startswith("l2")]
    setattr(model, "steerable_variables", steerable_variables)

    return model, {}
