"""
Simple CNN model
"""

from keras.layers import *
from keras.layers import Dense, Conv2D
from keras.models import Model
from keras.regularizers import l2 as l2_reg
from keras.initializers import constant

def build_simple_cnn(input_shape=(3, 32, 32), dropout=0.0, l2=0., training=None, n_filters=32, activation="relu",
        n_dense=128, kernel_size=3, n1=1, n2=1, nb_classes=10, bn=False, use_bias=True, init="glorot_uniform"):
    inputs = Input(shape=input_shape)

    x = inputs

    if training == False:
        prefix = "inference_"
    else:
        prefix = ""

    for id in range(n1):
        prefix_column = str(id) if id>0 else ""
        x = Conv2D(n_filters, (kernel_size, kernel_size), padding='same', kernel_regularizer=l2_reg(l2),
            name=prefix_column + prefix+"conv1", use_bias=use_bias, kernel_initializer=init)(x)

        if bn:
            x = BatchNormalization(axis=3, name=prefix_column + prefix+"bn1")(x, training=training)

        x = Activation(activation, name=prefix_column + "act_1")(x)

        x = Conv2D(n_filters, (kernel_size, kernel_size), kernel_regularizer=l2_reg(l2), use_bias=use_bias
                , name=prefix_column + prefix+"conv2", kernel_initializer=init)(x)
        if bn:
            x = BatchNormalization(axis=3, name=prefix_column + prefix+"bn2")(x, training=training)

        x = Activation(activation, name=prefix_column + "act_2")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    for id in range(n2):
        prefix_column = str(id) if id > 0 else ""
        x = Conv2D(n_filters*2, (kernel_size, kernel_size), use_bias=use_bias, padding='same', kernel_regularizer=l2_reg(l2),
            name=prefix_column + prefix+"conv3", kernel_initializer=init)(x)

        if bn:
            x = BatchNormalization(axis=3, name=prefix_column + prefix+"bn3")(x, training=training)

        x = Activation(activation, name=prefix_column + "act_3")(x)
        x = Conv2D(n_filters*2, (kernel_size, kernel_size), use_bias=use_bias,
            kernel_regularizer=l2_reg(l2), name=prefix_column + prefix+"conv4", kernel_initializer=init)(x)

        if bn:
            x = BatchNormalization(axis=3,  name=prefix_column + prefix+"bn4")(x, training=training)

        x = Activation(activation, name=prefix_column + "act_4")(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(n_dense, kernel_regularizer=l2_reg(l2), use_bias=use_bias, name=prefix+"dense2", kernel_initializer=init)(x)

    if bn:
        x = BatchNormalization(name=prefix+"bn5")(x, training=training)

    x = Activation(activation, name="act_5")(x) # Post act
    if dropout > 0:
        x = Dropout(dropout)(x, training=training)
    x = Dense(nb_classes, activation="linear", use_bias=use_bias, name=prefix+"pre_softmax", kernel_initializer=init)(x)
    x = Activation(activation="softmax", name=prefix+"post_softmax")(x)

    model = Model(inputs=[inputs], outputs=[x])

    setattr(model, "steerable_variables", {})

    return model