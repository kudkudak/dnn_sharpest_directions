"""
Models used on imdb
"""

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

def build_cnn(max_features=5000, maxlen=400, dropout=0.2, embedding_dims=50,
        filters=250, kernel_size=3, hidden_dims=250, training=None):
    input = Input(shape=(maxlen,))

    x = Embedding(max_features, embedding_dims)(input)
    x = Dropout(dropout)(x, training=training)

    x = Conv1D(filters,
        kernel_size,
        padding='valid',
        activation='relu',
        strides=1)(x)
    # we use max pooling:
    x = GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = Dense(hidden_dims)(x)
    x = Dropout(dropout)(x, training=training)
    x = Activation('relu')(x)

    x = Dense(2)(x)
    predictions = Activation('softmax')(x)

    model = Model(input=input, output=predictions)

    return model