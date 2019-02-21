"""
Stolen from https://gist.github.com/p-baleine/e6f57dd6f89c932eccadec5d78fed0b5

TODO: Consider using CuDNNLSTM?
"""

from keras.layers import Dense, Activation, Dropout, LSTM, Input
from keras.layers.embeddings import Embedding
from keras.models import Model

def ptb_lstm(config, training=None):
    """Return the PTB model."""
    num_steps = config['num_steps']
    num_layers = config['num_layers']
    size = config['hidden_size']
    vocab_size = config['vocab_size']
    keep_prob = config['keep_prob']

    prefix = ""
    if training is False:
        prefix = "inference_"

    lstm_parameters = {
        "units": size,
        "unit_forget_bias": False,
        "stateful": True,
        "unroll": True, # For Lanczos
    }

    x = Input(batch_shape=(config['batch_size'], num_steps))

    h = Embedding(vocab_size, size, name=prefix+"embedding")(x)

    if keep_prob < 1:
        h = Dropout(1 - keep_prob)(h, training=training)

    for i in range(num_layers - 1):
        h = LSTM(return_sequences=True, name=prefix+"lstm1", **lstm_parameters)(h)
        if keep_prob < 1:
            h = Dropout(1 - keep_prob)(h, training=training)

    h = LSTM(return_sequences=True, name=prefix+"lstm2", **lstm_parameters)(h)
    if keep_prob < 1:
        h = Dropout(1 - keep_prob)(h, training=training)

    h = Dense(vocab_size, name=prefix+"pre_softmax")(h)
    h = Activation('softmax', name=prefix+"post_softmax")(h)

    model = Model(inputs=x, outputs=h)

    setattr(model, "steerable_variables", {})

    return model
