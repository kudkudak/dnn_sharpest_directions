# -*- coding: utf-8 -*-
"""
Data getters for PTB

Stolen from https://gist.github.com/p-baleine/e6f57dd6f89c932eccadec5d78fed0b5
"""

from collections import Counter
import numpy as np
import os

from src import DATA_DIR

import logging
logger = logging.getLogger(__name__)

PTB_DATA_DIR = os.path.join(DATA_DIR, "simple-examples", "data")

def _read_words(filename):
    with open(filename) as f:
        return f.read().replace('\n', '<eos>').split()

def _build_vocab(filename, vocab_size=None):
    data = _read_words(filename)
    counter = Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    if vocab_size is not None:
        logger.warning("Cutting vocab to {}".format(vocab_size))
        word_to_id = {k: min(word_to_id[k], vocab_size - 1) for k in word_to_id}

    id_to_word = dict((i, v) for v, i in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data]

def ptb_raw_data(data_path=PTB_DATA_DIR, vocab_size=None):
    train_path = os.path.join(data_path, 'ptb.train.txt')
    valid_path = os.path.join(data_path, 'ptb.valid.txt')
    test_path = os.path.join(data_path, 'ptb.test.txt')
    word_to_id, id_to_word = _build_vocab(train_path, vocab_size=vocab_size)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return word_to_id, id_to_word, train_data, valid_data, test_data

def ptb_iterator(raw_data, batch_size, num_steps, vocab_size=None):
    # Import here to cut dependencies
    from keras.utils.np_utils import to_categorical

    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    epoch_size = (batch_len - 1) // num_steps
    # TODO: No shuffling?? weird.
    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        if vocab_size is not None:
            y = to_categorical(y, num_classes=vocab_size)
        yield (x, y)

if __name__ == "__main__":
    word_to_id, id_to_word, train_data, valid_data, test_data = ptb_raw_data()
    train_it = ptb_iterator(train_data, batch_size=128, num_steps=5)
    print(next(train_it))
