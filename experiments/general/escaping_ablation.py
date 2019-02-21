"""
Simple experiment to understand escaping
"""

import json

import pandas as pd
import tensorflow as tf
from keras.optimizers import SGD
from keras.utils import np_utils

from src.callback_constructors import add_lanczos_callbacks
from src.clr import *
from bin.train_resnet_cifar import init_data_and_model as resnet_idam
from bin.train_simple_cnn_cifar import init_data_and_model as simple_cnn_idam
from src.utils.vegab import wrap_no_config_registry

import tqdm
import os
import numpy as np

import logging
import keras.backend as K

def init_data_and_model(path):
    # A bit of a mouthful, but serves the purposs
    C = json.load(open(os.path.join(path, "config.json")))
    if os.path.exists(os.path.join(path, "train_resnet_cifar.py")):
        print("Initializing resnet")
        return resnet_idam(C)
    elif os.path.exists(os.path.join(path, "train_simple_cnn_cifar.py")):
        print("Initializing simpleCNN")
        return simple_cnn_idam(C)
    else:
        raise NotImplementedError()

def main(save_path="", E="", epoch=0, which="", N=1000, lrgeneral=-1.0, lrfactor=1.0, id=0, track_deltaW=False, linesearch=0):
    if os.path.exists(os.path.join(save_path,"H.csv")):
        print("Finished")
        return(0)

    K.clear_session()

    logger = logging.getLogger(__name__)

    C = config = json.load(open(os.path.join(E, "config.json")))
    checkpoint = os.path.join(E, "model_epoch_{}.h5".format(epoch))

    tf.set_random_seed(config['seed'])
    np.random.seed(config['seed'])

    (train, valid, test, meta_data), (model, model_inference, _) = init_data_and_model(E)

    config['lanczos_top_K'] = max(id + 2, 5)

    callbacks, lanczos_clbk = add_lanczos_callbacks(config=config, save_path=save_path,
        meta_data=meta_data, model=model, model_inference=model_inference, train=False,
        n_classes=int(config['which']))
    # TODO: Add second lanczos computation.
    optimizer = SGD(lr=config['lr'], momentum=config['m'])

    model.compile(optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'categorical_crossentropy'])

    model_inference.compile(optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'categorical_crossentropy'])

    model._make_train_function()

    lanczos_clbk._compile()

    parameter_names = lanczos_clbk.parameter_names
    parameter_shapes = lanczos_clbk.parameter_shapes
    mapping_g = {"names": parameter_names, "shapes": parameter_shapes}

    X, y = meta_data['x_train'], np_utils.to_categorical(meta_data['y_train'], 10)

    grad = K.gradients(model.total_loss, model.trainable_weights)
    calc_grad = K.function(model.inputs + model.targets + [K.learning_phase()] + model.sample_weights, grad)

    ## Some helpful functions ##

    def get_grad(xx, yy):
        z = calc_grad([xx, yy, 1, np.ones_like(yy[:, 0]).astype("float32")])
        return np.concatenate([zz.reshape(-1, ) for zz in z], axis=0)

    def get_random_grad():
        ids = np.random.choice(len(X), C['batch_size'], replace=False)
        return get_grad(X[ids], y[ids])

    def add_direction_to_weights(v, eps, mapping, model):
        idx = [0] + list(np.cumsum([np.prod(c) for c in mapping['shapes']]))
        setters = []
        for w in model.trainable_weights:
            if w.name in mapping['names']:
                idw = mapping['names'].index(w.name)
            else:
                idw = mapping['names'].index("inference_" + w.name)
            setters.append((w, K.get_value(w) + eps * v[idx[idw]: idx[idw + 1]].reshape(mapping['shapes'][idw])))
        K.batch_set_value(setters)

    def copy_weights(model_from, model_to):
        setters = []
        for wto, wfrom in zip(model_to.trainable_weights, model_from.trainable_weights):
            setters.append((wto, K.get_value(wfrom)))
        K.batch_set_value(setters)

    model.load_weights(checkpoint)
    model_inference.load_weights(checkpoint)
    topE, top_Ev = lanczos_clbk._compute_top_K()
    logger.info("Initial sharpness " + str(topE))

    if lrgeneral == -1:
        logger.info("Using lrgeneral from config")
        lrgeneral = C['lr']

    def get_dynamics(N=N, which="normal", lr_general=lrgeneral, lrfactor=lrfactor, alpha=0.3, id=0):
        model.load_weights(checkpoint)
        model_inference.load_weights(checkpoint)

        topE, top_Ev = lanczos_clbk._compute_top_K()
        dir = top_Ev[:, id]
        dir_start = dir
        H = []

        H_batch = model.evaluate(X[0:2250], y[0:2250])
        H_batch = dict(zip(model.metrics_names, H_batch))
        print(H_batch['loss'])
        print("=======")

        for _ in tqdm.tqdm(range(N), total=N):
            g = get_random_grad()

            if which == "onlytop":
                update = dir * np.dot(dir, g)
                lr = lr_general * lrfactor
            elif which == "removetop":
                update = g - dir * np.dot(dir, g)
                lr = lr_general * lrfactor
            elif which == "constanttop":
                update = dir_start * np.dot(dir_start, g)
                lr = lr_general * lrfactor
            elif which == "normal":
                update = g
                lr = lr_general * lrfactor
            elif which == "topscaled":
                onlytop = dir * np.dot(dir, g)
                removetop = g - dir * np.dot(dir, g)
                update = onlytop * lrfactor + removetop
                lr = lr_general
            elif which == "alltopscaled":
                update = g
                for iid in range(id):
                    onlytop = top_Ev[:, iid] * np.dot(top_Ev[:, iid], update)
                    removetop = update - top_Ev[:, iid] * np.dot(top_Ev[:, iid], update)
                    update = onlytop * lrfactor + removetop
                lr = lr_general
            elif which == "alldrop":
                update = g
                dropped = 0
                for iid in range(id):
                    onlytop = top_Ev[:, iid] * np.dot(top_Ev[:, iid], update)
                    removetop = update - top_Ev[:, iid] * np.dot(top_Ev[:, iid], update)
                    x = int(np.random.uniform(0, 1) > lrfactor)
                    dropped += x
                    update = onlytop * x + removetop

                print(("Dropped", dropped))
                lr = lr_general
            elif which == "bottomscaled":
                onlytop = dir * np.dot(dir, g)
                removetop = g - dir * np.dot(dir, g)
                update = onlytop + removetop*lrfactor
                lr = lr_general
            elif which == "constant_overshoot":
                update = (g - dir * np.dot(dir, g)) + dir * alpha * np.sign(np.dot(dir, g))
                lr = lr_general * lrfactor
            else:
                raise NotImplementedError()

            add_direction_to_weights(update, -lr, mapping=mapping_g, model=model)
            add_direction_to_weights(update, -lr, mapping=mapping_g, model=model_inference)

            # 5% of data
            H_batch = model.evaluate(X[0:2250], y[0:2250])
            H_batch = dict(zip(model.metrics_names, H_batch))
            print(H_batch['loss'])
            H.append(H_batch)
            topE, top_Ev = lanczos_clbk._compute_top_K()

            H[-1]['SN'] = topE[0]
            H[-1]['alpha'] = np.dot(dir, g)
            H[-1]['alpha_update'] = np.dot(dir, update)

            dir = top_Ev[:, id]
        H = pd.DataFrame(H)
        return H

    H = get_dynamics(which=which, id=id)
    print("Saving to " + os.path.join(save_path, "H.csv"))
    H.to_csv(os.path.join(save_path, "H.csv"))

if __name__ == "__main__":
    wrap_no_config_registry(main)
