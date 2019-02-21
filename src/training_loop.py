# -*- coding: utf-8 -*-
"""
Training loop
"""
import os
import pickle
import logging
import h5py
import pandas as pd

import keras.backend as K

from src.callback_constructors import construct_training_loop_callbacks
from src.callbacks import LambdaCallbackPickable

logger = logging.getLogger(__name__)

def training_loop(model, train, epochs, steps_per_epoch, save_path, valid=None, reload=False,
        checkpoint_monitor='val_acc', custom_callbacks=[], save_freq=0, verbose=1, validation_steps=None):
    """
    Params
    ------
    custom_callbacks: list
        List of pickable Callbacks
    """
    model_last_epoch_path = os.path.join(save_path, "model_last_epoch.h5")
    loop_state_path = os.path.join(save_path, "loop_state.pkl")
    loop_state = {'last_epoch_done_id': -1}
    epoch_start = 0
    callbacks = list(custom_callbacks)

    # check state
    history_path = os.path.join(save_path, "history.csv")
    if reload and os.path.exists(model_last_epoch_path) and os.path.exists(loop_state_path):
        logger.warning("Reloading weights!")
        model.load_weights(model_last_epoch_path)

        # code from https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/utils/save_load_utils.py
        with h5py.File(model_last_epoch_path) as f:
            if 'optimizer_weights' in f:
                # build train function (to get weight updates)
                model._make_train_function() # Note: might need call to model
                optimizer_weights_group = f['optimizer_weights']
                optimizer_weight_names = [n.decode('utf8') for n in optimizer_weights_group.attrs['weight_names']]
                logger.info(optimizer_weight_names)
                optimizer_weight_values = [optimizer_weights_group[n] for n in optimizer_weight_names]
                model.optimizer.set_weights(optimizer_weight_values)
            else:
                logger.warning("No optimizer weights in wieghts file!")

        logger.info("Reloading loop state!")
        loop_state = pickle.load(open(loop_state_path, 'rb'))

        H = pd.read_csv(history_path)
        H = {col: list(H[col].values) for col in H.columns}

        logger.info("Setting LR of optimizer to " + str(H['lr'][-1]))
        logger.info("If your optimizer has more things that change or decay -- please adapt")
        # h5 doesnt store lr, which is weird. It should. Perhaps if we were to add it to optimizer.weights?
        K.set_value(model.optimizer.lr, H['lr'][-1])

        # HACK: (TODO: Think how to do it nicely)
        os.system("cp " + os.path.join(save_path, "history.pkl") + " " + os.path.join(save_path, "history.pkl.bckp"))
        epoch_start = loop_state['last_epoch_done_id'] + 1
        logger.info("Starting from epoch " + str(epoch_start))
        def reload_pickled_history(epoch, logs):
            # wc -l history.csv 413 testreset/history.csv
            # 2018-09-12 06:59:44,283 - src.training_loop - INFO - reload_pickled_history(412)
            # This means history has 412 lines, and this is 413th epoch counted from 0
            logger.info("reload_pickled_history({})".format(epoch))
            if epoch == epoch_start:
                assert len(model.history.history) == 0
                logger.info("Loading pickled history")
                H_pickle = pickle.load(open(os.path.join(save_path, "history.pkl"), "rb"))
                setattr(model.history, "history", H_pickle)
                logger.info("Done loading pickled history")
                assert len(model.history.history) != 0
                for k in H:
                    assert len(H_pickle[k]) == len(H[k]), "Corrupted history files"
        # WARNING: After train_begin
        callbacks.insert(0, LambdaCallbackPickable(on_epoch_begin=reload_pickled_history))

        # load all callbacks from loop_state
        for e, e_loaded in zip(custom_callbacks, loop_state['callbacks']):
            assert type(e) == type(e_loaded)
            if hasattr(e, "__setstate__"):
                e.__setstate__(e_loaded.__dict__)
            else:
                e.__dict__.update(e_loaded.__dict__)
    else:
        logger.info("Removing " + history_path)
        os.system("rm " + history_path)
        H = {}
        model.save_weights(os.path.join(save_path, "init_weights.h5"))
        # A  bit ugly, but last_epoch is here -1 epoch
        model.save_weights(os.path.join(save_path, "model_last_epoch.h5"))

    callbacks = construct_training_loop_callbacks(callbacks, save_freq=save_freq,
        save_path=save_path, save_callbacks=custom_callbacks, model=model,
        model_last_epoch_path=model_last_epoch_path, checkpoint_monitor=checkpoint_monitor, H=H)

    # Note: there is a hack to control number of steps per epoch. It can be passed as a np.array.
    logger.info("Starting LR={}".format(K.get_value(model.optimizer.lr)))
    model.fit_generator(generator=train,
                        validation_steps=validation_steps,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        initial_epoch=epoch_start,
                        verbose=verbose,
                        validation_data=valid,
                        callbacks=callbacks)
