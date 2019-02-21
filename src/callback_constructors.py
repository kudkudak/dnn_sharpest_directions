"""
Some callbacks are tricky and sometimes mundane to create, so here are shared callback constructors
"""

import matplotlib

matplotlib.use('Agg')

from keras.utils import np_utils
from functools import partial

from src.optimizers import NSGD
from src.callbacks import *
from src.callbacks_analysis import CrossSectionLossSurfacePlotter
from src.lanczos import TopKEigenvaluesBatched
from src import DATA_FORMAT

import keras.backend as K

logger = logging.getLogger(__name__)


def _save_history(epoch, logs, save_path, H):
    logger.info("Saving history.csv")  # TODO: Debug?

    for key, value in logs.items():

        if key == "batch" or key == "examples_seen" or key == "batch_size":
            continue

        if isinstance(value, (int, float, complex, np.float32, np.float64)):
            if len(H.get(key, [])) != epoch:
                logger.warning("Skipping in history " + key)
                continue

            if key not in H:
                H[key] = [value]
            else:
                H[key].append(value)

            assert len(H[key]) == epoch + 1, "Len {} = ".format(key) + str(len(H[key]))
        else:
            pass

    pd.DataFrame(H).to_csv(os.path.join(save_path, "history.csv"), index=False)
    logger.info("Saved history.csv")  # TODO: Debug?


def _save_loop_state(epoch, logs, save_path, save_callbacks):
    logger.info("Saving loop_state.pkl")  # TODO: Debug?

    loop_state = {"last_epoch_done_id": epoch, "callbacks": save_callbacks}

    ## Small hack to pickle Callbacks in keras ##
    if len(save_callbacks):
        m, vd = save_callbacks[0].model, save_callbacks[0].validation_data
        for c in save_callbacks:
            c.model = None
            c.validation_data = None

    pickle.dump(loop_state, open(os.path.join(save_path, "loop_state.pkl"), "wb"))

    ## Revert hack ##
    if len(save_callbacks):
        for c in save_callbacks:
            c.model = m
            c.validation_data = vd

    logger.info("Saved loop_state.pkl")  # TODO: Debug?


def construct_training_loop_callbacks(callbacks,save_freq, model, save_path, save_callbacks, model_last_epoch_path,
        checkpoint_monitor, H):
    if save_freq > 0:
        def save_weights(epoch, logs):
            if epoch % save_freq == 0:
                logger.info("Saving model from epoch " + str(epoch))
                model.save(os.path.join(save_path, "model_epoch_{}.h5".format(epoch)))

        callbacks.append(LambdaCallback(on_epoch_end=save_weights))

    # Always save from first epoch
    def save_weights(epoch, logs):
        if epoch == 0:
            logger.info("Saving model from epoch " + str(epoch))
            model.save(os.path.join(save_path, "model_epoch_{}.h5".format(epoch)))

    callbacks.append(LambdaCallback(on_epoch_end=save_weights))

    # Warning - this is delayed by one.
    def _pickle_whole_history(epoch, logs):
        logger.info("Pickling all")
        for key in model.history.history:
            if key == "batch" or key == "examples_seen" or key == "batch_size":
                continue
            if len(model.history.history[key]) != epoch:
                logger.warning("Failed {} {} {}".format(key, epoch, len(model.history.history[key])))
        pickle.dump(model.history.history, open(os.path.join(save_path, "history.pkl"), "wb"))
        logger.info("Done pickling all")

    def _pickle_whole_history_end(*args, **kwargs):
        logger.info("Pickling all (end of training)")
        pickle.dump(model.history.history, open(os.path.join(save_path, "history.pkl"), "wb"))
        logger.info("Done pickling all")


    # NOTE: On epoch_begin because History is last callback in Keras
    # Might lead to very occassional corruption
    callbacks.insert(1, LambdaCallbackPickable(on_epoch_begin=_pickle_whole_history, on_train_end=_pickle_whole_history_end))
    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_loop_state, save_path=save_path,
        save_callbacks=save_callbacks)))
    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_history, save_path=save_path, H=H)))
    # callbacks.append(TensorBoard(log_dir=save_path))
    callbacks.append(DumpTensorflowSummaries(save_path=save_path))
    callbacks.append(ModelCheckpoint(monitor=checkpoint_monitor,
        save_weights_only=False,
        save_best_only=True,
        mode='max',
        filepath=os.path.join(save_path, "model_best_val.h5")))
    callbacks.append(ModelCheckpoint(save_weights_only=False,
        filepath=model_last_epoch_path))

    return callbacks


import tensorflow as tf


def create_lr_schedule(config, model, save_path):
    learning_rate_schedule = eval(config['lr_schedule'])
    learning_rate_schedule_type = config.get("lr_schedule_type", "list_of_lists")
    logging.info("Learning rate schedule")
    logging.info(learning_rate_schedule)
    logging.info("Learning rate schedule type")
    logging.info(learning_rate_schedule_type)

    if learning_rate_schedule_type == "list_of_lists":
        def lr_schedule(epoch, logs):
            if len(learning_rate_schedule):
                for e, v in learning_rate_schedule:
                    if epoch < e:
                        break
                    elif epoch == e:
                        # logger.info("Saving model from epoch " + str(epoch))
                        # model.save_weights(os.path.join(save_path, "model_epoch_{}.h5".format(epoch)))
                        # This means that next (e,v) will necessarily drop LR so:
                        if config.get('reload_before_drop', False):
                            logger.info("Loading best validation")
                            model.load_weights(os.path.join(save_path, "model_best_val.h5"))

                setattr(model.optimizer, "base_lr", v)
                if isinstance(model.optimizer.lr, tf.Tensor):
                    K.get_session().run(model.optimizer.lr.assign(v))
                else:
                    K.set_value(model.optimizer.lr, v)
                logger.info("Set learning rate to {}".format(v))

        return LambdaCallbackPickable(on_epoch_begin=lr_schedule, on_epoch_end=lr_schedule)
    elif learning_rate_schedule_type == "epoch_mapping":
        def lr_schedule(epoch, logs):
            logger.info("Treatins chedule as mapping")
            v = learning_rate_schedule(epoch)
            setattr(model.optimizer, "base_lr", v)
            K.set_value(model.optimizer.lr, v)
            if config['reload_before_drop']:
                raise NotImplementedError()

            logger.info("Set learning rate to {}".format(v))

        return LambdaCallbackPickable(on_epoch_begin=lr_schedule, on_epoch_end=lr_schedule)
    elif learning_rate_schedule_type == "batch_mapping":
        # TODO: Create a proper class out of it..
        cls = LambdaCallbackPickable()
        cls.set_callback_state({"batch_id": np.array([0])})

        def lr_schedule(batch, logs):
            batch_id = cls.callback_state['batch_id']
            v = learning_rate_schedule(batch_id[0])
            setattr(model.optimizer, "base_lr", v)
            K.set_value(model.optimizer.lr, v)
            if config['reload_before_drop']:
                raise NotImplementedError()
            batch_id[0] += 1

        cls.on_batch_begin = lr_schedule
        return cls
    elif learning_rate_schedule_type == "eval":
        return learning_rate_schedule
    else:
        raise NotImplementedError("Not implemented learning rate schedule")


def add_random_labels_evaluation(config, model, x_train, y_train, ids_random_train):
    callbacks = []
    logger.info("Will measure loss on clean and randomized subsets of train")
    logger.info(ids_random_train.shape)
    ids_random_train = set(ids_random_train.reshape(-1, ))
    is_random = np.array([(i in ids_random_train) for i in range(len(x_train))])

    def evaluate_on_train(epoch, logs=None):
        metrics_tensors = model.evaluate(x_train[is_random], y_train[is_random], batch_size=config['batch_size'])
        for m_name, m_val in zip(model.metrics_names, metrics_tensors):
            logs['train_random_' + m_name] = m_val
        metrics_tensors = model.evaluate(x_train[~is_random], y_train[~is_random], batch_size=config['batch_size'])
        for m_name, m_val in zip(model.metrics_names, metrics_tensors):
            logs['train_clean_' + m_name] = m_val

    callbacks.append(LambdaCallbackPickable(on_epoch_end=evaluate_on_train))
    return callbacks


def create_variable_schedule(config, model, n_epoch):
    # Allows for easy change for any of the "steerable_variable". Executed every epoch
    variable_schedule = eval(config['variable_schedule'])

    def schedule_variables(call_id, example_id, logs):
        epoch_id = example_id / n_epoch
        logging.info("Start schedule on epoch " + str(epoch_id))
        for variable in variable_schedule:
            for e, v in variable_schedule[variable]:
                if epoch_id < e:
                    logging.info("Setting value of {} to {}".format(variable, v))
                    if variable not in model.steerable_variables:
                        raise RuntimeError("Didn't find variable " + str(variable))
                    if isinstance(model.steerable_variables[variable], np.ndarray):
                        logging.info("Setting as np.array")
                        logging.info(variable_schedule[variable])
                        model.steerable_variables[variable][:] = v
                    else:
                        if isinstance(model.steerable_variables[variable], list):
                            logger.info("Setting as list of variables")
                            for k in model.steerable_variables[variable]:
                                K.set_value(k, v)
                        else:
                            K.set_value(model.steerable_variables[variable], v)
                    break

    return LambdaCallbackPickableEveryKExamples(on_k_examples=schedule_variables, call_on_batch_0=True, k=n_epoch)


def config_alex(optimizer, model, top_eigenvalues_clbk, config):
    callbacks = []
    if isinstance(optimizer, NSGD):
        opt_kwargs = eval(config['opt_kwargs'])
        KK = opt_kwargs.get("KK", 1)
        burnin = opt_kwargs.get("burnin", 0)
        logger.info([np.prod(K.int_shape(p)) for p in model.trainable_weights])
        projections = {
            p.name: K.variable(np.zeros(shape=(np.prod(K.int_shape(p)), KK)).astype("float32")) \
            for p in model.trainable_weights}
        optimizer.set_projections(projections)

        if isinstance(optimizer, NSGD):
            optimizer.set_model(model)
            model.metrics_names.append("alex_alpha_sum")
            model.metrics_names.append("alex_alpha_norm_sum")
            model.metrics_names.append("alex_alpha_0")
            model.metrics_names.append("alex_alpha_0_norm")
            model.metrics_tensors.append(K.sum(K.abs(K.reshape(optimizer.alpha, (-1,)))))
            model.metrics_tensors.append(K.sum(K.abs(K.reshape(optimizer.alpha_normalized, (-1,)))))
            model.metrics_tensors.append(K.reshape(optimizer.alpha, (-1,))[0])
            model.metrics_tensors.append(K.reshape(optimizer.alpha_normalized, (-1,))[0])
            model.metrics_names.append("alex_v_alpha_sum")
            model.metrics_names.append("alex_v_alpha_norm_sum")
            model.metrics_names.append("alex_v_alpha_0")
            model.metrics_names.append("alex_v_alpha_0_norm")
            model.metrics_tensors.append(K.sum(K.abs(K.reshape(optimizer.alpha_v, (-1,)))))
            model.metrics_tensors.append(K.sum(K.abs(K.reshape(optimizer.alpha_v_normalized, (-1,)))))
            model.metrics_tensors.append(K.reshape(optimizer.alpha_v, (-1,))[0])
            model.metrics_tensors.append(K.reshape(optimizer.alpha_v_normalized, (-1,))[0])

        # TODO: Rewrite into checker every batch
        # Callback setting subspace for callback computing distance..
        def set_projections_clbk(epoch, logs):
            logger.info("Adding projections to Alex")
            eigv = top_eigenvalues_clbk._this_epoch_ev.copy()

            ps = top_eigenvalues_clbk.parameters
            p_names = top_eigenvalues_clbk.parameter_names
            if all([p.startswith("inference") for p in p_names]):
                p_names = [p_name.replace("inference_", "") for p_name in p_names]
            shapes = top_eigenvalues_clbk.parameter_shapes
            assert len(p_names) == len(shapes) == len(ps)

            projections = optimizer.projections

            cur = 0
            setters = []
            if config.get("alex_random", False):
                logger.warning("Using random RV")
                rv = np.random.uniform(-1,1,size=eigv[:, 0:KK].shape)
                rv = rv/np.linalg.norm(rv)
            else:
                rv = None
            for p, p_name, s in zip(ps, p_names, shapes):
                if config.get("alex_random", False):
                    eigv_param = rv[cur:(cur + np.prod(s)), 0:KK].copy()  # Shape [N, K]
                else:
                    eigv_param = eigv[cur:(cur + np.prod(s)), 0:KK].copy()  # Shape [N, K]

                if epoch < burnin:
                    logger.info("Burnin")
                    eigv_param = np.zeros_like(eigv_param)

                logger.info(eigv_param.shape)

                if p_name in projections:
                    setters.append((projections[p_name], eigv_param))
                else:
                    logger.warning("Failed to add proj for " + p_name)
                    raise RuntimeError()
                cur += np.prod(s)

            K.batch_set_value(setters)

        callbacks.append(LambdaCallbackPickable(on_epoch_begin=set_projections_clbk))

    return callbacks


def add_lanczos_callbacks(model, save_path, model_inference, meta_data, config, prefix="",
        lanczos_data=None, lanczos_data_steps=None, n_classes=10, callbacks=None, which="hessian"):
    """
    Adds callbacks computing top K eigenvalues (can also compute eigenvectors) using
    routines from TF.
    """
    if callbacks is None:
        callbacks = []

    assert lanczos_data is None, "Not supported"

    # Define data
    if not config.get("lanczos_aug", False):
        logger.info("Not using augmentation in Lanczos computation")
        lanczos_data = [meta_data['x_train'][0:config['lanczos_top_K_N']],
            np_utils.to_categorical(meta_data['y_train'], n_classes)[:config['lanczos_top_K_N']]]
    else:
        logger.info("Using augmentation in Lanczs computation")
        if config['lanczos_top_K_N_sample'] > 0:
            logger.info("Sampling from an independent ")
            lanczos_data = meta_data['train_2']
        else:
            logger.info("Sampling examples from the trainingset")
            if config['lanczos_top_K_N'] < len(meta_data['x_train']):
                logger.info("Sampling")
                L = config['lanczos_top_K_N']
                L_sampled = 0
                d = {"x_train": [], "y_train": []}
                while L_sampled < L:
                    x, y = next(meta_data['train_2'])
                    L_sampled += len(x)
                    d['x_train'].append(x)
                    d['y_train'].append(y)
                logger.info("Sampled {}".format(len(d['x_train'])))
                lanczos_data = [np.concatenate(d['x_train']), np.argmax(np.concatenate(d['y_train']), axis=1)]
                logger.info("Sampled total {}".format(len(lanczos_data[0])))

    if config['lanczos_top_K'] == -1 and config['lanczos_fullgrad_approx']:
        if 'lanczos_top_K_bs' in config:
            batch_size = config['lanczos_top_K_bs']
        else:
            batch_size = None

        # Create callback
        top_eigenvalues_clbk = FullBatchGradApprox(
            data=lanczos_data,
            sample_N=config.get("lanczos_top_K_N_sample", -1),
            save_path=save_path,
            K=config['lanczos_top_K'], batch_size=batch_size)

        if config.get("lanczos_inference_mode", True):
            top_eigenvalues_clbk.set_my_model(model_inference)  # To avoid using BN/Dropout
        else:
            top_eigenvalues_clbk.set_my_model(model)
        callbacks.append(top_eigenvalues_clbk)

        # Warning: fails if training is interrupted!!
        def save_evolution(logs):
            H = model.history.history  # We should be able to just pickle history?

            d = {}
            if prefix + "top_K_e" in H:
                d[prefix + 'top_K_e'] = np.array(H[prefix + 'top_K_e'])
            else:
                logger.warning("No top_K_e in history. Might be bad if epoch!=0")

            # Not sure what this is
            for k in H:
                if "delta_subspace" in k:
                    d[k] = np.array(H[k])
            np.savez(join(save_path, prefix + "lanczos.npz"), **d)

        callbacks.append(LambdaCallbackPickable(on_epoch_end=lambda epoch, logs: save_evolution(logs)))
    elif config['lanczos_top_K'] > 0:
        # Callback loading weights to model_inference
        def load_w(*args, **kwargs):
            logger.info("loading weights to inference model")
            if os.path.exists(os.path.join(save_path, "model_last_epoch.h5")):
                model_inference.load_weights(os.path.join(save_path, "model_last_epoch.h5"))
            else:
                logger.warning("No model_last_epoch!")
                model_inference.load_weights(os.path.join(save_path, "init_weights.h5"))

        if model != model_inference:
            callbacks.append(LambdaCallbackPickable(on_epoch_begin=load_w, on_train_begin=load_w))

        # This is a bit of sorcery
        if 'lanczos_top_K_bs' in config:
            batch_size = config['lanczos_top_K_bs']
        else:
            batch_size = None

        # Create callback
        kwargs = eval(config.get('lanczos_kwargs', {}))
        logger.info(kwargs)
        logger.info(type(kwargs))
        top_eigenvalues_clbk = TopKEigenvaluesBatched(
            data=lanczos_data,
            sample_N=config.get("lanczos_top_K_N_sample", -1),
            data_steps=lanczos_data_steps,
            which=which,
            prefix=prefix,
            n_classes=n_classes,
            save_eigenv=config['save_eigendirections'], save_path=save_path,
            K=config['lanczos_top_K'], batch_size=batch_size, **kwargs)

        if config.get("lanczos_inference_mode", True):
            top_eigenvalues_clbk.set_my_model(model_inference)  # To avoid using BN/Dropout
        else:
            top_eigenvalues_clbk.set_my_model(model)
        callbacks.append(top_eigenvalues_clbk)

        # Save history as matrix.
        # Warning: fails if training is interrupted!!
        def save_evolution(logs):
            H = model.history.history  # We should be able to just pickle history?

            d = {}

            if prefix + "top_K_e" in H:
                d[prefix + 'top_K_e'] = np.array(H[prefix + 'top_K_e'])
            else:
                logger.warning("No top_K_e in history. Might be bad if epoch!=0")

            # Not sure what this is
            for k in H:
                if "delta_subspace" in k:
                    d[k] = np.array(H[k])

            np.savez(join(save_path, prefix + "lanczos.npz"), **d)

        callbacks.append(LambdaCallbackPickable(on_epoch_end=lambda epoch, logs: save_evolution(logs)))
    else:
        top_eigenvalues_clbk = None

    return callbacks, top_eigenvalues_clbk


def add_eigenloss_callback(save_path, top_eigenvalues_clbk, meta_data, config, model_inference,
        n_classes=10):
    callbacks = []
    if config['eigen_loss']:
        cross_section_clbk = CrossSectionLossSurfacePlotter(save_path=save_path,
            training_mode=config.get("eigen_loss_training_mode", False),
            scaling=config.get("eigen_loss_scaling", "by_grad_norm_lr"),
            X=meta_data['x_train'][0:config['eigen_loss_N']], batch_size=config['batch_size'],
            y=np_utils.to_categorical(meta_data['y_train'][0:config['eigen_loss_N']], n_classes),
            ids=eval(config['eigen_loss_ids'])
        )

        cross_section_clbk.set_my_model(model_inference)

        def set_mapping_ev(epoch, logs):
            cross_section_clbk.set_mapping({"shapes": top_eigenvalues_clbk.parameter_shapes,
                "names": top_eigenvalues_clbk.parameter_names})
            cross_section_clbk.set_directions(top_eigenvalues_clbk._this_epoch_ev)

        # On epoch begin so aligned
        callbacks.append(LambdaCallbackPickable(on_epoch_begin=set_mapping_ev))
        callbacks.append(cross_section_clbk)
    return callbacks

def divide_every_k_schedule(n_epochs, freq, lr0, mult):
    schedule = [[freq*(i+1), lr0/(mult**i)] for i in range(n_epochs / freq)]
    schedule += [[n_epochs, schedule[-1][1]]] # Finish using same lr
    logger.info(schedule)
    return schedule

def add_common_callbacks(model, save_path, model_inference=None, meta_data=None, config=None, train=None,
        n_classes=10):
    """
    Helper function for adding simple callbacks shared between scripts.

    Notes
    -----
    Supposed to work for any model
    Assumes meta_data['y_train'] is not one-hot encoded
    """

    callbacks = []

    if hasattr(model.optimizer, "lr"):
        model.metrics_names.append("current_lr")
        model.metrics_tensors.append(model.optimizer.lr)

    # Save logs from btaches (important! keep it the last!)
    batch_history_clbk = StoreBatchLogs(save_path=os.path.join(save_path, "history_batch.csv"),
        frequency=1000)  # 50x each epoch. TOOD: Think how to make it faster
    callbacks.append(batch_history_clbk)

    # Add dynamic bs to logs. Important for some callbacks, unfortunately
    def add_dynamic_bs(epoch, logs):
        logs['batch_size'] = meta_data['batch_size_np'][0]

    callbacks.append(LambdaCallbackPickable(on_batch_begin=add_dynamic_bs))

    # Travelled distance
    callbacks.append(DistanceTravelled(os.path.join(save_path, "init_weights.h5")))

    ### BS/LR schedule ###
    def add_base_lr(epoch, logs):
        setattr(model.optimizer, "base_lr", config['lr'])

    callbacks.append(LambdaCallbackPickable(on_epoch_begin=add_base_lr))

    if len(config.get('lr_schedule', '')):
        callbacks.append(create_lr_schedule(config=config, model=model, save_path=save_path))

    if len(config.get("variable_schedule", "")):
        callbacks.append(create_variable_schedule(config=config, model=model, n_epoch=len(meta_data['x_train'])))

    callbacks.append(LambdaCallbackPickable(on_epoch_end=lambda e, l: logging.info(save_path)))
    callbacks.append(MeasureTime())
    #
    # if K.ndim(model.outputs[0]) == 2:
    #     # Dead neurons metric
    #     tensors = []
    #     names = []
    #     for l in model.layers:
    #         if "act" in l.name or "softmax" in l.name:
    #             i = 0
    #             while True:
    #                 try:
    #                     tensors.append(l.get_output_at(i))
    #                 except ValueError:
    #                     break
    #                 names.append(l.name + "_" + str(i))
    #                 i += 1
    #     add_dead_neurons_metric_tensors(model, tensors, names, data_format=DATA_FORMAT)

    # Add train evaluation
    if train is not None and config.get('measure_train_loss', False):
        logger.info("Will measure loss on train")

        def evaluate_on_train(epoch, logs=None):
            metrics_tensors = model.evaluate(train[0], train[1], batch_size=config['batch_size'])
            for m_name, m_val in zip(model.metrics_names, metrics_tensors):
                logs['train_' + m_name] = m_val

        callbacks.append(LambdaCallbackPickable(on_epoch_end=evaluate_on_train))

    # Add logs from batches to main history
    def add_to_main_history(e, l):
        if hasattr(model, "history_batch"):
            for k in model.history_batch:
                l[k] = model.history_batch[k][-1]
        else:
            logging.warning("No history_batch in model")

    add_batch_history_to_main_clbk = LambdaCallbackPickable(on_epoch_end=add_to_main_history)
    callbacks.append(add_batch_history_to_main_clbk)

    # Print logs (for verbose = 0)
    def print_logs(epoch, logs):
        printout = ""
        for k in logs:
            if isinstance(logs[k], (int, float, complex, np.float32, np.float64)):
                printout += "{}={}\t".format(k, logs[k])
        logging.info(printout)

    callbacks.append(LambdaCallbackPickable(on_epoch_end=print_logs))

    def gc_collect(epoch, logs):
        import gc
        gc.collect()

    callbacks.append(LambdaCallbackPickable(on_epoch_end=gc_collect))

    def start_epoch_printout(epoch, logs):
        logger.info("Start epoch {}".format(epoch))

    def finished_epoch_printout(epoch, logs):
        logger.info("Finished epoch {}".format(epoch))

    callbacks.append(LambdaCallbackPickable(on_epoch_begin=start_epoch_printout, on_epoch_end=finished_epoch_printout))

    if config.get("early_stopping", False):
        callbacks.append(EarlyStopping(monitor="val_acc", patience=config.get("early_stopping_patience", 100)))

    return callbacks
