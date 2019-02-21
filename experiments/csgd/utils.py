# Living script, but script
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The main purpose is to report results of experiment checking different overshooting LR.

"""

from src.utils.plotting_preamble import *
from src import PROJECT_DIR, RESULTS_DIR

from src.utils.plotting_preamble import _construct_colorer

SAVE_DIR = os.path.join(PROJECT_DIR, "experiments", "csgd", "csgd")
os.system("mkdir -p " + SAVE_DIR)

import h5py

from src.utils.vegab import configure_logger
from bin.train_resnet_cifar import init_data_and_model as resnet_cifar_init_data_and_model
from bin.train_vgg_cifar import init_data_and_model as vgg_cifar_init_data_and_model
from bin.train_simple_cnn_cifar import init_data_and_model as simple_cnn_cifar_init_data_and_model
from src.lanczos import TopKEigenvaluesBatched
from os.path import *

import logging

_load_HC_csv = load_HC_csv

logger = logging.getLogger(__name__)


def eval_exp_call(E, checkpoint="best_val", check_if_done=False):

    try:
        J = json.load(open(os.path.join(E, "eval_results_{}.json".format(checkpoint))))
        if "{}_dist_init".format(checkpoint) not in J or '{}_top_K_e'.format(checkpoint) not in J:
            print("Corrupted json, no dist_init")
            corrupted = True
        else:
            corrupted=False
    except:
        print("Corrupted json or non-existent, cannot load")
        corrupted=True

    if corrupted or not check_if_done or not os.path.exists(os.path.join(E, "eval_results_{}.json".format(checkpoint))):
        print("Running evaluation for " + E)
        ret = os.system("python bin/evaluate/evaluate.py -p {} -L=50 -M=1 -e=1 --checkpoint={}".format(E, checkpoint))
        assert ret == 0

    print(os.path.exists(os.path.join(E, "eval_results_{}.json".format(checkpoint))))

    return json.load(open(os.path.join(E, "eval_results_{}.json".format(checkpoint))))


def load_D(Es, K_smooth=10, evaltype="", ignore_running=False):
    # Collect data
    N_start = 1  # Skips first entry
    D = []
    for E in Es:

        if ignore_running and not os.path.exists(os.path.join(E, "FINISHED")):
            print("Skipping " + E)
            continue

        row = {}
        try:
            H, C = _load_HC_csv(E)
        except:
            print("Failed " + E)
            continue

        try:
            HE = np.load(join(E, "lanczos.npz"))['top_K_e'][:]
        except:
            print("Faile loading Hessian for " + E)
            HE = None

        if len(H['acc']) < 50:
            print("N epochs below 50. Skipping")
            continue

        keys = ['acc', 'val_acc', 'loss']

        # TODO: Do we want smoothed curves?
        # H_smoothed = {k: list(np.convolve(H[k], [1 for _ in range(K_smooth)], mode="same") / K_smooth) for k in keys}
        H = {k: list(H[k]) for k in H}
        best_val_id = np.argmax(H['val_acc'])

        # Read in or compute evaluation
        if evaltype == "calc":
            H_eval = eval_exp_call(E, checkpoint="best_val", check_if_done=True)
            H_eval.update(eval_exp_call(E, checkpoint="last_epoch", check_if_done=True))
        elif evaltype == "calc_force":
            H_eval = eval_exp_call(E, checkpoint="best_val", check_if_done=False)
            H_eval.update(eval_exp_call(E, checkpoint="last_epoch", check_if_done=False))
        else:
            H_eval = None

        # Determine 'SN' and 'F' entries
        if H_eval is not None:
            row['test_acc'] = H_eval['best_val_test_acc']
            best_val_eigv = H_eval['best_val_top_K_e']
            print("Number of samples: " + str(len(best_val_eigv))) # Typicallly 1
            best_val_eigv = np.mean( H_eval['best_val_top_K_e'], axis=0)  # Average out
            best_val_FN = np.mean( H_eval['best_val_FN'], axis=0)  # Average out

            row['SN'] = best_val_eigv[0]
            # row['F'] =  np.sqrt(np.sum(np.abs(best_val_eigv) ** 2))
            row['F'] = best_val_FN

            last_epoch_eigv = H_eval['last_epoch_top_K_e']
            print("Number of samples: " + str(len(last_epoch_eigv)))
            last_epoch_eigv = np.mean(last_epoch_eigv, axis=0)  # Average out
            last_epoch_FN = np.mean(H_eval['last_epoch_FN'], axis=0)  # Average out
            row['final_SN'] = last_epoch_eigv[0]
            row['final_F'] = last_epoch_FN
            # row['final_F'] = np.sqrt(np.sum(np.abs(last_epoch_eigv) ** 2))

            # TODO: Refactor this code
            if HE is not None:
                HE = HE[N_start:].T
                HE = np.array([np.convolve(zz,
                    [1 for _ in range(K_smooth)], mode="same") for zz in HE]) / K_smooth
                HE = HE.T
                id_epoch = best_val_id + 1 - N_start
                if id_epoch >= len(HE):
                    eigv = HE[-1]
                else:
                    eigv = HE[id_epoch]
                F_all = np.sqrt(np.sum(np.abs(HE) ** 2, axis=1))
                SN_all = HE[:, 0]
                row['max_SN'] = max(eigv)
                row['max_F'] = max(F_all)
            else:
                row['max_SN'] = 0
                row['max_F'] = 0
        else:
            row['test_acc'] = max(H['val_acc'])
            if HE is not None:
                HE = HE[N_start:].T
                HE = np.array([np.convolve(zz,
                    [1 for _ in range(K_smooth)], mode="same") for zz in HE]) / K_smooth
                HE = HE.T
                id_epoch = best_val_id + 1 - N_start
                if id_epoch >= len(HE):
                    eigv = HE[-1]
                else:
                    eigv = HE[id_epoch]
                F_all = np.sqrt(np.sum(np.abs(HE) ** 2, axis=1))
                SN_all = HE[:, 0]
                F = np.sqrt(np.sum(np.abs(eigv) ** 2))
                SN = eigv[0]
                row['SN'] = SN
                row['F'] = F
                row['final_SN'] = np.mean(SN_all[-10:])
                row['final_F'] = np.mean(F_all[-10:])
                row['max_SN'] = max(eigv)
                row['max_F'] = max(F_all)
            else:
                row['SN'] = 0
                row['F'] = 0
                row['final_SN'] = 0
                row['final_F'] = 0
                row['max_SN'] = 0
                row['max_F'] = 0

        # Determine other entries
        if H_eval is not None:
            row['dist_init'] = H_eval['best_val_dist_init']
            row['SVD'] = np.sum([v for k, v in H_eval.items() if "best_val" in k and "SVD" in k])
        else:
            row['dist_init'] = 0
            row['SVD'] = 0

        row['opt_kwargs'] = C['opt_kwargs']
        row['seed'] = C['seed'] + 100 * C['data_seed']
        try:
            row['gamma'] = float(re.findall(r"overshoot=(\d.*)_", E)[0].split("_")[0])
        except:
            row['gamma'] = -1

        row['min_loss'] = min(H['loss'])

        # row['final_loss'] = np.mean(H['loss'][-10:])
        # Improved final loss
        if H_eval is not None:
            row['final_loss'] = H_eval['last_epoch_train_loss']
        else:
            print("Warning, using as the final loss augmentation")
            row['final_loss'] = np.mean(H['loss'][-10:])

        row['max_acc'] = max(H['acc'])
        if len(H['val_acc']) >= 100:
            row['val_acc_100'] = H['val_acc'][99]
        else:
            row['val_acc_100'] = -1
        if len(H['val_acc']) >= 50:
            row['val_acc_50'] = H['val_acc'][49]
        else:
            row['val_acc_50'] = -1
        if len(H['val_acc']) >= 10:
            row['val_acc_10'] = H['val_acc'][9]
        else:
            row['val_acc_10'] = -1
        row['name'] = os.path.basename(E)
        row['loss'] = H['loss'][best_val_id]
        row['acc'] = H['acc'][best_val_id]
        row['n_epochs'] = len(H['acc'])
        row['finished'] = os.path.exists(os.path.join(E, "FINISHED"))
        row.update(C)
        D.append(row)
    return D


def print_table(D, early_val_acc=50, sharpness="FN"):
    # For before eval
    D = pd.DataFrame(D)
    D = D.sort_values("gamma")
    D = D.groupby(["gamma", "lr"]).mean().reset_index()

    # Collect labels
    labels = []
    for idx, row in D.iterrows():
        if row['gamma'] != -1:
            labels.append("$\gamma="+str(row['gamma'])+"$")
        else:
            labels.append("SGD({})".format(row['lr']))
    D['label'] = labels

    #### CONFIGURE ####
    columns = ['label', sharpness, 'final_' + sharpness, 'test_acc', "val_acc_{}".format(early_val_acc), 'final_loss']
    columns_labels = ['name', '$||\mathbf{H}||_'+ (sharpness) +"$", 'final_' + sharpness, 'Test acc.', 'Val. acc. ({})'.format(early_val_acc), "Loss"]
    if "dist_init" in D:
        columns += ['dist_init']
        columns_labels += ["Dist."]#, '$\rho(W)$']

    ### COMPONSE #####
    D2 = D.copy()
    D2 = D2[columns]
    D2['test_acc'] *= 100
    if "val_acc_50" in D2:
        D2['val_acc_50'] *= 100
    if "val_acc_10" in D2:
        D2['val_acc_10'] *= 100
    D2.columns = columns_labels
    for column in ["Dist.", '$\rho(W)$']:  # , "$\rho(W)$"
        if column in D2:
            D2[column] = D2[column].map('${:,.2f}$'.format)
    for column in ['final_' + sharpness, '$||\mathbf{H}||_' + (sharpness) + "$"]:
        if column in D2:
            D2[column] = D2[column].map('${:,.2f}$'.format)
    D2['$||\mathbf{H}||_' + sharpness + "$"] = D2['$||\mathbf{H}||_'+ (sharpness) + "$"] + "/" + D2['final_' + sharpness]
    del D2['final_' + sharpness]
    for column in ["Loss"]:
        D2[column] = D2[column].map('${:,.5f}$'.format)
    for column in ["Test acc.", 'Val. acc. (50)']:
        if column in D2:
            D2[column] = D2[column].map('${:,.2f}\%$'.format)

    return(D2.to_latex(escape=False, index=False))


def plot_against_K(whichy, save_to, D, B1, B2):
    D = D.sort_values("lanczos_top_K")
    plt.figure(figsize=(fs, fs / 1.5))
    ax = plt.gca()

    ax.tick_params(labelsize=labelfontsize)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    cm = _construct_colorer(D['lanczos_top_K'])
    if whichy == "test_acc":
        ax.plot(D['lanczos_top_K'], 100 * D['test_acc'], label="NSGD(K)", linewidth=linewidth)
        ax.set_ylabel("Test accuracy", fontsize=fontsize)
        if len(B1) > 0 and len(B2) > 0:
            print(B1['test_acc'])
            print(B2['test_acc'])
            ax.axhline(100 * B1['test_acc'], linewidth=linewidth, linestyle="--", color="purple",
                label="SGD({})".format(B1['lr']))
            ax.axhline(100 * B2['test_acc'], linewidth=linewidth, linestyle="--", color="red",
                label="SGD({})".format(B2['lr']))
        ax.legend(fontsize=fontsize*0.8)
    elif whichy == "SN":
        axt = ax.twinx()
        axt.tick_params(labelsize=labelfontsize)
        ax.plot(D['lanczos_top_K'], D['F'], label="NSGD(K)", color="blue", linewidth=linewidth)
        axt.plot(D['lanczos_top_K'], D['SN'], label="__nolabel__", color="blue", linestyle="--", linewidth=linewidth)
        ax.set_ylabel("$\lambda_{max}$", fontsize=fontsize) #/$||H||_F$
        if len(B2) and "F" in B2:
            ax.axhline(B2['F'], linewidth=linewidth, color="purple", label="SGD({})".format(B2['lr']))
            ax.axhline(B2['SN'], linewidth=linewidth, color="purple", linestyle="--", label="__nolabel__")
        ax.legend(fontsize=fontsize*0.8)
    elif whichy == "max_SN":
        axt = ax.twinx()
        axt.tick_params(labelsize=labelfontsize)
        ax.plot(D['lanczos_top_K'], D['max_F'], label="NSGD(K)", color="blue", linewidth=linewidth)
        axt.plot(D['lanczos_top_K'], D['max_SN'], label="__nolabel__", color="blue", linestyle="--", linewidth=linewidth)
        ax.set_ylabel("$\lambda_{max}$", fontsize=fontsize)  # /$||H||_F$
        ax.legend(fontsize=fontsize*0.8)
    elif whichy == "both":
        axt = ax.twinx()
        axt.tick_params(labelsize=labelfontsize)
        ax.plot(D['lanczos_top_K'], 100 * D['test_acc'], marker='o', label="NSGD(K)", color="red", linewidth=linewidth)
        ax.set_ylabel("Test accuracy", fontsize=fontsize)
        if len(B2) > 0 and len(B2) > 0:
            ax.axhline(100 * B2['test_acc'], linewidth=linewidth, linestyle="--", color="red",
                label="SGD({})".format(B2['lr']))

        axt.plot(D['lanczos_top_K'], D['F'], marker='o', label="NSGD(K)", color="blue", linewidth=linewidth)
        # axt.plot(D['lanczos_top_K'], D['SN'], label="__nolabel__", color="blue", linestyle="--", linewidth=linewidth)
        axt.set_ylabel("$||H||_F$", fontsize=fontsize)  # /$||H||_F$
        ax.yaxis.label.set_color('red')
        axt.yaxis.label.set_color('blue')
        if len(B2) and "F" in B1:
            axt.axhline(B2['F'], linewidth=linewidth, color="blue", label="SGD({})".format(B2['lr']))
            # axt.axhline(B2['SN'], linewidth=linewidth, color="purple", linestyle="--", label="__nolabel__")
    else:
        raise NotImplementedError()
    # ax.plot(D['lanczos_top_K'], D['SN']) # test_acc

    ax.set_xlabel("K", fontsize=fontsize)
    save_fig(save_to)


def plot_curves(Es, expname, labels, nstart=1, maxn=800, smooth=1, cm=None, save_dir=SAVE_DIR, plot_frobenius_norm=True):
    for key in ['val_acc', 'acc']:
        fs = 7
        plt.figure(figsize=(fs, fs / 1.5))
        ax = plt.gca()
        #         axt = ax.twinx()
        ax.tick_params(labelsize=labelfontsize)

        for E, l in zip(Es, labels):
            # TODO: Uncomment
            if cm is None:
                color = next(ax._get_lines.prop_cycler)['color']
            else:
                color = cm(E)

            H, C = _load_HC_csv(E)


            if len(H) == 0 or len(H['acc']) < 50:
                print("Skipping " + E)
                continue

            try:
                if os.path.exists(join(E, "history.npz")):
                    HE = np.load(join(E, "history.npz"))
                else:
                    HE = np.load(join(E, "lanczos.npz"))
            except:
                print("Warning: no Hessian!")
                continue

            curve = H[key][nstart:maxn]
            curve = np.convolve(curve, [1] * smooth, mode="valid") / smooth
            # print(("Plotting", os.path.basename(E), len(curve), curve[-10:], color, l))
            ax.plot(curve, label=l, color=color, linestyle="--", linewidth=linewidth)
            ax.set_xlabel("Epoch", fontsize=fontsize)
            if key == "val_acc":
                ax.set_ylabel("Validation accuracy", fontsize=fontsize)
            else:
                ax.set_ylabel("Accuracy", fontsize=fontsize)
        ax.legend(fontsize=fontsize * 0.8)
        save_fig(os.path.join(save_dir, "counterexample_{}_{}.pdf".format(expname, key)))
        plt.show()
        plt.close()

    plt.figure(figsize=(fs, fs / 1.5))

    ax = plt.gca()
    axt = ax.twinx()
    ax.tick_params(labelsize=labelfontsize)
    ax.ticklabel_format(style='sci', axis='both')
    axt.tick_params(labelsize=labelfontsize)
    for E, l in zip(Es, labels):

        if cm is None:
            color = next(ax._get_lines.prop_cycler)['color']
        else:
            color = cm(E)

        try:
            H, C = _load_HC_csv(E)
        except:
            continue

        if len(H) == 0 or len(H['acc']) < 50:
            print("Skipping " + E)
            continue

        try:
            if os.path.exists(join(E, "history.npz")):
                HE = np.load(join(E, "history.npz"))
            else:
                HE = np.load(join(E, "lanczos.npz"))
        except:
            continue
        eigv = HE['top_K_e']

        ax.plot(eigv[1:maxn, 0], label=l, color=color, linewidth=linewidth)
        # ax.plot(np.sqrt((np.abs(eigv[1:maxn, :]) ** 2).sum(axis=1))[0:maxn],
        #     label=l, color=color, linewidth=linewidth, linestyle="--")
        if plot_frobenius_norm:
            axt.plot(np.sqrt((np.abs(eigv[1:maxn, :]) ** 2).sum(axis=1))[0:maxn],
                label=l, color=color, linewidth=linewidth, linestyle="--")
        ax.set_xlabel("Epoch", fontsize=fontsize)
        # ax.set_ylabel("$\lambda_{max}$ / $||H||_F$", fontsize=fontsize)
        ax.set_ylabel("$\lambda_{max}$", fontsize=fontsize)
        if plot_frobenius_norm:
            axt.set_ylabel("$||H||_F$", fontsize=fontsize)
        # axt.set_ylabel("$||H||_F$", fontsize=fontsize)
    # ax.set_yscale("log")
    #     axt.set_yscale("log")
    ax.legend(fontsize=fontsize * 0.8)
    save_fig(os.path.join(save_dir, "counterexample_{}_SN.pdf".format(expname)))
