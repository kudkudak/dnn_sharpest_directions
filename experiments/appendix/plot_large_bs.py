#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run as python experiments/appendix/plot_large_bs.py
"""

from src.utils.plotting_preamble import *
from src import PROJECT_DIR, RESULTS_DIR

if __name__ == "__main__":
    # Config a bit
    SAVE_DIR=os.path.join(PROJECT_DIR, "experiments/appendix/large_bs")
    import os
    os.system("mkdir " + SAVE_DIR)
    print(SAVE_DIR)
    bn_dir=os.path.join(RESULTS_DIR, "large_bs")
    nobn_dir=os.path.join(RESULTS_DIR, "large_bs")

    # Plot
    N_start = 1
    N = 100
    N_whole= 100

    Es = []
    Es.append(os.path.join(bn_dir,"nobn"))
    Es.append(os.path.join(bn_dir, "nobn_lr=0.05"))
    Es.append(os.path.join(nobn_dir,"bn"))
    Es.append(os.path.join(nobn_dir, "bn_lr=0.05"))

    labels = []
    labels.append("nobn")
    labels.append("nobn_lr=0.05")
    labels.append("bn")
    labels.append("bn_lr=0.05")

    names = []
    names.append("Resnet-32 $\eta$=0.01")
    names.append("Resnet-32 $\eta$=0.05")
    names.append("Resnet-BN-32 $\eta$=0.01")
    names.append("Resnet-BN-32 $\eta$=0.05")

    ranges = []
    ranges.append([0, N_whole])
    ranges.append([0, N_whole])
    ranges.append([0, N_whole])
    ranges.append([0, N_whole])

    ## Learning curves for the appendix

    for E, label, name, (zoom_start, zoom_end) in zip(Es, labels, names, ranges):
        f = plt.figure(figsize=(fs, fs / 1.7))

        ax = plt.gca()
        import matplotlib.ticker as ticker

        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.tick_params(labelsize=fontsize)

        H, C = load_HC_csv(E)

        # Select range
        if "zoom" in label:
            xaxis = [C['epoch_size'] * i / 45000. for i in range(len(H['acc']))]
        else:
            xaxis = range(len(H['acc']))
        xaxis = np.array(xaxis)
        ids = np.where((xaxis >= zoom_start) & (xaxis <= zoom_end))
        H = {k: np.array(v)[ids] for k, v in H.iteritems()}
        print((C['lr'], C['batch_size']))

        if "zoom" in label:
            key_acc = "train_acc" if "train_acc" in H else "acc"
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        else:
            key_acc = "acc"
            ax.xaxis.set_major_locator(ticker.MultipleLocator(20))

        ax.plot(xaxis[ids], H[key_acc], linewidth=linewidth, label="Accuracy")
        ax.plot(xaxis[ids], H['val_acc'], linewidth=linewidth, linestyle="--", label="Validation accuracy")

        ax.set_xlabel("Epoch", fontsize=fontsize)
        ax.set_ylabel("Accuracy", fontsize=fontsize)
        ax.set_yticklabels(["{0:.0f} %".format(100 * tick) for tick in ax.get_yticks()], fontsize=fontsize)

        ax.legend(fontsize=int(0.7*labelfontsize))
        plt.title(name, fontsize=fontsize)
        plt.savefig(os.path.join(SAVE_DIR, "evolution_{}_acc".format(label).replace("=", "_").replace(".", "_")) + ".pdf", bbox_inches='tight',
            transparent=True,
            pad_inches=0)
        plt.show()


    for E, label, name, (zoom_start, zoom_end) in zip(Es, labels, names, ranges):
        f = plt.figure(figsize=(fs, fs / 1.7))
        ax = plt.gca()
        import matplotlib.ticker as ticker

        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.tick_params(labelsize=fontsize)

        H, C = load_HC(E)

        try:
            HE = np.load(join(E, "lanczos.npz"))
        except:
            HE = np.load(join(E, "history.npz"))

        # Select range
        if "zoom" in label:
            xaxis = [C['epoch_size'] * i / 45000. for i in range(len(H['acc']))]
        else:
            xaxis = range(len(H['acc']))
        xaxis = np.array(xaxis)
        ids = np.where((xaxis >= zoom_start) & (xaxis <= zoom_end))
        xaxis = xaxis[ids]
        H = {k: np.array(v)[ids] for k, v in H.iteritems()}
        eigv = HE['top_K_e']
        eigv = eigv[N_start:][ids]

        if "zoom" in label:
            #         xaxis = np.array([C['epoch_size']*i/45000. for i in range(len(eigv))][:])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        else:
            #         xaxis = np.array(range(len(eigv))[:])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(20))

        print(xaxis)

        cm = plt.cm.get_cmap("coolwarm_r", eigv.shape[1])
        for i in range(0, eigv.shape[1]):
            ax.plot(xaxis, eigv[:, i], color=cm(i / float(eigv.shape[1])),
                linewidth=linewidth)

        ax.set_xlabel("Epoch", fontsize=fontsize)
        ax.set_ylabel("$\lambda$", fontsize=fontsize)
        ax.set_yticklabels(["{0:.0f}".format(tick) for tick in ax.get_yticks()], fontsize=fontsize)

        plt.title(name, fontsize=fontsize)
        plt.savefig(os.path.join(SAVE_DIR, "spectrum_{}_acc".format(label).replace("=", "_").replace(".", "_")) + ".pdf", bbox_inches='tight',
            transparent=True,
            pad_inches=0)
        plt.show()