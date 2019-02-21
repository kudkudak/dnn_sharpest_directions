#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run as python experiments/general/plot_general_shape.py
"""

from src.utils.plotting_preamble import *
from src import PROJECT_DIR, RESULTS_DIR

if __name__ == "__main__":
    # Config a bit
    SAVE_DIR=os.path.join(PROJECT_DIR, "experiments/general/general_shape")
    import os
    os.system("mkdir " + SAVE_DIR)
    print(SAVE_DIR)
    simple_cnn_dir=os.path.join(RESULTS_DIR, "general_shape")
    resnet_dir=os.path.join(RESULTS_DIR, "general_shape")

    # Plot
    N_start = 1
    N = 180
    N_whole=180

    Es = []
    Es.append(os.path.join(resnet_dir,"resnet32_constant_l2_lr=0.1"))
    Es.append(os.path.join(resnet_dir, "resnet32_zoom_constant_l2_lr=0.1"))
    Es.append(os.path.join(simple_cnn_dir,"simplecnn_constant_lr=0.1"))
    Es.append(os.path.join(simple_cnn_dir, "simplcnn_zoom_constant_lr=0.1"))

    labels = []
    labels.append("resnet32_l2")
    labels.append("zoom_resnet32_l2")
    labels.append("simplecnn")
    labels.append("zoom_simplcnn")

    names = []
    names.append("Resnet-32")
    names.append("Resnet-32 (zoom)")
    names.append("SimpleCNN")
    names.append("SimpleCNN (zoom)")

    ranges = []
    ranges.append([0, N_whole])
    # ranges.append([0.25, 0.5])
    ranges.append([0.0, 0.5])
    # ranges.append([0, 0.25])
    ranges.append([0, N_whole])
    ranges.append([0, 0.5])

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
            ax.xaxis.set_major_locator(ticker.MultipleLocator(50))

        ax.plot(xaxis[ids], H[key_acc], linewidth=linewidth, label="Accuracy")
        ax.plot(xaxis[ids], H['val_acc'], linewidth=linewidth, linestyle="--", label="Validation accuracy")

        ax.set_xlabel("Epoch", fontsize=fontsize)
        ax.set_ylabel("Accuracy", fontsize=fontsize)
        ax.set_yticklabels(["{0:.0f} %".format(100 * tick) for tick in ax.get_yticks()], fontsize=fontsize)

        ax.legend(fontsize=labelfontsize)
        plt.title(name, fontsize=fontsize)
        plt.savefig(os.path.join(SAVE_DIR, "evolution_{}_acc.pdf".format(label)), bbox_inches='tight',
            transparent=True,
            pad_inches=0)
        plt.show()


    for E, label, name, (zoom_start, zoom_end) in zip(Es, labels, names, ranges):
        f = plt.figure(figsize=(fs, fs / 1.7))
        ax = plt.gca()
        import matplotlib.ticker as ticker

        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
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
            ax.xaxis.set_major_locator(ticker.MultipleLocator(50))

        print(xaxis)

        cm = plt.cm.get_cmap("coolwarm_r", eigv.shape[1])
        for i in range(0, eigv.shape[1]):
            ax.plot(xaxis, eigv[:, i], color=cm(i / float(eigv.shape[1])),
                linewidth=linewidth)

        ax.set_xlabel("Epoch", fontsize=fontsize)
        ax.set_ylabel("$\lambda$", fontsize=fontsize)
        ax.set_yticklabels(["{0:.0f}".format(tick) for tick in ax.get_yticks()], fontsize=fontsize)

        plt.title(name, fontsize=fontsize)
        plt.savefig(os.path.join(SAVE_DIR, "spectrum_{}_acc.pdf".format(label)), bbox_inches='tight',
            transparent=True,
            pad_inches=0)
        plt.show()