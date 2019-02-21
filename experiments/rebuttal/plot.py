#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run as:

python experiments/rebuttal/plot.py adam/init/imdb
python experiments/rebuttal/plot.py imdb
"""

from src.utils.plotting_preamble import *
from src import PROJECT_DIR, RESULTS_DIR

import logging
logger = logging.getLogger(__name__)

SAVE_DIR=os.path.join(PROJECT_DIR, "experiments", "rebuttal", "rebuttal_31")
os.system("mkdir -p " + SAVE_DIR)

def plot(Es, lrs, save_prefix, title, which="lr", ymin=None,
        showxlabel=False, showylabel=False,
        ymax=None, K=10,  n_epochs=1000000, K_smooth=1):
    plt.figure(figsize=(fs, fs / 1.7))
    plt.title(title, fontsize=fontsize)

    cm = lambda lr: ["red", "green", "blue"][lrs.index(lr)]

    ax = plt.gca()
    ax.tick_params(labelsize=labelfontsize)
    for E in Es:
        try:
            H, C = load_HC_csv(E)
        except:
            logger.error("Failed loading " + str(E))
            continue

        if which == "lr":
            if C['lr'] not in lrs:
                continue
            key = C['lr']
            label = "$\eta={}$".format(C['lr'])
        elif which == "bs":
            if C['batch_size'] not in lrs:
                continue
            key = C['batch_size']
            label = "$S={}$".format(key)
        elif which == "init":
            key = C['init']
            label = C['init']
        else:
            raise NotImplementedError()

        if len(H) == 0:
            logger.error("Failed loading (len=0)" + str(E))
            continue

        try:
            if os.path.exists(join(E, "history.npz")):
                HE = np.load(join(E, "history.npz"))['top_K_e']
            else:
                HE = np.load(join(E, "lanczos.npz"))['top_K_e']
        except:
            logger.error("Failed loading eigenvalues of " + str(E))
            continue

        HE = np.array([np.convolve(zz,
            [1 for _ in range(K_smooth)], mode="same") for zz in HE]) / K_smooth

        eigv = HE[:, 0:K]

        ax.plot(eigv[:n_epochs, 0], color=cm(key), label=label, linewidth=2)
        ax.plot(eigv[:n_epochs, 1], color=cm(key), linestyle="-.", linewidth=2)
        if ymax is not None:
            ax.set_ylim([ymin, ymax])
        ax.set_yscale("log")
        if showxlabel:
            ax.set_xlabel("Epoch", fontsize=fontsize)
        if showylabel:
            ax.set_ylabel("$\lambda$", fontsize=fontsize)

    if which == "init":
        plt.legend()
    plt.savefig(join(SAVE_DIR, which + save_prefix + "_grid.pdf"), bbox_inches='tight',
        transparent=True,
        pad_inches=0)
    plt.show()
    plt.close()

    # for E in Es:
    #     H, C = load_HC_csv(E)
    #
    #     if which == "lr":
    #         if C['lr'] not in lrs:
    #             continue
    #         key = C['lr']
    #     elif which == "bs":
    #         if C['batch_size'] not in lrs:
    #             continue
    #         key = C['batch_size']
    #     elif which == "init":
    #         key = C['init']
    #
    #     if len(H) == 0:
    #         print("Skipping " + E)
    #         continue
    #
    #     if os.path.exists(join(E, "history.npz")):
    #         HE = np.load(join(E, "history.npz"))
    #     else:
    #         HE = np.load(join(E, "lanczos.npz"))
    #     eigv = HE['top_K_e']
    #
    #     ax.plot(H['loss'].values[N_start:], color=cm(key), label="$\eta=" + str(C['lr']) + "$")
    #     ax.set_xlabel("Epoch", fontsize=fontsize)
    #     ax.set_ylabel("$\lambda_1$ / $\lambda_2$", fontsize=fontsize)

    # plt.legend(fontsize=legendfontsize)
    # plt.savefig(join(SAVE_DIR, which + save_prefix + "_grid_loss.pdf"), bbox_inches='tight',
    #     transparent=True,
    #     pad_inches=0)
    # plt.show()

if __name__ == "__main__":
    adam_dir = os.path.join(RESULTS_DIR, "rebuttal/adam", "adam")
    init_dir = os.path.join(RESULTS_DIR, "rebuttal/init", "init2")
    imdb_dir = os.path.join(RESULTS_DIR, "rebuttal", "imdb2")

    which = sys.argv[1]
    print(which)

    # Defaults
    yminbs=None
    ymaxbs = None
    n_epochs = 1000000
    ymaxlr = None
    showylabel = False
    K_smooth=5

    if which == "adam":
        res_dir = adam_dir
        bss = [1024, 128, 8]
        lrs = [0.00001, 0.0001, 0.001]
        Es_bs = [os.path.join(res_dir, 'constant_bs={}_lr=0.0005'.format(bs)) for bs in bss]
        Es_lr = [os.path.join(res_dir, 'constant_lr=' + str(bs)) for bs in ['0.00001', '0.0001', '0.001']]

        bslr = 0.05
        titlebs="SimpleCNN+Adam, $S$:$1024/128/8$"
        titlelr="SimpleCNN+Adam, $\eta$: $10^{-5}/10^{-4}/10^{-3}$"

    elif which == "init":
        res_dir = init_dir
        inits = ['random_normal', 'glorot_uniform', 'random_uniform']
        Es_bs = [os.path.join(res_dir, 'init={}'.format(bs)) for bs in inits]
        logger.info(Es_bs)
        bslr = 0.05
        titlebs = "SimpleCNN+Different inits"
        plot(Es_bs, ymax=ymaxbs, ymin=yminbs, K_smooth=K_smooth, n_epochs=n_epochs, which="init",
            lrs=inits,  save_prefix="init", showxlabel=True, showylabel=showylabel,
            title=titlebs)
        exit(0)
    elif which == "imdb":
        res_dir = imdb_dir
        bss = [32, 8, 2]
        lrs = [0.025, 0.05, 0.1]
        Es_bs = [os.path.join(res_dir, 'imdb2_config=cnn_seed=777_lr=0.01_dropout=0.0_bs={}'.format(bs)) for bs in bss]
        Es_lr = [os.path.join(res_dir, 'imdb2_config=cnn_seed=777_lr={}_KK=_dropout=0.0_bs=32'.format(bs)) for bs in lrs]
        bslr = 0.05
        n_epochs = 100
        titlebs = "CNN IMDB, $S$:$32/8/2$"
        titlelr = "CNN IMDB, $\eta$: $0.025/0.05/0.1$"
    else:
        raise NotImplementedError()
    logger.info(Es_bs)
    logger.info(Es_lr)
    plot(Es_bs, ymax=ymaxbs, ymin=yminbs, K_smooth=K_smooth, n_epochs=n_epochs,
        lrs=bss, which="bs", save_prefix=which, showxlabel=True, showylabel=showylabel,
        title=titlebs)
    plot(Es_lr, showxlabel=True, n_epochs=n_epochs,
        lrs=lrs, save_prefix=which, ymin=0, ymax=ymaxlr, showylabel=showylabel, K_smooth=K_smooth,
        title=titlelr)