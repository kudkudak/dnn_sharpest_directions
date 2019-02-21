#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run as:

python experiments/noise_grid/plot.py resnet/scnn/vgg11/ptb
for i in resnet scnn vgg11 ptb; do python experiments/noise_grid/plot.py $i;  done
for i in resnet bn_resnet scnn vgg11 ptb; do python experiments/noise_grid/plot.py $i;  done
"""

from src.utils.plotting_preamble import *
from src import PROJECT_DIR, RESULTS_DIR

SAVE_DIR=os.path.join(PROJECT_DIR, "experiments", "noise_grid", "noise_grid")
os.system("mkdir -p " + SAVE_DIR)

def plot_figure_avg_evol_merged(Es, Es_zoom, lrs, save_prefix, title, which="lr", ymin=None,
        showxlabel=False, showylabel=False,
        ymax=None, K=10, n_epochs_zoom=2., epoch_size=45000, n_epochs=1000000, K_smooth=1):
    N_start = 1
    N = 10000

    plt.figure(figsize=(fs, fs / 1.7))
    plt.title(title, fontsize=fontsize)

    cm = lambda lr: ["red", "green", "blue"][lrs.index(lr)]

    ax = plt.gca()
    ax.tick_params(labelsize=labelfontsize)
    for E, E_zoom in zip(Es, Es_zoom):
        try:
            H, C = load_HC_csv(E)
            H_zoom, C_zoom = load_HC_csv(E_zoom)
        except:
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

        if len(H) == 0:
            print("Skipping " + E)
            continue

        try:
            if os.path.exists(join(E, "history.npz")):
                HE = np.load(join(E, "history.npz"))['top_K_e']
            else:
                HE = np.load(join(E, "lanczos.npz"))['top_K_e']

        except:
            continue

        if os.path.exists(join(E_zoom, "history.npz")):
            HE_zoom = np.load(join(E_zoom, "history.npz"))['top_K_e']
        else:
            HE_zoom = np.load(join(E_zoom, "lanczos.npz"))['top_K_e']

        HE = np.array([np.convolve(zz,
            [1 for _ in range(K_smooth)], mode="same") for zz in HE]) / K_smooth
        HE_zoom = np.array([np.convolve(zz,
            [1 for _ in range(K_smooth)], mode="same") for zz in HE_zoom]) / K_smooth

        print("Found {} epochs and {} zoom iteration".format(len(HE), len(HE_zoom)))

        eigv = HE[N_start:n_epochs, 0:K]
        eigv_zoom = HE_zoom[:, 0:K]

        prefix = int(epoch_size * n_epochs_zoom) // C_zoom['epoch_size']
        prefix = int(prefix)

        eigv_merged = np.concatenate([eigv_zoom[0:prefix], eigv[:]], axis=0)

        # print(prefix)
        #
        # print(C_zoom['batch_size'])
        # print(C_zoom['epoch_size'])
        # print(C_zoom['n_epochs'])
        # print(len(eigv_zoom))  ## Unfinished!
        # print("Needed " + str(prefix))
        print("Found {} iterations".format(len(eigv_merged)))
        print("Epoch size{} {}".format(C['epoch_size'], C_zoom['epoch_size']))

        assert len(eigv_zoom) >= prefix

        print(eigv_merged[-10:,0])


        # X axis
        M = len(eigv[:])
        x_a = list(np.linspace(0, 1, prefix))  # Fictional unit
        x_b = list(np.linspace(1, 2, M))
        xs = x_a + x_b

        ax.plot(xs, eigv_merged[:, 0], color=cm(key), label=label, linewidth=2)
        ax.plot(xs, eigv_merged[:, 1], color=cm(key), linestyle="-.", linewidth=2)
        ax.axvline(x=1.)
        if ymax is not None:
            ax.set_ylim([ymin, ymax])
        ax.set_yscale("log")
        if showxlabel:
            ax.set_xlabel("Epoch", fontsize=fontsize)
        if showylabel:
            ax.set_ylabel("$\lambda$", fontsize=fontsize)

        new_labels = []
        current_labels = list(ax.get_xticks())

        print(current_labels)

        current_labels = [float(y) for y in current_labels]

        for i in range(len(current_labels)):

            if current_labels[i] <= 1.0:
                new_labels.append(current_labels[i] * n_epochs_zoom)
            else:
                val = int(1 + M * (current_labels[i] - 1.0))
                if val % 10 != 0:
                    val = (int(val / 10) + 1) * 10
                new_labels.append(val)

        print(new_labels)
        ax.set_xticklabels(new_labels)

    # ax.legend(fontsize=legendfontsize)
    plt.savefig(join(SAVE_DIR, which + save_prefix + "_grid.pdf"), bbox_inches='tight',
        transparent=True,
        pad_inches=0)
    plt.show()
    plt.close()
    ax = plt.gca()
    axt = ax.twinx()

    for E in Es:
        H, C = load_HC_csv(E)

        if which == "lr":
            if C['lr'] not in lrs:
                continue
            key = C['lr']
        elif which == "bs":
            if C['batch_size'] not in lrs:
                continue

            key = C['batch_size']
        if len(H) == 0:
            print("Skipping " + E)
            continue

        if os.path.exists(join(E, "history.npz")):
            HE = np.load(join(E, "history.npz"))
        else:
            HE = np.load(join(E, "lanczos.npz"))
        eigv = HE['top_K_e']

        cc = eigv[N_start:N, 0]
        cc = np.convolve(cc, [1 for _ in range(K_smooth)]) / K_smooth

        ax.plot(H['loss'].values[N_start:], color=cm(key), label="$\eta=" + str(C['lr']) + "$")
        ax.set_xlabel("Epoch", fontsize=fontsize)
        ax.set_ylabel("$\lambda_1$ / $\lambda_2$", fontsize=fontsize)

    plt.legend(fontsize=legendfontsize)
    plt.savefig(join(SAVE_DIR, which + save_prefix + "_grid_loss.pdf"), bbox_inches='tight',
        transparent=True,
        pad_inches=0)
    plt.show()

if __name__ == "__main__":
    vgg_dir = os.path.join(RESULTS_DIR, "noise_grid", "ng_vgg11")
    resnet_dir = os.path.join(RESULTS_DIR, "noise_grid", "ng_resnet")
    bn_resnet_dir = os.path.join(RESULTS_DIR, "noise_grid", "bn_ng_resnet")
    simple_cnn_dir = os.path.join(RESULTS_DIR, "noise_grid", "ng_scnn")
    ptb_dir = os.path.join(RESULTS_DIR, "noise_grid", "ng_ptb3")

    which = sys.argv[1]
    print(which)

    # Defaults
    Es_lr = None
    n_epochs_zoom = 2.0
    yminbs=0
    ymaxbs = 100
    epoch_size = 45000
    n_epochs = 1000000
    ymaxlr = None
    showylabel = False
    K_smooth=5


    if which == "scnn":
        res_dir = simple_cnn_dir
        # bss = [512, 128, 32]
        bss = [1024, 128, 32]
        lrs = [0.001, 0.01, 0.1]
        bslr = 0.05
        titlebs="SimpleCNN, $S$:$1024/128/32$"
        # titlebs="SimpleCNN, $S$:$512/64/8$"
        titlelr="SimpleCNN, $\eta$: $10^{-3}/10^{-2}/10^{-1}$"
    elif which == "resnet":
        res_dir = resnet_dir
        # bss = [512, 128, 32]
        bss = [1024, 128, 32]
        # bss = [512, 128, 32]
        # bss = [512, 64, 8]
        lrs = [0.001, 0.01, 0.1]
        bslr = 0.05
        showylabel = True
        # titlebs="ResNet-32, $S$:$512/128/32$"
        titlebs="ResNet-32, $S$:$1024/128/32$"
        titlelr="ResNet-32, $\eta$: $10^{-3}/10^{-2}/10^{-1}$"
    elif which == "bn_resnet":
        res_dir = bn_resnet_dir
        bss = [512, 128, 8]
        ymaxbs = 1000
        lrs = [0.001, 0.01, 0.1]
        bslr = 0.05
        showylabel = True
        titlebs = "ResNet-BN-32, $S$:$512/128/8$"
        titlelr = "ResNet-BN-32, $\eta$: $10^{-3}/10^{-2}/10^{-1}$"
    elif which == "vgg11":
        res_dir = vgg_dir
        # bss = [512, 128, 32]
        bss = [512, 128, 32]
        lrs = [0.001, 0.01, 0.1]
        bslr = 0.05
        ymaxbs = 2000
        titlebs="VGG11, $S$:$512/128/32$"
        titlelr="VGG11, $\eta$: $10^{-3}/10^{-2}/10^{-1}$"
    elif which == "ptb":
        res_dir = ptb_dir
        bss = [128, 32, 8]
        lrs = sorted([5.0, 1.0, 0.2])
        Es_lr = [res_dir + "/constant_lr={}".format(lr) for lr in lrs]
        bslr = 1.0
        ymaxbs = ymaxlr = 50
        n_epochs=25
        n_epochs_zoom=0.5
        epoch_size=46000
        titlebs = "LSTM, $S$:$128/32/8$"
        titlelr = "LSTM, $\eta$: $5 \cdot 10^{-1}/10^{-1}/2 \cdot 10^{-1}$"
    else:
        raise NotImplementedError()


    # Plot BS grid
    Es = glob.glob(res_dir + "/constant*bs*lr=" + str(bslr))
    Es_zoom = [E.replace("constant_bs=", "zoom_constant_bs=") for E in Es]
    plot_figure_avg_evol_merged(Es, Es_zoom, ymax=ymaxbs, ymin=yminbs, K_smooth=K_smooth, n_epochs=n_epochs,
        lrs=bss, which="bs", save_prefix=which, showxlabel=True, showylabel=showylabel,
        title=titlebs, n_epochs_zoom=n_epochs_zoom, epoch_size=epoch_size)

    # Plot LR grid
    if Es_lr is None:
        Es_lr = glob.glob(res_dir + "/constant*lr*")
    Es_zoom = [E.replace("constant_lr=", "zoom_constant_lr=") for E in Es_lr]
    plot_figure_avg_evol_merged(Es_lr, Es_zoom, showxlabel=True, n_epochs=n_epochs,
        lrs=lrs, save_prefix=which, ymin=0, ymax=ymaxlr, showylabel=showylabel, K_smooth=K_smooth,
        title=titlelr, n_epochs_zoom=n_epochs_zoom, epoch_size=epoch_size)