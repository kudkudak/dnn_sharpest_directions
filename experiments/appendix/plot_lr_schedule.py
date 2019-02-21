#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run as python experiments/appendix/plot_lr_schedule.py
"""

from src.utils.plotting_preamble import *
from src import PROJECT_DIR, RESULTS_DIR

SAVE_DIR = os.path.join(PROJECT_DIR, "experiments", "appendix", "lr_schedule")
os.system("mkdir -p " + SAVE_DIR)

def plot_curves(Es, expname, labels, nstart=1, maxn=800, smooth=1, cm=None):
    fs = 7
    plt.figure(figsize=(fs, fs / 1.5))
    ax = plt.gca()
    color_map = {}
    for key in ['val_acc', 'acc']:
        #         axt = ax.twinx()
        ax.tick_params(labelsize=labelfontsize)

        for E, l in zip(Es, labels):

            if E not in color_map:
                color = next(ax._get_lines.prop_cycler)['color']
                color_map[E] = color
            else:
                color = color_map[E]

            try:
                H, C = load_HC_csv(E)
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

            curve = H[key][nstart:maxn]
            curve = np.convolve(curve, [1] * smooth, mode="valid") / smooth

            ax.plot(curve, label=(l if key=='acc' else "__nolabel__"), color=color, linestyle=("--" if key=='val_acc' else "-"), linewidth=linewidth)
            id_max_val = np.argmax(H['val_acc'])
            F = np.sqrt(np.sum(np.abs(eigv) ** 2, axis=1))
            ax.set_xlabel("Epoch", fontsize=fontsize)
            ax.set_ylabel("Accuracy", fontsize=fontsize)

    ax.legend(fontsize=fontsize * 0.8)
    save_fig(os.path.join(SAVE_DIR, "{}_acc.pdf".format(expname)))
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
            H, C = load_HC_csv(E)
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
        axt.plot(np.sqrt((np.abs(eigv[1:maxn, :]) ** 2).sum(axis=1))[0:maxn],
            label=l, color=color, linewidth=linewidth, linestyle="--")
        ax.set_xlabel("Epoch", fontsize=fontsize)
        ax.set_ylabel("$\lambda_{max}$", fontsize=fontsize)

    ax.legend(fontsize=fontsize * 0.5)
    save_fig(os.path.join(SAVE_DIR, "{}_SN.pdf".format(expname)))


    plt.figure(figsize=(fs, fs / 1.5))
    ax = plt.gca()
    ax.tick_params(labelsize=labelfontsize)
    ax.ticklabel_format(style='sci', axis='both')
    for E, l in zip(Es, labels):

        if cm is None:
            color = next(ax._get_lines.prop_cycler)['color']
        else:
            color = cm(E)

        try:
            H, C = load_HC_csv(E)
        except:
            continue

        if len(H) == 0 or len(H['acc']) < 50:
            print("Skipping " + E)
            continue

        ax.plot(H['lr'], label=l, color=color, linewidth=linewidth)
        ax.set_xlabel("Epoch", fontsize=fontsize)
        ax.set_ylabel("$\eta$", fontsize=fontsize)
    ax.legend(fontsize=fontsize * 0.5)
    save_fig(os.path.join(SAVE_DIR, "{}_eta.pdf".format(expname)))

if __name__=="__main__":
    results_dir = os.path.join(RESULTS_DIR, "nsgd")
    ex = os.path.join(results_dir, "lrsch_resnet")
    Es = glob.glob(os.path.join(ex, "*L=*"))
    print(Es)
    labels = ["L=" + str(int(re.findall(r'L=(\d.*)_', E)[0].split("_")[0])) for E in Es]
    plot_curves(Es, "lrschedule_length", labels, maxn=500, smooth=5, nstart=10, cm=None)
