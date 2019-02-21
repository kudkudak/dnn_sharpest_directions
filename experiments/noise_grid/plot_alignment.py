#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plots experiment exploring alignment vs LR

Run as:

python experiments/noise_grid/plot_alignment.py
"""

from src.utils.plotting_preamble import *
from src import PROJECT_DIR, RESULTS_DIR

SAVE_DIR=os.path.join(PROJECT_DIR, "experiments", "noise_grid", "alignment")
os.system("mkdir -p " + SAVE_DIR)

def plot(Es, lrs, save_prefix, title, which="lr", K_smooth=1, showylabel=True, showlegend=True):
    plt.figure(figsize=(fs, fs / 1.7))
    plt.title(title, fontsize=fontsize)

    cm = lambda lr: ["red", "green", "blue", "purple"][lrs.index(lr)]

    ax = plt.gca()
    ax.tick_params(labelsize=labelfontsize)
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    # Random alignment
    def _dot(a, b):
        return a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b) + 1e-5)
    D = 350000 # Hack, but # of params is around this point for Rsnet-32 and SimpleCNN
    random_alig = [np.abs(_dot(np.random.uniform(-1,1,size=(D,)), np.random.uniform(-1,1,size=(D,)))) for _ in range(1000)]
    print((np.mean(random_alig), np.std(random_alig)))
    random_alig = np.mean(random_alig)

    for E in Es:
        try:
            H, C = load_HC_csv(E)
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

        n = 0
        for i in range(1):
            n += (np.convolve(H['alig/{}_abs_mean'.format(i)], [1 for _ in range(K_smooth)],
                mode="valid") / K_smooth) ** 2
        n = np.sqrt(n)

        cc = np.convolve(H['alig/0_abs_mean'], [1 for _ in range(K_smooth)], mode="valid") / K_smooth
        cc += np.convolve(H['alig/1_abs_mean'], [1 for _ in range(K_smooth)], mode="valid") / K_smooth
        cc += np.convolve(H['alig/2_abs_mean'], [1 for _ in range(K_smooth)], mode="valid") / K_smooth
        cc += np.convolve(H['alig/3_abs_mean'], [1 for _ in range(K_smooth)], mode="valid") / K_smooth
        cc += np.convolve(H['alig/4_abs_mean'], [1 for _ in range(K_smooth)], mode="valid") / K_smooth
        cc /= 5
        ccx = np.convolve(100 * H['acc'], [1 for _ in range(K_smooth)], mode="valid") / K_smooth

        ax.scatter(ccx, cc, color=cm(key), label=label, linewidth=2)

        ax.set_xlabel("Accuracy ($\%$)", fontsize=fontsize)
        if showylabel:
            ax.set_ylabel("Cosine", fontsize=fontsize)

    # ax.plot(ccx, [random_alig for _ in range(len(ccx))], color="purple", label="Baseline (random vectors)", linewidth=2)
    ax.axhline(random_alig, color="purple", label="Random vector", linewidth=2)
    if showlegend:
        plt.legend(fontsize=0.8 * fontsize, loc="lower left")

    plt.savefig(join(SAVE_DIR, which + save_prefix + "_grid.pdf"), bbox_inches='tight',
        transparent=True,
        pad_inches=0)
    plt.show()
    plt.close()


if __name__ == "__main__":
    resnet_dir = os.path.join(RESULTS_DIR, "noise_grid", "alig_resnet_2")
    simple_cnn_dir = os.path.join(RESULTS_DIR, "noise_grid", "alig_scnn_2")

    # Plot
    Es = glob.glob(simple_cnn_dir + "/*")
    Es = sorted(Es, key=lambda E: load_HC_csv(E)[1]['lr'])
    lrs = sorted([0.1, 0.01, 0.001])
    plot(Es,
        lrs=lrs, save_prefix="simplecnn", K_smooth=10, showylabel=True, showlegend=False,
        title="SimpleCNN")
    Es = sorted(glob.glob(resnet_dir + "/*"), key=lambda E: load_HC_csv(E)[1]['lr'])
    lrs = sorted([0.1, 0.01, 0.001])
    plot(Es,
        lrs=lrs, save_prefix="resnet32", K_smooth=10, showylabel=True,
        title="Resnet-32")
