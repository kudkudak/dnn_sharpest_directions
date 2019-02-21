#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Run as : python experiments/general/plot_escaping_ablation.py

from src.utils.plotting_preamble import *
from src import PROJECT_DIR, RESULTS_DIR

SAVE_DIR=os.path.join(PROJECT_DIR, "experiments", "general", "escape_ablation")
os.system("mkdir -p " + SAVE_DIR)

if __name__ == "__main__":
    Es = glob.glob(os.path.join(RESULTS_DIR, "escaping_ablation_topbottom/top*"))
    Es = sorted(Es, key=lambda E: float(os.path.basename(E).split("=")[-1]))
    labels = [os.path.basename(l) for l in Es]
    labels = ["$\gamma={}$".format(l.split("=")[-1]) for l in labels]

    baseline = os.path.join(os.path.join(RESULTS_DIR, "escaping_ablation2"), "normal_n=1000")
    Es.insert(2, baseline)
    labels.insert(2, "$\gamma=1.0$")

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.tick_params(labelsize=fontsize * 0.8)
    import matplotlib.ticker as ticker

    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))

    for E, l in zip(Es, labels):
        try:
            H = pd.read_csv(os.path.join(E, "H.csv"))
        except:
            continue
        plt.plot(H['SN'], label=l)
    plt.legend(fontsize=fontsize * 0.8)
    import itertools

    for l, ms in zip(ax.lines, itertools.cycle('>^+*')):
        l.set_marker(ms)

    plt.xlabel("Iteration", fontsize=fontsize)
    plt.ylabel("$\lambda_{max}$", fontsize=fontsize)

    # plt.legend(fontsize=labelfontsize)
    save_fig(os.path.join(SAVE_DIR, "lrchange_escape_SN_evolution.pdf"))
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.tick_params(labelsize=labelfontsize * 1.2)
    for E, l in zip(Es, labels):
        try:
            H = pd.read_csv(os.path.join(E, "H.csv"))
        except:
            continue
        plt.plot(H['loss'], label=l)
        print((os.path.basename(E), np.mean(H['loss'][-10:])))

    import itertools

    for l, ms in zip(ax.lines, itertools.cycle('>^+*')):
        l.set_marker(ms)

    plt.xlabel("Iteration", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.legend(fontsize=fontsize * 0.8, loc="lower left")
    save_fig(os.path.join(SAVE_DIR, "lrchange_escape_loss_evolution.pdf"))

    E = os.path.join(RESULTS_DIR, "escaping_ablation2")

    names = ["normal_n=1000", "onlytop_n=1000", "constanttop_n=1000", "removetop_n=1000"]
    labels = ["Baseline", "Top", "Const. Top", "No Top"]

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.tick_params(labelsize=labelfontsize * 1.2)
    import matplotlib.ticker as ticker

    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    for n, l in zip(names, labels):
        try:
            H = pd.read_csv(os.path.join(E, n, "H.csv"))
        except:
            continue
        plt.plot(H['SN'], label=l)
    # plt.legend(fontsize=fontsize * 0.8)
    import itertools

    for l, ms in zip(ax.lines, itertools.cycle('>^+*')):
        l.set_marker(ms)

    plt.xlabel("Iteration", fontsize=fontsize)
    plt.ylabel("$\lambda_{max}$", fontsize=fontsize)
    save_fig(os.path.join(SAVE_DIR, "escape_SN_evolution.pdf"))
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.tick_params(labelsize=fontsize * 0.8)
    for n, l in zip(names, labels):
        try:
            H = pd.read_csv(os.path.join(E, n, "H.csv"))
        except:
            continue
        plt.plot(H['loss'], label=l)

    import itertools

    for l, ms in zip(ax.lines, itertools.cycle('>^+*')):
        l.set_marker(ms)

    plt.xlabel("Iteration", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.legend(fontsize=fontsize * 0.9, loc="lower left")
    save_fig(os.path.join(SAVE_DIR, "escape_loss_evolution.pdf"))