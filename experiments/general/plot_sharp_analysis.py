#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run as:
python experiments/general/plot_sharp_analysis.py resnet_lr=0.01_logg
python experiments/general/plot_sharp_analysis.py resnet_lr=0.01
python experiments/general/plot_sharp_analysis.py resnet_lr=0.05_slow
python experiments/general/plot_sharp_analysis.py scnn_lr=0.01_long
python experiments/general/plot_sharp_analysis.py scnn_lr=0.01
python experiments/general/plot_sharp_analysis.py scnn_lr=0.01_nsgd; python experiments/general/plot_sharp_analysis.py resnet_lr=0.01_nsgd
python experiments/general/plot_sharp_analysis.py scnn_lr=0.05_slow
for i in resnet_lr=0.01_logg resnet_lr=0.01 scnn_lr=0.01_long scnn_lr=0.01; do python experiments/general/plot_sharp_analysis.py $i; done
"""

from src.utils.plotting_preamble import *
from src.utils.plotting_preamble import _construct_colorer, _construct_colorer_lin_scale
from src import PROJECT_DIR, RESULTS_DIR

SAVE_DIR=os.path.join(PROJECT_DIR, "experiments", "general", "sharp_analysis")

os.system("mkdir -p " + SAVE_DIR)

_load_HC= load_HC
_save_fig=save_fig

def plot_overshoot_figure(path, subpath, save_prefix="",
        which_x="epoch", space_x=-1,
        ids=[0, 2], lrs=None, max_epoch=4000,
        ymin=None, ymax=None, title=""):
    def extract_metric(metrics, metric_key):
        # Metric dict or list -> float
        if isinstance(metrics, list):
            metrics = {k: np.mean(np.array([m[k] for m in metrics]), axis=0) for k in metrics[0]}
        if "*" in metric_key:
            keys = sorted(metrics.keys())
            return np.array([metrics[k] for k in keys if re.match(metric_key, k)])
        elif "loss" in metric_key:
            return metrics['loss']
        elif "norm" in metric_key:
            return metrics['E'][int(metric_key.split("_")[1])]
        else:
            raise NotImplementedError(metric_key)

    def extract_metric_at_LR_dir(data, dir, LR, metric_key):
        dir = str(dir)

        ## LR is given as an ordinal!
        if isinstance(LR, int):
            keys = [k for k in data if k != "before" and k != "after"]
            keys = sorted(keys, key=lambda k: float(k.split("_")[0]))
            key = keys[LR]
            return extract_metric(data[key][dir], metric_key=metric_key)
        else:
            keys = [k for k in data if k != "before" and k != "after" and float(k.split("_")[0]) == float(LR)]
            assert len(keys) == 1
            return extract_metric(data[keys[0]][dir], metric_key=metric_key)

    def get_dirs(data, LR_BS):
        return [k for k in data[LR_BS]]

    def get_LRs(data):
        return [float(k.split("_")[0]) for k in data if k != "before" and k != "after"]

    E = path
    E_track = path + subpath

    H, C = _load_HC(E)

    if lrs is None:
        lrs = [C['lr']]

    # Read epochs_to_run
    print("Query " + os.path.join(E_track, "*decompose*json"))
    runs_p = glob.glob(os.path.join(E_track, "*decompose*json"))  # Take only finished
    runs = [json.load(open(r)) for r in tqdm.tqdm(runs_p, total=len(runs_p))]
    # Annoying code. Remove json
    epochs = [int(os.path.basename(x).split("_")[1][0:-5]) for x in runs_p]
    epoch_to_run = dict(zip(epochs, runs))
    print("Found {} epochs computed".format(len(epoch_to_run)))

    #### HACK: FILTERING ####
    def _filter(run):
        keys = [k for k in run if k != "after" and k != "before"]
        return all([len(run[k]) > 1 for k in keys])

    epoch_to_run = {k: epoch_to_run[k] for k in epoch_to_run if _filter(epoch_to_run[k])}
    epoch_to_run = {k: epoch_to_run[k] for k in epoch_to_run if k <= max_epoch}

    print("Filtered down to {} epochs computed".format(len(epoch_to_run)))

    # Diagnostics
    epochs = []
    SNs = []
    LRs = []
    for epoch in epoch_to_run:
        epochs.append(epoch)
        SNs.append(epoch_to_run[epoch]['before']['E'][0])
        LRs += get_LRs(epoch_to_run[epoch])
    LRs = sorted(set(LRs))
    largerLRs = [lr for lr in LRs if lr >= C['lr']]
    max_epoch = epochs[np.argmax(SNs)]
    min_epoch = epochs[np.argmin(SNs)]

    ## Parametrize
    dir = 0  # Largest eigenvalue
    which_y = "loss"
    label_x = "$\eta$"
    label_y = "loss"
    figsize = fs

    ## Hack - transform passed lrs to normal lrs
    if all(isinstance(lr, int) for lr in lrs):
        lrs = [LRs[lr] for lr in lrs]
        print("Transformed lrs list to {}".format(lrs))

    def y_transformer_rel_change(after, before):
        # after=before => 0
        # after=2*before => +100
        # after=0.5*before => -100%
        return (np.mean(100 * after / before - 100))

    outttxtall = ""

    def plot_correlation_1(dir, which_x, which_y, label_x, label_y,
            C=None, ymax=None, ymin=None, lrs=None, xmin=None, xmax=None, color_scheme="epoch", title="",
            y_transformer=y_transformer_rel_change):
        outtxt = ""

        x = []
        y = []
        y_before = []
        y_after = []
        color = []
        lrs_runs = []

        spectral_norms = []
        lrs_epochs = []
        for epoch in epoch_to_run:
            sn = extract_metric(epoch_to_run[epoch]['before'], "norm_0")
            spectral_norms.append(sn)
            lrs_epochs += [lr for lr in sorted(get_LRs(epoch_to_run[epoch]))]
        spectral_norms = sorted(spectral_norms)

        if color_scheme == "epoch":
            cm = _construct_colorer(sorted(list(epoch_to_run)))
        elif color_scheme == "SN":
            cm = _construct_colorer(sorted(spectral_norms))
        else:
            if lrs is not None:
                cm = _construct_colorer(sorted(lrs))
                if len(lrs) == 3:
                    cm = lambda lr: ["blue", "green", "red"][sorted(lrs).index(lr)]
            else:
                cm = _construct_colorer_lin_scale(vmin=min(lrs_epochs), vmax=max(lrs_epochs))

        plt.figure(figsize=(fs, int(fs * 0.9)))
        plt.title(title, fontsize=fontsize)
        ax = plt.gca()
        import matplotlib.ticker as ticker
        if space_x != -1:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(space_x))

        for epoch in epoch_to_run:
            data = epoch_to_run[epoch]

            if lrs is None:
                lrs_epoch = sorted(get_LRs(data))
            else:
                lrs_epoch = lrs

            metric_before = extract_metric(epoch_to_run[epoch]['before'], which_y)
            spectral_norm = extract_metric(epoch_to_run[epoch]['before'], "norm_0")

            ## Process data. Format from track_decomposed_step.py
            for lr in lrs_epoch:
                if lr not in get_LRs(epoch_to_run[epoch]):
                    continue

                if which_x == "LR":
                    x.append(lr)
                elif which_x == "SN":
                    x.append(spectral_norm)
                elif which_x == "epoch":
                    if C['epoch_size'] != -1:
                        x.append(int(epoch) * C['epoch_size'] / 45000.)
                    else:
                        x.append(int(epoch))
                elif which_x == "iteration_to_epoch":
                    x.append(int(epoch) * C['batch_size'] / 45000)
                elif which_x == "iteration":
                    x.append(int(epoch) * C['epoch_size'] / C['batch_size'])
                else:
                    raise NotImplementedError()

                metric_after = np.mean(extract_metric_at_LR_dir(data=data, LR=lr,
                    dir=dir, metric_key=which_y))

                y.append(y_transformer(metric_after, metric_before))  # Larger then

                y_before.append(metric_before)
                y_after.append(metric_after)

                if color_scheme == "epoch":
                    color.append(cm(epoch))
                elif color_scheme == "SN":
                    color.append(cm(spectral_norm))
                elif color_scheme == "none":
                    pass
                else:
                    color.append(cm(lr))

                lrs_runs.append(lr)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.tick_params(labelsize=labelfontsize)
        ax.axhline(y=0, linewidth=linewidth, linestyle="--", color="black")

        if C is not None and which_x == "LR":
            ax = plt.gca()
            ax.axvline(x=C['lr'], linestyle="--")
        if len(color):
            plt.scatter(x, y, color=color, s=150)
        else:
            plt.scatter(x, y, s=100)

        plt.ylim(ymin, ymax)
        plt.xlim(xmin, xmax)
        plt.xlabel(label_x, fontsize=fontsize)
        plt.ylabel(label_y, fontsize=fontsize)

        # Analysis (last 25 points)
        # TODO: Move as separate code. Perhaps merge with decompose script?
        lrs_runs = np.array(lrs_runs)
        y_before = np.array(y_before)
        y_after = np.array(y_after)
        y_before = y_before[np.argsort(x)]
        y_after = y_after[np.argsort(x)]
        lrs = lrs_runs[np.argsort(x)]

        print("Running")
        print(sorted(list(set(lrs_runs))))
        for lr in sorted(list(set(lrs_runs))):
            print("BUM")
            print(lr)

            selector = lrs == lr
            y_after_lr = y_after[lrs == lr]
            y_before_lr = y_before[lrs == lr]

            print(sorted(x)[-25:])
            print(("For id=" + str(dir) + " lr=" + str(lr)))
            print("Loss after:")
            print((np.mean(y_after_lr[-25:])))
            print("Loss before:")
            print((np.mean(y_before_lr[-25:])))
            print("Rel. loss change:")
            print((np.mean(y_after_lr[-25:] / y_before_lr[-25:])))

            outtxt += str(lr)
            outtxt += str((np.mean(y_after_lr[-25:] / y_before_lr[-25:])))
            outtxt += "\n"

        return outtxt

    x_label = which_x[0].upper() + which_x[1:]

    for id in ids + ['none']:
        if id != 'none':
            labelid = id + 1
        else:
            labelid = id
        outttxtall += str(id)
        outttxtall += "\n ======== \n"
        outttxtall += plot_correlation_1(C=C, dir=id, which_x=which_x, which_y="loss", ymin=ymin,
            ymax=ymax, color_scheme="LR",
            title=title + " $e_{}$".format(labelid),
            label_x=x_label, label_y="$\Delta$ Loss ($\%$)", lrs=lrs)
        print("Saving to " + save_prefix + "_decompose_{}.pdf".format(id))
        _save_fig(join(SAVE_DIR, save_prefix + "_decompose_{}.pdf".format(id)))
        plt.show()
        plt.close()

    open(join(SAVE_DIR, save_prefix + "_decompose_{}.txt".format(id)), "w").write(outttxtall)


def estimate_width(path, maxepoch=400, save_prefix=""):
    H = pickle.load(open(path + "/history.pkl", "rb"))
    C = json.load(open(os.path.join(path, "config.json")))
    epses = np.array(pickle.load(open(path + "/loss_surfaces_eps.pkl", "rb")))

    thresholds = [1.01, 1.1, 1.5, 2.0]
    width_estimates = [[] for _ in thresholds]

    N = len(H[next(iter(H))])
    N = min(N, maxepoch)

    outtxt = ""

    for id2 in [0]:
        for id in (range(1, N, 1)):
            vals = H['loss_curve_' + str(id2)][id]  # It should be epoch to the left
            vals = np.array(vals)
            epses_plot = np.array(epses)

            id0 = np.where(epses_plot == 0)[0][0]

            for idt in range(len(thresholds)):
                t = thresholds[idt]

                vals_right = vals[id0 + 1:]
                epses_right = epses_plot[id0 + 1:]
                vals_left = vals[0:id0]
                epses_left = epses_plot[0:id0]

                if id == 400:
                    print(epses_right)
                    print(vals_right)
                    print(epses_left)
                    print(vals_left)

                # id-1 is annoying, but this is because one is computed at the end
                # and other at the beginning of the epoch
                first_right = epses_right[np.where(vals_right >= t * H['loss'][id - 1])[0]]
                first_left = epses_left[np.where(vals_left >= t * H['loss'][id - 1])[0]]

                if id == 400:
                    print(first_right)
                    print(first_left)

                if len(first_right) and len(first_left):
                    width_estimates[idt].append(max(first_right[0], first_left[0]))
                else:
                    width_estimates[idt].append(0)

    for idt in range(len(thresholds)):
        Z = np.convolve(width_estimates[idt], [1 for _ in range(5)], mode="valid") / 5
        plt.plot(Z)
        plt.title(thresholds[idt])
        print((thresholds[idt], np.mean(width_estimates[idt][-25:])))
        outtxt += str((thresholds[idt], np.mean(width_estimates[idt][-25:])))
        outtxt += "\n"
        plt.savefig(os.path.join(SAVE_DIR, "estimate_width_" + save_prefix + "_thr" + str(idt) + ".pdf"),
            bbox_inches = 'tight',
            transparent = True,
            pad_inches = 0)
        plt.show()
        plt.close()

    open(os.path.join(SAVE_DIR, "estimate_width_" + save_prefix + "_thr" + str(idt) + ".txt"), "w").write(outtxt)


def plot_figure_2(path, save_prefix="", ids=[0, 2], zmin=0, zmax=5, space_x=2.0,
        which_x="epoch", title="Resnet-32",
        maxepoch=100000, resolution=10, add_line=False, subsample=1):
    H = pickle.load(open(path + "/history.pkl", "rb"))
    C = json.load(open(os.path.join(path, "config.json")))
    epses = np.array(pickle.load(open(path + "/loss_surfaces_eps.pkl", "rb")))
    N = len(H[next(iter(H))])
    N = min(N, maxepoch)
    SNs = H['SN'][0:N]
    cm = _construct_colorer(sorted(SNs))

    thresholds = [1.01, 1.1, 1.5, 2.0]
    width_estimates = [[] for _ in thresholds]

    for id2 in ids:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.gca(projection='3d')
        ax.tick_params(labelsize=labelfontsize * 0.7)
        ax.set_title(title + " $e_{}$".format(id2 + 1), fontsize=fontsize, y=0.9)

        ## Matplotlib hack
        if zmax == -1:
            zmax = np.max(H['loss'])
            print("Picking as max loss")

        for id in (range(1, N, subsample)):
            vals = H['loss_curve_' + str(id2)][id]
            vals = np.array(vals)
            vals = vals

            ids = np.where((np.abs(epses) <= resolution))  # & (vals <= (1.5 * zmax)))

            if len(ids) == 0:
                continue

            vals = vals[ids]

            vals = np.minimum(vals, 1.0 * zmax)

            epses_plot = np.array(epses[ids])

            if vals[0] < vals[-1]:  # Visualization aid
                vals = vals[::-1]

            if which_x == "iteration":
                y = (len(vals) - id) * C['epoch_size'] / C['batch_size']
            elif which_x == "epoch":
                # Small hack
                y = float(((len(vals) - id) * C['epoch_size'])) / 45000
            else:
                y = len(vals) - id

            ax.plot(
                epses_plot,
                [y for _ in vals],
                vals,
                label="",
                linewidth=linewidth,
                color=cm(SNs[id]))

        ax.set_zlim([zmin, zmax + 0.3])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        #         ax.set_zscale("log")
        if which_x == "epoch":
            ax.set_yticklabels([str(-(y)) for y in list(ax.get_yticks())][1:-1])
            ax.set_ylabel("Epoch", fontsize=fontsize)
        elif which_x == "iteration":
            ax.set_yticklabels([str(-int(y)) for y in list(ax.get_yticks())][1:-1])
            ax.set_ylabel("Iteration", fontsize=fontsize)
        elif which_x == "":
            ax.set_yticklabels([str(-int(y)) for y in list(ax.get_yticks())][1:-1])
            ax.set_ylabel("Epoch", fontsize=fontsize)

        ax.set_xlabel("$k$", fontsize=fontsize)
        ax.set_zlabel("L", fontsize=fontsize)
        ax.axes.xaxis.labelpad = 10
        ax.axes.yaxis.labelpad = 10
        #         ax.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(space_x))
        ax.grid(False)

        ## Plot
        print("Saving fig to " + os.path.join(SAVE_DIR, save_prefix + '_3d_{}.pdf'.format(id2)))
        plt.savefig(os.path.join(SAVE_DIR, save_prefix + '_3d_{}.pdf'.format(id2)),
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)
        plt.show()

    # More matplotlib hacks
    # x_label = "Batch" if C['epoch_size'] != -1 else "Epoch"
    # plt.plot(H['SN'][1:])
    # plt.xlabel(x_label, fontsize=fontsize)
    # plt.ylabel("$\lambda$", fontsize=fontsize)
    # plt.savefig(os.path.join(SAVE_DIR, save_prefix + 'SN.pdf'),
    #     bbox_inches='tight',
    #     transparent=True,
    #     pad_inches=0)
    # plt.show()
    # ax = plt.gca()
    # ax.tick_params(labelsize=labelfontsize)
    # plt.xlabel(x_label, fontsize=fontsize)
    # plt.ylabel("Accuracy", fontsize=fontsize)
    # plt.plot(H['val_acc'][0:maxepoch], linestyle="--", label="Val. accuracy")
    # plt.plot(H['acc'][0:maxepoch], label="Train accuracy")
    # plt.legend(fontsize=fontsize)
    # ax.set_xlabel("Iteration", fontsize=fontsize)
    # plt.savefig(os.path.join(SAVE_DIR, save_prefix + '_acc.pdf'),
    #     bbox_inches='tight',
    #     transparent=True,
    #     pad_inches=0)
    #
    # print(np.mean(H['acc'][0:maxepoch][-10:]))
    # plt.show()
if __name__=="__main__":
    which = sys.argv[1]
    results_dir=os.path.join(RESULTS_DIR, "sharp_analysis")

    # Configure

    resolution = 8
    if which=="resnet_lr=0.01_logg":
        title = "Resnet-32"
        maxepoch = 300
        space_x = 100
        which_x = ""
        dy=50
    elif which=="scnn_lr=0.01_long":
        title = "SimpleCNN"
        maxepoch = 300
        space_x = 100
        which_x = ""
        dy=50
    elif which=="resnet_lr=0.01":
        maxepoch=550
        which_x = "epoch"
        title = "Resnet-32"
        space_x=2
        dy=10
    elif which=="scnn_lr=0.01":
        title = "SimpleCNN"
        dy=50
        which_x = "epoch"
        maxepoch = 550
        space_x = 2
    elif which=="resnet_lr=0.01_nsgd":
        maxepoch=550
        which_x = "epoch"
        title = "Resnet-32 + NSGD"
        space_x=2
        dy=10
    elif which=="scnn_lr=0.01_nsgd":
        title = "SimpleCNN + NSGD"
        dy=50
        which_x = "epoch"
        maxepoch = 550
        space_x = 2
    elif which == "resnet_lr=0.05_slow":
        maxepoch = 550
        which_x = "epoch"
        resolution = 4
        title = "Resnet-32"
        space_x = 2
        dy = 10
    elif which == "scnn_lr=0.05_slow":
        title = "SimpleCNN"
        resolution = 4
        dy = 50
        which_x = "epoch"
        maxepoch = 550
        space_x = 2
    else:
        raise NotImplementedError()

    estimate_width(os.path.join(results_dir, which), save_prefix=which.replace(".", "_").replace("=", "_"))
    plot_figure_2(os.path.join(results_dir, which),
        which.replace(".", "_").replace("=", "_"), # For overleaf compliance
        ids=[0, 2, 4],
        maxepoch=maxepoch,
        which_x=which_x,
        resolution=resolution,
        title=title,
        subsample=1,
        space_x=space_x,
        zmin=None,
        zmax=-1)
    plot_overshoot_figure(os.path.join(results_dir, which),
        "", which.replace(".", "_").replace("=", "_"), # For overleaf compliance
        ids=[0, 1, 2, 3],
        lrs=[2, 3, 4],
        max_epoch=maxepoch,
        space_x=space_x,
        which_x="epoch",
        title=title,
        ymin=-dy,
        ymax=dy)