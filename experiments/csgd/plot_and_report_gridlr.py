#!/usr/bin/env python
"""
Run as:

python experiments/csgd/plot_and_report_gridlr.py resnet calc_force
python experiments/csgd/plot_and_report_gridlr.py scnn calc_force
python experiments/csgd/plot_and_report_gridlr.py ffmn_scnn calc
python experiments/csgd/plot_and_report_gridlr.py scnn calc_force; python experiments/csgd/plot_and_report_gridlr.py resnet calc_force
python experiments/csgd/plot_and_report_gridlr.py highlr_resnet calc; python experiments/csgd/plot_and_report_gridlr.py mom2_resnet calc; python experiments/csgd/plot_and_report_gridlr.py resnet_c100 calc
python experiments/csgd/plot_and_report_gridlr.py mom3_resnet calc
"""
# -*- coding: utf-8 -*-

from src.utils.plotting_preamble import *
from experiments.csgd.utils import _construct_colorer
from experiments.csgd.utils import *

from src import RESULTS_DIR
results_dir = RESULTS_DIR

if __name__ == "__main__":
    which = sys.argv[1]
    calculation = sys.argv[2]

    if which == "resnet":
        ex = os.path.join(results_dir, "nsgd/resnet")
        expname = "resnet32_cifar10"
        baselines = glob.glob(os.path.join(results_dir, "nsgd/baselines/*resnet*"))
        Es = glob.glob(os.path.join(ex, "*KK=5*"))
    elif which == "ffmn_scnn":
        ex = os.path.join(results_dir, "nsgd/ffmn_scnn")
        expname = "scnn_fmnist"
        baselines = None
        Es = glob.glob(os.path.join(ex, "*KK=5*"))
    elif which == "highlr_resnet":
        ex = os.path.join(results_dir, "nsgd/highlr_resnet")
        expname = "resnet32_cifar10_highlr"
        baselines = None
        Es = glob.glob(os.path.join(ex, "*KK=5*"))
    elif which == "mom2_resnet":
        ex = os.path.join(results_dir, "nsgd/mom2_resnet")
        expname = "resnet32_cifar10_mom2"
        baselines = None
        Es = glob.glob(os.path.join(ex, "*KK=5*"))
    elif which == "mom3_resnet":
        ex = os.path.join(results_dir, "nsgd/mom3_resnet")
        expname = "resnet32_cifar10_mom3"
        baselines = None
        Es = glob.glob(os.path.join(ex, "*KK=5*"))
    elif which == "resnet_c100":
        ex = os.path.join(results_dir, "nsgd/resnet_c100")
        expname = "resnet32_cifar100"
        baselines = None
        Es = glob.glob(os.path.join(ex, "*KK=5*"))
    elif which == "scnn":
        ex = os.path.join(results_dir, "nsgd/scnn")
        expname = "scnn_cifar10"
        baselines = glob.glob(os.path.join(results_dir, "nsgd/baselines/*medium*"))
        Es = glob.glob(os.path.join(ex, "*KK=10*"))
    else:
        raise NotImplementedError()


    Es = sorted(Es, key=lambda E: float(re.findall(r"overshoot=(\d.*)_", E)[0].split("_")[0]))
    Es = [E for E in Es if "overshoot=10" not in E]  # Ugly filtering
    Es_table = list(Es)
    if baselines is not None:
        Es_table += baselines
    D = pd.DataFrame(load_D(Es_table, K_smooth=5, evaltype=calculation))
    Es_777 = [E for E in Es if "777" in E]  # Ugly filtering
    Es_778 = [E for E in Es if "778" in E]  # Ugly filtering
    labels = []
    for E in Es_777:
        overshoot = re.findall(r"overshoot=(\d.*)_", E)[0].split("_")[0]
        labels.append("$\gamma={}$".format(overshoot))
    table = print_table(D)
    plot_curves(Es_777, expname, labels=labels, maxn=500, smooth=5)
    print(table)
    open(os.path.join(SAVE_DIR, "{}_table.txt".format(expname)), "w").write(table)