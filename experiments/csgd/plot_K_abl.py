#!/usr/bin/env python
"""
Run as:

python experiments/csgd/plot_K_abl.py resnet
python experiments/csgd/plot_K_abl.py scnn
"""
# -*- coding: utf-8 -*-

from experiments.csgd.utils import *
from experiments.csgd.utils import _construct_colorer
from src import RESULTS_DIR
results_dir = RESULTS_DIR

if __name__ == "__main__":
    which = sys.argv[1]

    ## Config

    if which =="resnet":
        Kabl_resnet = os.path.join(results_dir, "nsgd/Kabl_sresnet")
        ex = Kabl_resnet
        baseline_01 = glob.glob(os.path.join(results_dir, "nsgd/baselines/*resnet*lr=0.1*"))
        baseline_001 = glob.glob(os.path.join(results_dir, "nsgd/baselines/*resnet*lr=0.01*"))
        D_001 = pd.DataFrame(load_D(baseline_001, ignore_running=True, evaltype="calc")).mean()
        D_01 = pd.DataFrame(load_D(baseline_01, ignore_running=True, evaltype="calc")).mean()

        print(D_001)

        expname = "resnet_kabl"
        Es = glob.glob(os.path.join(ex, "*K*"))
        Es = sorted(Es, key=lambda E: float(re.findall(r"KK=(\d.*)", E)[0].split("_")[0]))
        Es = [E for E in Es if "KK=50" not in E]  # Ugly filtering
        Es = [E for E in Es if "KK=100" not in E]  # Ugly filtering
        # Es = [E for E in Es if "777" in E]  # Ugly filtering. TODO: FIX!
        D = pd.DataFrame(load_D(Es, K_smooth=5, evaltype="calc"))
        D = D.groupby("opt_kwargs").mean()
    elif which == "scnn":
        # Load baselines!
        Kabl_scnn = os.path.join(results_dir, "nsgd/skablcnn")
        ex = Kabl_scnn
        baseline_01 = glob.glob(os.path.join(results_dir, "nsgd/baselines/*medium*lr=0.1*"))
        baseline_001 = glob.glob(os.path.join(results_dir, "nsgd/baselines/*medium*lr=0.01*"))
        D_001 = load_D(baseline_001, ignore_running=True, evaltype="calc")[0]
        D_01 = load_D(baseline_01, ignore_running=True, evaltype="calc")[0]
        expname = "scnn_kabl"
        Es = glob.glob(os.path.join(ex, "*K*"))
        Es = sorted(Es, key=lambda E: float(re.findall(r"KK=(\d.*)", E)[0].split("_")[0]))
        Es = [E for E in Es if "KK=50" not in E]  # Ugly filtering
        Es = [E for E in Es if "KK=100" not in E]  # Ugly filtering
        # Es = [E for E in Es if "777" in E]  # Ugly filtering. TODO: FIX!

        D = pd.DataFrame(load_D(Es, K_smooth=5, evaltype="calc"))
        D = D.groupby("opt_kwargs").mean()
    else:
        raise NotImplementedError()

    print("Baselines")
    print((D_001['SN'], D_001['test_acc']))
    print((D_01['SN'], D_01['test_acc']))

    ## Plot

    for whichy in ['SN', 'test_acc', 'both', 'max_SN']:
        save_to = os.path.join(SAVE_DIR, "ablation_K_{}_{}.pdf".format(which, whichy))
        plot_against_K(whichy, save_to, D, D_01, D_001)
    Es_plot = [E for E in Es if "777" in E] # Hack filter tso avoid clutter..
    labels = []
    i = 0
    for E in Es_plot:
        overshoot = re.findall(r"KK=(\d.*)", E)[0].split("_")[0]
        if i % 2 ==0 or i==len(Es_plot)-1:
            labels.append("$K={}$".format(overshoot))
        else:
            labels.append("__nolabel__") # Declutter a bit
        i += 1
    baseline_001_777 = glob.glob(os.path.join(results_dir, "nsgd/baselines/*resnet*777*lr=0.01*"))[0]
    Es_plot.append(baseline_001_777) # Hack: it has hessian..

    labels.append("SGD(0.01)")
    cm_base = _construct_colorer(Es)
    print(Es_plot)
    def cm(x):
        if x in Es:
            return cm_base(x)
        else:
            return "black"
    plot_curves(Es_plot, expname, labels, maxn=500, smooth=5, nstart=10, cm=cm)
