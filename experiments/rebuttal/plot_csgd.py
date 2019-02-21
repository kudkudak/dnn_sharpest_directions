#!/usr/bin/env python
"""
(Living script)

Run as:

python experiments/rebuttal/plot_csgd.py imdb 0
python experiments/rebuttal/plot_csgd.py imdb 1

"""
# -*- coding: utf-8 -*-

from src.utils.plotting_preamble import *
from experiments.csgd.utils import _construct_colorer
from experiments.csgd.utils import *

from src import RESULTS_DIR
results_dir = RESULTS_DIR

SAVE_DIR = os.path.join(PROJECT_DIR, "experiments", "rebuttal", "rebuttal_csgd")
os.system("mkdir -p " + SAVE_DIR)

if __name__ == "__main__":
    which = sys.argv[1]
    eval = int(sys.argv[2])

    if which == "imdb":
        ex = os.path.join(results_dir, "rebuttal/csgd_imdb7")
        expname = "csgd_imdb"
        Es = glob.glob(os.path.join(ex, "*KK=1*bs=8"))
    else:
        raise NotImplementedError()

    Es = sorted(Es, key=lambda E: float(re.findall(r"overshoot=(\d.*)_", E)[0].split("_")[0]))
    labels = []
    for E in Es:
        overshoot = re.findall(r"overshoot=(\d.*)_", E)[0].split("_")[0]
        labels.append("$\gamma={}$".format(overshoot))

    logger.info("Collected:")
    logger.info(Es)

    # TODO: Run evaluate (so we have proper frobneius norm etc)

    if eval == 1:
        D = pd.DataFrame(load_D(Es, K_smooth=5, evaltype="calc"))
        print(print_table(D, early_val_acc=10, sharpness="SN"))

    plot_curves(Es, expname, labels=labels, maxn=120, smooth=1, save_dir=SAVE_DIR, plot_frobenius_norm=False)
