# Common constants, routines and functions for plotting

fs=figsize=7
fontsize=28
labelfontsize=25
legendfontsize=26
linewidth=3

import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

import os

import re
## Formatting ticks
from matplotlib.ticker import FormatStrFormatter
import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import os
from os.path import join, basename
import json
import matplotlib.pylab as plt
import pandas as pd
from scipy.stats import *


def _construct_colorer(sorted_vals, cmap="coolwarm"):
    cm = plt.get_cmap(cmap, len(sorted_vals))

    N = float(len(sorted_vals))

    def _get_cm(val):
        return cm(sorted_vals.index(val) / N)

    return _get_cm


def _construct_colorer_lin_scale(vmin, vmax, ticks=20, cmap="coolwarm"):
    assert vmax > vmin

    cm = plt.get_cmap(cmap, ticks)

    def _get_cm(val):
        #         assert val <= vmax
        #         assert val >= vmin
        alpha = (val - vmin) / float((vmax - vmin))
        tick = int(alpha * ticks)
        tick = min(tick, ticks - 1)
        return cm(tick)

    return _get_cm


def load_HC(E):
    H = pickle.load(open(join(E, "history.pkl"), "rb"))
    C = json.load(open(join(E, "config.json")))
    return H, C


def load_HC_csv(E):
    H = pd.read_csv(join(E, "history.csv"))
    C = json.load(open(join(E, "config.json")))
    return H, C


def save_fig(path):
    plt.savefig(path, bbox_inches='tight',
        transparent=True,
        pad_inches=0)


import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc
