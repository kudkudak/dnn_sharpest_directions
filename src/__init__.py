"""
Config for project
"""

import os
from keras import backend as K
import tensorflow as tf
from src.utils.vegab import MetaSaver, SyncSmallFiles, CheckIfFinished

# Configure TF
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

# Configure paths
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))
TB_DIR = os.environ.get("TB_DIR", os.path.join(os.path.dirname(__file__), "tb"))

# Configure keras
DATA_FORMAT = "channels_last"
assert DATA_FORMAT in {"channels_last", "channels_first"}

# Configure vegab
REMOTE_RESULTS_DIR = os.environ.get("REMOTE_RESULTS_DIR", "/mnt/users/jastrzebski/local/dnnsharpest")
REMOTE_USER = os.environ.get("REMOTE_RESULTS_DIR", "jastrzebski")
REMOTE_HOST = os.environ.get("REMOTE_RESULTS_DIR", "truten.ii.uj.edu.pl")
# sync_plugin = SyncSmallFiles(user=REMOTE_USER, host=REMOTE_HOST, root_folder_src=RESULTS_DIR,
#     root_folder_dst=REMOTE_RESULTS_DIR)
vegab_plugins = [MetaSaver(), CheckIfFinished()]#, sync_plugin]

# Misc
PROJECT_DIR=os.path.join(os.path.dirname(__file__), "..")

# Configure logging
from src.utils.vegab import configure_logger
configure_logger('', log_file=None)