import numpy as np
import pandas as pd
from torchvision.transforms import *

BATCH_SIZE = 128
N_WORKERS = 14
H5_PATH = "/home/francesco/Desktop/carino/vaevictis/data_many_dist_fixed_step.h5"
GROUP = np.arange(1)
EPOCHES = 100
TRAIN = False

df = pd.read_hdf()
