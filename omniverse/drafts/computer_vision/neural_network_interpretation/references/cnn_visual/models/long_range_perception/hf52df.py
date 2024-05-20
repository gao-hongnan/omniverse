import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import *

BATCH_SIZE = 128
N_WORKERS = 14
H5_PATH = "/home/francesco/Desktop/carino/vaevictis/data_many_dist_fixed_step.h5"
GROUP = np.arange(1)
EPOCHES = 100
TRAIN = False

df = pd.read_hdf()
