import math
import os
import random
import re
import string
import time
from tempfile import TemporaryDirectory
from typing import *
from typing import Tuple
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
import torch
from rich.pretty import pprint
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator

from omnivault.transformer.config.decoder import *
from omnivault.transformer.core.dataset import (
    construct_dummy_batch_future_masks,
    construct_dummy_batch_target_padding_masks,
)
import tiktoken
from omnivault.transformer.decoder.core import GPTDecoder
from omnivault.transformer.modules.attention.core import ScaledDotProductAttention
from omnivault.transformer.utils.device import get_device
from omnivault.transformer.utils.reproducibility import seed_all

# Directories for training and testing data
data_dir = "./data/aclImdb"
directories = ["aclImdb/train/pos", "aclImdb/train/neg", "aclImdb/test/pos", "aclImdb/test/neg"]
directories = [os.path.join(data_dir, dir) for dir in directories]

def get_filenames(directories):
    filenames = []
    for dir in directories:
        for f in os.listdir(dir):
            filenames.append(os.path.join(dir, f))
    return filenames


all_filenames = get_filenames(directories)
random.shuffle(all_filenames)

# Splitting the data into training, validation, and testing sets
validation_split = 0.1  # 10% for validation
test_split = 0.1  # 10% for testing

val_count = int(len(all_filenames) * validation_split)
test_count = int(len(all_filenames) * test_split)

validation_filenames = all_filenames[:val_count]
test_filenames = all_filenames[val_count : val_count + test_count]
train_filenames = all_filenames[val_count + test_count :]

print(f"Total files: {len(all_filenames)}")
print(f"Training files: {len(train_filenames)}")
print(f"Validation files: {len(validation_filenames)}")
print(f"Testing files: {len(test_filenames)}")

print(tiktoken.list_encoding_names())
encoding = tiktoken.get_encoding("gpt2")

train_data = []
train_dataset = []
for filename in train_filenames:
    with open(filename, "r", encoding="utf-8") as file:
        train_data.append(file.read())
pprint(train_data[0])
for data in tqdm(train_data):
    train_dataset.append(encoding.encode(data, disallowed_special=()))

pprint(encoding.decode(train_dataset[0]))