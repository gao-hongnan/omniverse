from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from rich.pretty import pprint
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import TensorDataset
from omnivault.transformer.core.dataset import construct_dummy_batch_future_masks, construct_dummy_batch_target_padding_masks

def get_data_loader(data: Tensor, batch_size: int) -> DataLoader:
    """
    Creates a DataLoader from the given data.

    Arguments:
        data: Tensor, already batched data of shape ``[seq_len, batch_size]``
        batch_size: int, batch size

    Returns:
        DataLoader
    """
    # Since data is already batched, we use a batch size of 1 for the DataLoader
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=1, shuffle=False)



train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# ``train_iter`` was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)
train_dataset = TensorDataset(data_process(train_iter))
val_dataset = TensorDataset(data_process(val_iter))
test_dataset = TensorDataset(data_process(test_iter))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import DataLoader

batch_size = 20
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for i, x in enumerate(train_loader):
    print(x)
    print(x[0].shape)
    break
# def batchify(data: Tensor, bsz: int) -> Tensor:
#     """Divides the data into ``bsz`` separate sequences, removing extra elements
#     that wouldn't cleanly fit.

#     Arguments:
#         data: Tensor, shape ``[N]``
#         bsz: int, batch size

#     Returns:
#         Tensor of shape ``[N // bsz, bsz]``
#     """
#     seq_len = data.size(0) // bsz
#     data = data[:seq_len * bsz]
#     data = data.view(bsz, seq_len).t().contiguous()
#     return data.to(device)

# batch_size = 20
# eval_batch_size = 10
# train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
# val_data = batchify(val_data, eval_batch_size)
# test_data = batchify(test_data, eval_batch_size)
# print(train_data.shape)