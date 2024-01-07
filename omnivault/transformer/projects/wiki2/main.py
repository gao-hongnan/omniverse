from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from rich.pretty import pprint
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
from omnivault.transformer.decoder.core import GPTDecoder
from omnivault.transformer.config.decoder import *

import torch
from torch import nn, Tensor
from omnivault.transformer.modules.attention.core import ScaledDotProductAttention
from torch.utils.data import dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import TensorDataset
from omnivault.transformer.core.dataset import construct_dummy_batch_future_masks, construct_dummy_batch_target_padding_masks
from omnivault.transformer.utils.device import get_device

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pprint(DEVICE)


###############################################################################
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
pprint(train_data)
pprint(train_data.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

pprint(train_data)
pprint(train_data.shape)

bptt = 35
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

###############################################################################

# Create individual component configurations
masked_self_attention_mha_config = MultiHeadedAttentionConfig(
     attention=ScaledDotProductAttention(),
    d_model=200, H=2, dropout=0.1
)

feed_forward_config = PositionwiseFeedForwardConfig(
    d_model=200, d_ff=200, activation=nn.GELU(approximate="tanh"), dropout=0.1, bias=True
)

add_norm_config_1 = AddNormConfig(feature_dim=200, dropout=0.1)
add_norm_config_2 = AddNormConfig(feature_dim=200, dropout=0.1)

# Create DecoderBlockConfig
decoder_block_config = DecoderBlockConfig(
    masked_self_attention_mha=masked_self_attention_mha_config,
    feed_forward=feed_forward_config,
    add_norm_1=add_norm_config_1,
    add_norm_2=add_norm_config_2,
)

vocab_size = len(vocab)

max_seq_len = 35
assert max_seq_len == bptt

# Create the overall DecoderConfig
model_config = DecoderConfig(
    d_model=200,
    vocab_size=vocab_size,
    context_length=max_seq_len,
    num_decoder_blocks=2,
    dropout=0.1,
    decoder_block=decoder_block_config,
)

model = GPTDecoder(model_config).to(DEVICE)

model_size = sum([p.numel() for p in model.parameters()])
print(f'model_size: {model_size}, train_set_size: {len(train_data)}')

import time

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        output = model(data)
        output_flat = output.view(-1, vocab_size)
        loss = criterion(output_flat, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            output = model(data)
            output_flat = output.view(-1, vocab_size)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


best_val_loss = float('inf')
epochs = 3

with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model)
        val_loss = evaluate(model, val_data)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)

        scheduler.step()
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states

test_loss = evaluate(model, test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)