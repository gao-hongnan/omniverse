# type: ignore
# ruff: noqa
import gc
import os
import pickle
from typing import Callable, List, Tuple

import keras
import keras_nlp
import numpy as np
import tensorflow as tf
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
import tensorflow_datasets as tfds
import torch
from keras.saving import deserialize_keras_object
from rich.pretty import pprint
from tensorflow.data import TextLineDataset
from tensorflow.keras.utils import get_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from omnivault.transformer.config.composer import Composer
from omnivault.transformer.config.criterion import CRITERION_REGISTRY
from omnivault.transformer.config.decoder import *
from omnivault.transformer.config.generator import GeneratorConfig
from omnivault.transformer.config.optim import OPTIMIZER_REGISTRY
from omnivault.transformer.config.scheduler import SCHEDULER_REGISTRY, LambdaLRConfig
from omnivault.transformer.config.trainer import TrainerConfig
from omnivault.transformer.core.dataset import (
    construct_dummy_batch_future_masks,
    construct_dummy_batch_target_padding_masks,
)
from omnivault.transformer.core.state import State
from omnivault.transformer.core.trainer import Trainer, TrainerEvent
from omnivault.transformer.decoder.core import GPTDecoder
from omnivault.transformer.modules.attention.core import ScaledDotProductAttention
from omnivault.transformer.utils.visualization import save_plot_history

# Paths
DATA_URL: str = "https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip"
BASE_DATA_DIR: str = os.path.join(os.path.expanduser("~/.keras/datasets/"), "simplebooks/")
TRAIN_FILE: str = os.path.join(BASE_DATA_DIR, "simplebooks-92-raw/train.txt")
VALID_FILE: str = os.path.join(BASE_DATA_DIR, "simplebooks-92-raw/valid.txt")

# Data
BATCH_SIZE = 64
MIN_STRING_LEN = 512  # Strings shorter than this will be discarded
SEQ_LEN = 128  # Length of training sequences, in tokens

# Vocabulary
VOCABULARY_SERIALIZABLED: str = "./assets/vocabulary_serializabled.pkl"

# Inference
NUM_TOKENS_TO_GENERATE = 80


def download_and_extract_data(url: str) -> str:
    """Download and extract dataset from a given URL."""
    return keras.utils.get_file(origin=url, extract=True)


def prepare_dataset(file_path: str, batch_size: int, min_string_len: int, shuffle: bool = False) -> TextLineDataset:
    """Prepare the dataset by loading, filtering and batching."""
    dataset = TextLineDataset(file_path)
    dataset = dataset.filter(lambda x: tf_strings.length(x) > min_string_len)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    return dataset.batch(batch_size)


download_and_extract_data(DATA_URL)
raw_train_ds = prepare_dataset(TRAIN_FILE, BATCH_SIZE, MIN_STRING_LEN, shuffle=True)
raw_valid_ds = prepare_dataset(VALID_FILE, BATCH_SIZE, MIN_STRING_LEN, shuffle=False)


def load_vocab(vocab_file: str) -> List[str]:
    """Load the vocabulary from a serialized file."""
    try:
        with open(vocab_file, "rb") as f:
            vocab_config = pickle.load(f)
    except FileNotFoundError:
        raise Exception(f"Vocabulary file not found: {vocab_file}")
    except Exception as e:
        raise Exception(f"Error loading vocabulary: {str(e)}")

    return deserialize_keras_object(vocab_config)


vocab = load_vocab(VOCABULARY_SERIALIZABLED)


def create_tokenizer_and_packer(
    vocab: dict, seq_len: int
) -> Tuple[keras_nlp.tokenizers.WordPieceTokenizer, keras_nlp.layers.StartEndPacker]:
    """
    Create a WordPiece tokenizer and a StartEnd packer using the given vocabulary and sequence length.
    """
    tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
        vocabulary=vocab,
        sequence_length=seq_len,
        lowercase=True,
    )

    start_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=seq_len,
        start_value=tokenizer.token_to_id("[BOS]"),
    )

    return tokenizer, start_packer


tokenizer, start_packer = create_tokenizer_and_packer(vocab, SEQ_LEN)


def preprocess(inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Preprocess the input data by tokenizing and then packing with start tokens.

    Parameters
    ----------
    inputs : tf.Tensor
        Input text data in tensor format.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        A tuple containing:
        - features: Tokenized and packed input sequences.
        - labels: Tokenized output sequences (identical to `outputs` from tokenizer).
    """
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels


def prepare_dataset(
    dataset: tf.data.Dataset, preprocess_fn: Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]
) -> tf.data.Dataset:
    """
    Prepares the dataset by applying preprocessing, batching, and prefetching.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The raw dataset to be processed.
    preprocess_fn : callable
        The preprocessing function to be applied to the dataset.
    batch_size : int
        The size of batches to divide the dataset into.

    Returns
    -------
    tf.data.Dataset
        The processed and batched dataset.
    """
    return dataset.map(preprocess_fn, num_parallel_calls=tf_data.AUTOTUNE).prefetch(tf_data.AUTOTUNE)


train_ds = prepare_dataset(raw_train_ds, preprocess)
valid_ds = prepare_dataset(raw_valid_ds, preprocess)

train_ds_as_numpy = list(tfds.as_numpy(train_ds))
valid_ds_as_numpy = list(tfds.as_numpy(valid_ds))


def flatten_dataset(dataset: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Flatten each pair of source and target arrays in the dataset from shape [B, L] to [B*L].

    Parameters
    ----------
    dataset : List[Tuple[np.ndarray, np.ndarray]]
        The dataset to be flattened, consisting of tuples of source and target numpy arrays.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        The flattened dataset, where each source and target array is reshaped to [B*L].
    """
    flattened_data = [
        (source[i].reshape(-1), target[i].reshape(-1))
        for source, target in tqdm(dataset)
        for i in range(source.shape[0])
    ]
    return flattened_data


train_ds_as_numpy = flatten_dataset(train_ds_as_numpy)
valid_ds_as_numpy = flatten_dataset(valid_ds_as_numpy)


class TFDatasetWrapper(Dataset):
    def __init__(self, tf_dataset_as_numpy: List[int]) -> None:
        super().__init__()
        self.tf_dataset_as_numpy = tf_dataset_as_numpy

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source, target = self.tf_dataset_as_numpy[index]

        # Convert from list to tensors
        source = torch.tensor(source, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        return source, target

    def __len__(self) -> int:
        return len(self.tf_dataset_as_numpy)


def custom_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Unzip the batch into separate lists for sources and targets
    sources, targets = zip(*batch)

    # Convert lists to tensors
    sources = torch.stack(sources)
    targets = torch.stack(targets)

    # Get batch size and sequence length
    batch_size, seq_len = targets.size(0), targets.size(1)

    # Generate dummy future masks and target padding masks
    future_masks = construct_dummy_batch_future_masks(batch_size, seq_len)
    target_padding_masks = construct_dummy_batch_target_padding_masks(batch_size, seq_len)

    return sources, targets, future_masks, target_padding_masks


train_ds_pytorch = TFDatasetWrapper(train_ds_as_numpy)
valid_ds_pytorch = TFDatasetWrapper(valid_ds_as_numpy)

# Create a PyTorch data loader
train_loader = DataLoader(
    train_ds_pytorch, shuffle=False, batch_size=16, collate_fn=custom_collate_fn
)  # shuffled by tensorflow earlier
valid_loader = DataLoader(
    valid_ds_pytorch, shuffle=False, batch_size=32, collate_fn=custom_collate_fn
)  # shuffled by tensorflow earlier

masked_self_attention_mha_config = MultiHeadedAttentionConfig(
    attention=ScaledDotProductAttention(), d_model=256, H=4, dropout=0.2
)

feed_forward_config = PositionwiseFeedForwardConfig(
    d_model=256, d_ff=256 * 4, activation=nn.GELU(approximate="tanh"), dropout=0.2, bias=True
)

add_norm_config_1 = AddNormConfig(feature_dim=256, dropout=0.2)
add_norm_config_2 = AddNormConfig(feature_dim=256, dropout=0.2)

# Create DecoderBlockConfig
decoder_block_config = DecoderBlockConfig(
    masked_self_attention_mha=masked_self_attention_mha_config,
    feed_forward=feed_forward_config,
    add_norm_1=add_norm_config_1,
    add_norm_2=add_norm_config_2,
)

vocab_size = len(vocab)
context_length = SEQ_LEN

# Create the overall DecoderConfig
model_config = DecoderConfig(
    d_model=256,
    vocab_size=vocab_size,
    context_length=context_length,
    num_decoder_blocks=2,
    dropout=0.2,
    decoder_block=decoder_block_config,
)

GRADIENT_ACCUMULATION_STEPS = 4

optimizer_config_cls = OPTIMIZER_REGISTRY["torch.optim.Adam"]
optimizer_pydantic_config = optimizer_config_cls(name="torch.optim.Adam", lr=0.00025 * GRADIENT_ACCUMULATION_STEPS)

criterion_config_cls = CRITERION_REGISTRY["torch.nn.CrossEntropyLoss"]
criterion_pydantic_config = criterion_config_cls(name="torch.nn.CrossEntropyLoss")

scheduler_config_cls = SCHEDULER_REGISTRY["torch.optim.lr_scheduler.CosineAnnealingLR"]
scheduler_pydantic_config = scheduler_config_cls(name="torch.optim.lr_scheduler.CosineAnnealingLR", T_max=6)

trainer_config = TrainerConfig(
    device="cuda",
    max_epochs=6,
    eval_every_n_steps=10000,
    log_every_n_steps=10000,
    use_amp=True,
    autocast_config={"enabled": True, "dtype": torch.float16, "cache_enabled": True},
    scaler_config={
        "enabled": True,
        "init_scale": 2.0**16,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
    },
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    clip_grad_norm={"max_norm": 1.0, "norm_type": 2.0, "error_if_nonfinite": False, "foreach": None},
    apply_weight_decay_to_different_param_groups=False,
    step_scheduler_on_batch_or_epoch="epoch",
    save_dir="./data/simplybooks92/checkpoints",
    save_every_epoch=False,
    save_best_only=True,
    monitor="valid_this_epoch_average_loss",
    mode="min",
)

generator_config = GeneratorConfig(temperature=1.0, max_tokens=80, greedy=False, top_k=10, top_p=None)

composer = Composer(
    model=model_config,
    optimizer=optimizer_pydantic_config,
    criterion=criterion_pydantic_config,
    scheduler=scheduler_pydantic_config,
    trainer=trainer_config,
    generator=generator_config,
)
composer.pretty_print()

model = GPTDecoder(model_config).to(composer.trainer.device)
optimizer = optimizer_pydantic_config.build(params=model.parameters())
criterion = criterion_pydantic_config.create_instance()

state = State(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=None,
    # vocabulary=vocab,
    # tokenizer=tokenizer,
)

device = composer.trainer.device
trainer = Trainer(
    state=state,
    composer=composer,
    logger=None,
    device=device,  # type: ignore[arg-type]
)

_trained_state = trainer.fit(train_loader=train_loader, valid_loader=valid_loader)
# _trained_state.pretty_print()
history = _trained_state.history
_ = save_plot_history(history, plot=False, save_path=f"{composer.trainer.save_dir}/history.png")

# Delete variables
del optimizer, state, trainer

# Collect garbage
gc.collect()

# Clear GPU memory cache
torch.cuda.empty_cache()

import tensorflow as tf

from omnivault.transformer.utils.reproducibility import seed_all

seed_all()
tf.random.set_seed(1992)

model.eval() # important!

# The "packer" layers adds the [BOS] token for us.
prompt_tokens = start_packer(tokenizer([""]))
prompt_tokens

def next(prompt, cache, index):
    prompt = prompt.numpy()
    prompt = torch.from_numpy(prompt).to(composer.trainer.device)
#     print(prompt)
#     print(prompt.shape)
#     print(model(prompt))
#     print(index)

    index = int(index)
    with torch.no_grad():
        logits = model(prompt)[:, index - 1, :]
    logits = logits.detach().cpu().numpy()
    logits = tf.convert_to_tensor(logits)
    # Ignore hidden states for now; only needed for contrastive search.
    hidden_states = None
    return logits, hidden_states, cache

sampler = keras_nlp.samplers.GreedySampler()
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,  # Start sampling immediately after the [BOS] token.
)
txt = tokenizer.detokenize(output_tokens)
print(f"Greedy search generated text: \n{txt}\n")

sampler = keras_nlp.samplers.TopKSampler(k=10)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-K search generated text: \n{txt}\n")

sampler = keras_nlp.samplers.TopPSampler(p=0.5)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-P search generated text: \n{txt}\n")