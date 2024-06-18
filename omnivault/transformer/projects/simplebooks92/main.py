# type: ignore
import os
import pickle
from pathlib import Path
from typing import Any, Callable, List, Tuple

import keras
import keras_nlp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from keras.saving import deserialize_keras_object, serialize_keras_object
from keras_nlp.samplers import Sampler
from numpy.typing import NDArray
from tensorflow.python.framework.ops import EagerTensor
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from omnivault.transformer.config.composer import Composer
from omnivault.transformer.config.criterion import CRITERION_REGISTRY
from omnivault.transformer.config.decoder import (
    AddNormConfig,
    DecoderBlockConfig,
    DecoderConfig,
    MultiHeadedAttentionConfig,
    PositionwiseFeedForwardConfig,
)
from omnivault.transformer.config.generator import GeneratorConfig
from omnivault.transformer.config.optim import OPTIMIZER_REGISTRY
from omnivault.transformer.config.scheduler import SCHEDULER_REGISTRY
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
from omnivault.utils.reproducibility.seed import seed_all
from omnivault.utils.torch_utils.cleanup import purge_global_scope

seed_all(set_torch_deterministic=False)  # set to False since it may cause a slight increase in memory usage
tf.random.set_seed(1992)

# Paths
DATA_URL: str = "https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip"
BASE_DATA_DIR: str = os.path.join(os.path.expanduser("~/.keras/datasets/"), "simplebooks/")
TRAIN_FILE: str = os.path.join(BASE_DATA_DIR, "simplebooks-92-raw/train.txt")
VALID_FILE: str = os.path.join(BASE_DATA_DIR, "simplebooks-92-raw/valid.txt")

TRAIN_DATASET_PICKLED_PATH = "/kaggle/input/simplebooks-92/train_dataset.pkl"
VALID_DATASET_PICKLED_PATH = "/kaggle/input/simplebooks-92/valid_dataset.pkl"
SAVE_DATASET_AS_PICKLE = False
LOAD_DATASET_FROM_PICKLE = True
COMPUTE_WORD_PIECE_VOCABULARY = False


# Data
BATCH_SIZE = 64
MIN_STRING_LEN = 512  # Strings shorter than this will be discarded
SEQ_LEN = 128  # Length of training sequences, in tokens
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent

# Vocabulary
VOCABULARY_SERIALIZABLED: str = CURRENT_SCRIPT_DIR / "./assets/vocabulary_serializabled.pkl"
VOCAB_SIZE = 5000  # Limits parameters in model.

# Training
GRADIENT_ACCUMULATION_STEPS = 4
BASE_LR = 0.00025
BASE_TRAIN_BATCH_SIZE = 16
NUM_GPUS = 1

# Inference
NUM_TOKENS_TO_GENERATE = 80


def download_and_extract_data(url: str) -> str:
    """Download and extract dataset from a given URL."""
    return keras.utils.get_file(origin=url, extract=True)


def load_and_prepare_dataset(
    file_path: str, batch_size: int, min_string_len: int, shuffle: bool = False
) -> tf.data.TextLineDataset:
    """Prepare the dataset by loading, filtering and batching."""
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.filter(lambda x: tf.strings.length(x) > min_string_len)
    dataset = dataset.batch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    return dataset


def load_vocab(vocab_file: str) -> List[str]:
    """Load the vocabulary from a serialized file."""
    try:
        with open(vocab_file, "rb") as f:
            vocab_config = pickle.load(f)
    except FileNotFoundError:
        raise Exception(f"Vocabulary file not found: {vocab_file}") from None
    except Exception as err:
        raise Exception(f"Error loading vocabulary: {str(err)}") from err

    return deserialize_keras_object(vocab_config)


def create_tokenizer(vocabulary: List[str], seq_len: int) -> keras_nlp.tokenizers.WordPieceTokenizer:
    """
    Create a WordPiece tokenizer and a StartEnd packer using the given
    vocabulary and sequence length.
    """
    tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
        vocabulary=vocabulary,
        sequence_length=seq_len,
        lowercase=True,
    )

    return tokenizer


def create_start_packer(tokenizer: keras_nlp.tokenizers.WordPieceTokenizer) -> keras_nlp.layers.StartEndPacker:
    """Create a StartEnd packer using the given tokenizer."""
    return keras_nlp.layers.StartEndPacker(
        sequence_length=tokenizer.sequence_length,
        start_value=tokenizer.token_to_id("[BOS]"),
    )


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


def prepare_and_preprocess_dataset(
    dataset: tf.data.Dataset, preprocess_fn: Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]
) -> tf.data.Dataset:
    """
    Prepares the dataset by applying preprocessing, batching, and prefetching.

    Parameters
    ----------
    dataset: tf.data.Dataset
        The raw dataset to be processed.
    preprocess_fn: callable
        The preprocessing function to be applied to the dataset.
    batch_size: int
        The size of batches to divide the dataset into.

    Returns
    -------
    tf.data.Dataset
        The processed and batched dataset.
    """
    return dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


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
    sources, targets = zip(*batch)

    sources = torch.stack(sources)
    targets = torch.stack(targets)

    batch_size, seq_len = targets.size(0), targets.size(1)

    future_masks = construct_dummy_batch_future_masks(batch_size, seq_len)
    target_padding_masks = construct_dummy_batch_target_padding_masks(batch_size, seq_len)

    return sources, targets, future_masks, target_padding_masks


def next(prompt: tf.Tensor, cache: Tuple[Any, ...], index: EagerTensor) -> Tuple[tf.Tensor, None, Tuple[Any, ...]]:
    prompt: NDArray[np.int32] = prompt.numpy()
    prompt: torch.LongTensor = torch.from_numpy(prompt).to(composer.trainer.device)

    index: int = int(index)
    with torch.no_grad():
        logits: torch.Tensor = model(prompt)[:, index - 1, :]
    logits: NDArray[float] = logits.detach().cpu().numpy()
    logits: tf.Tensor = tf.convert_to_tensor(logits)
    # Ignore hidden states for now; only needed for contrastive search.
    hidden_states = None
    return logits, hidden_states, cache


@torch.no_grad()
def generate_on_train_epoch_end(trainer: Trainer) -> None:
    """Evaluate and generate on train batch end."""

    model = trainer.model
    model.eval()

    def get_samplers() -> List[Sampler]:
        samplers = [
            keras_nlp.samplers.GreedySampler(temperature=1.0),
            keras_nlp.samplers.BeamSampler(num_beams=10, temperature=1.0),
            keras_nlp.samplers.RandomSampler(temperature=1.0),
            keras_nlp.samplers.TopKSampler(k=20, temperature=0.7),
            keras_nlp.samplers.TopPSampler(p=0.5, temperature=0.7, k=20),
        ]
        return samplers

    samplers = get_samplers()
    tokenizer = trainer.state.tokenizer

    start_packer = create_start_packer(tokenizer)
    # The "packer" layers adds the [BOS] token for us.
    prompt_tokens = start_packer(tokenizer([""]))

    for sampler in samplers:
        generated_tokens = sampler(
            next=next,
            prompt=prompt_tokens,
            index=1,  # Start sampling immediately after the [BOS] token.
        )
        generated_tokens_decoded = tokenizer.detokenize(generated_tokens)
        sampler_name = sampler.__class__.__name__
        trainer.logger.info("%s search Generated text %s", sampler_name, generated_tokens_decoded)
    # Revert model to training mode
    model.train()


if __name__ == "__main__":
    download_and_extract_data(DATA_URL)
    raw_train_ds = load_and_prepare_dataset(TRAIN_FILE, BATCH_SIZE, MIN_STRING_LEN, shuffle=True)
    raw_valid_ds = load_and_prepare_dataset(VALID_FILE, BATCH_SIZE, MIN_STRING_LEN, shuffle=False)

    if COMPUTE_WORD_PIECE_VOCABULARY:
        vocabulary = keras_nlp.tokenizers.compute_word_piece_vocabulary(
            raw_train_ds,
            vocabulary_size=VOCAB_SIZE,
            lowercase=True,
            reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
        )
        with open(VOCABULARY_SERIALIZABLED, "wb") as f:
            pickle.dump(serialize_keras_object(vocabulary), f)
    else:
        vocabulary = load_vocab(VOCABULARY_SERIALIZABLED)

    tokenizer = create_tokenizer(vocabulary, SEQ_LEN)
    start_packer = create_start_packer(tokenizer)

    train_ds = prepare_and_preprocess_dataset(raw_train_ds, preprocess)
    valid_ds = prepare_and_preprocess_dataset(raw_valid_ds, preprocess)

    if LOAD_DATASET_FROM_PICKLE:
        with open(TRAIN_DATASET_PICKLED_PATH, "rb") as file:
            train_ds_as_list_of_numpy = pickle.load(file)

        with open(VALID_DATASET_PICKLED_PATH, "rb") as file:
            valid_ds_as_list_of_numpy = pickle.load(file)

    if SAVE_DATASET_AS_PICKLE and not LOAD_DATASET_FROM_PICKLE:
        train_ds_as_list_of_numpy = list(tfds.as_numpy(train_ds))
        valid_ds_as_list_of_numpy = list(tfds.as_numpy(valid_ds))

        with open(TRAIN_DATASET_PICKLED_PATH, "wb") as file:
            pickle.dump(train_ds_as_list_of_numpy, file)

        with open(VALID_DATASET_PICKLED_PATH, "wb") as file:
            pickle.dump(valid_ds_as_list_of_numpy, file)

    train_ds_as_numpy = flatten_dataset(train_ds_as_list_of_numpy)
    valid_ds_as_numpy = flatten_dataset(valid_ds_as_list_of_numpy)

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

    # Create the overall DecoderConfig
    model_config = DecoderConfig(
        d_model=256,
        vocab_size=len(vocabulary),
        context_length=SEQ_LEN,
        num_decoder_blocks=2,
        dropout=0.2,
        decoder_block=decoder_block_config,
    )

    optimizer_config_cls = OPTIMIZER_REGISTRY["torch.optim.Adam"]
    optimizer_pydantic_config = optimizer_config_cls(name="torch.optim.Adam", lr=BASE_LR * GRADIENT_ACCUMULATION_STEPS)

    criterion_config_cls = CRITERION_REGISTRY["torch.nn.CrossEntropyLoss"]
    criterion_pydantic_config = criterion_config_cls(name="torch.nn.CrossEntropyLoss")

    scheduler_config_cls = SCHEDULER_REGISTRY["torch.optim.lr_scheduler.CosineAnnealingLR"]
    scheduler_pydantic_config = scheduler_config_cls(name="torch.optim.lr_scheduler.CosineAnnealingLR", T_max=8)

    trainer_config = TrainerConfig(
        device="cuda",
        max_epochs=8,
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
    model = GPTDecoder(model_config)
    model = model.to(device=composer.trainer.device, dtype=next(model.parameters()).dtype, non_blocking=True)

    optimizer = optimizer_pydantic_config.build(params=model.parameters())
    criterion = criterion_pydantic_config.create_instance()

    composer.scheduler = scheduler_pydantic_config
    scheduler = scheduler_pydantic_config.build(optimizer=optimizer)

    composer.pretty_print()

    state = State(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        vocabulary=vocabulary,
        tokenizer=tokenizer,
    )

    device = composer.trainer.device
    trainer = Trainer(
        state=state,
        composer=composer,
        logger=None,
        device=device,  # type: ignore[arg-type]
    )
    trainer.add_callback(TrainerEvent.ON_TRAIN_EPOCH_END.value, generate_on_train_epoch_end)

    _trained_state = trainer.fit(train_loader=train_loader, valid_loader=valid_loader)
    _trained_state.pretty_print()
    history = _trained_state.history
    _ = save_plot_history(history, plot=False, save_path=f"{composer.trainer.save_dir}/history.png")

    purge_global_scope(["optimizer", "state", "trainer"])
