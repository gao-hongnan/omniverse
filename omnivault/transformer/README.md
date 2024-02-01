# Decoder

-   [Decoder](#decoder)
    -   [Overview](#overview)
    -   [Setup and Installation](#setup-and-installation)
        -   [Step 1: Clone the Repository](#step-1-clone-the-repository)
        -   [Step 2: Create Virtual Environment](#step-2-create-virtual-environment)
        -   [Step 3: Install Dependencies](#step-3-install-dependencies)
    -   [Training Techniques](#training-techniques)
        -   [Mixed Precision, Gradient Scaling and Gradient Accumulation](#mixed-precision-gradient-scaling-and-gradient-accumulation)
        -   [Improving Performance](#improving-performance)
    -   [Adder](#adder)
        -   [Composer (Configuration)](#composer-configuration)
        -   [State](#state)
        -   [Some Quirks of the Adder Project](#some-quirks-of-the-adder-project)
        -   [Experiments](#experiments)
            -   [Run 1. CPU Bound 3 Epochs (Debug)](#run-1-cpu-bound-3-epochs-debug)
            -   [Run 2. CPU Bound 20 Epochs](#run-2-cpu-bound-20-epochs)
            -   [Run 3. CPU Bound 20 Epochs with Automatic Mixed Precision](#run-3-cpu-bound-20-epochs-with-automatic-mixed-precision)
            -   [Run 4. CPU Bound 20 Epochs with Automatic Mixed Precision and Gradient Scaler](#run-4-cpu-bound-20-epochs-with-automatic-mixed-precision-and-gradient-scaler)
            -   [Run 5. GPU Bound 30 Epochs with Automatic Mixed Precision and Gradient Scaler](#run-5-gpu-bound-30-epochs-with-automatic-mixed-precision-and-gradient-scaler)
        -   [Run 6: GPU Bound 30 Epochs with Automatic Mixed Precision, Gradient Scaler and Gradient Accumulation](#run-6-gpu-bound-30-epochs-with-automatic-mixed-precision-gradient-scaler-and-gradient-accumulation)
            -   [Generalization](#generalization)

## Overview

This projects implements a decoder-only transformer model with GPT like
architecture. This project is for educational purposes only and is not intended
to be used for production.

This project is not possible without Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) implementation, as it provides a
source of truth of how GPT is actually implemented.

## Setup and Installation

We assume a macOS setup for this project. The steps for other operating systems
may vary.

### Step 1: Clone the Repository

Clone the project repository to your local machine using the following command:

```bash
git clone --branch dev https://github.com/gao-hongnan/omniverse.git
cd omniverse
```

### Step 2: Create Virtual Environment

We recommend using a virtual environment to run this project. To create a
virtual environment, run the following command:

```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the project dependencies using the following command:

```bash
(venv) $ pip install -r requirements.txt
```

and to develop the project, install the development dependencies using:

```bash
(venv) $ pip install -r requirements-dev.txt
```

## Training Techniques

### Mixed Precision, Gradient Scaling and Gradient Accumulation

For Mixed Precision experiments, see the
[the playbook here](https://pytorch.org/docs/stable/notes/amp_examples.html).

### Improving Performance

See
[Improving Performance](https://pytorch-lightning.readthedocs.io/en/0.10.0/performance.html)

## Adder

The Adder project implements a training pipeline for the Adder dataset, inspired
by Andrej Karpathy's
[Adder project](https://github.com/karpathy/minGPT/blob/master/projects/adder/readme.md).
It mainly trains on a dataset of 2-digit addition problems, and the model is
expected to predict the result of the addition problem. It seems trivial, but
note that GPT models are after all, language models, and not well trained on
arithmetic problems. This is why ChatGPT sometimes hallucinates and gives wrong
answers to simple arithmetic problems.

### Composer (Configuration)

The core configuration for the project is defined in the
`./projects/adder/config.yaml` file. This YAML file includes various settings
such as constants, logger configurations, data specifications, model parameters,
optimizer details, and more.

Here is a snippet of the configuration file:

```yaml
constants:
    NUM_DIGITS: 2
    TOKENS:
        - "0"
        - "1"
        - "2"
        - "3"
        - "4"
        - "5"
        - "6"
        - "7"
        - "8"
        - "9"
        - "+"
        - "*"
        - "-"
        - "="
        - "<BOS>"
        - "<EOS>"
        - "<PAD>"
        - "<UNK>"
logger:
    log_file: "decoder.log"
    module_name: null
    propagate: false
    log_root_dir: "./data/adder/logs"
    rich_handler_config:
        level: "INFO"
        show_level: true
        show_path: true
        show_time: true
        rich_tracebacks: true
        markup: true
        log_time_format: "[%Y-%m-%d %H:%M:%S]"
global_:
    seed: 42
    debug: false
    debug_samples: null
data:
    context_length: 11
    dataset_name: adder_dataset
    dataset_size: 10000
    dataset_path: ./data/adder/adder_dataset.txt
    dataset_dir: ./data/adder
    dataset_url: https://raw.githubusercontent.com/gao-hongnan/omniverse/dev/omnivault/transformer/projects/adder/assets/adder_dataset.txt
    split:
        - 0.7
        - 0.2
        - 0.1
    collate_fn: # The collate function config.
        batch_first: true
        pad_token_id: 16 # TODO: `pad_token_id` should be interpolated from `MaybeConstant`.
    train_loader:
        batch_size: 32
        shuffle: true
        num_workers: 0
        pin_memory: false
        drop_last: false
    valid_loader:
        batch_size: 32
        shuffle: false
        num_workers: 0
        pin_memory: false
        drop_last: false
    test_loader:
        batch_size: 128
        shuffle: false
        num_workers: 0
        pin_memory: false
        drop_last: false
model:
    d_model: 128
    vocab_size: ??? # MISSING so need fill up later
    context_length: ${data.context_length}
    num_decoder_blocks: 2
    dropout: 0.1
    decoder_block:
        masked_self_attention_mha:
            attention:
                _target_: omnivault.transformer.modules.attention.core.ScaledDotProductAttention
            d_model: ${model.d_model}
            H: 4
            dropout: 0.1
        feed_forward:
            d_model: ${model.d_model}
            d_ff: 256
            activation:
                _target_: torch.nn.GELU
                approximate: "tanh"
            dropout: 0.1
            bias: true
        add_norm_1:
            feature_dim: ${model.d_model}
            dropout: 0.1
        add_norm_2:
            feature_dim: ${model.d_model}
            dropout: 0.1
optimizer:
    name: "torch.optim.Adam"
    lr: 0.2
    betas:
        - 0.9
        - 0.98
    eps: 1e-9
    weight_decay: 0.0
criterion:
    name: "torch.nn.CrossEntropyLoss"
    ignore_index: 16
    reduction: "mean"
scheduler:
    name: "torch.optim.lr_scheduler.LambdaLR"
trainer:
    device: "auto"
    max_epochs: 2
    log_every_n_steps: 100
    eval_every_n_steps: 4
    step_scheduler_on_batch_or_epoch: "epoch"
    use_amp: false
    autocast_config:
        enabled: false
    scaler_config:
        enabled: false
        init_scale: 65536.0
        growth_factor: 2.0
        backoff_factor: 0.5
        growth_interval: 2000
    gradient_accumulation_steps: 1
    clip_grad_norm:
        {
            max_norm: 1.0,
            norm_type: 2.0,
            error_if_nonfinite: false,
            "foreach": null,
        }
    apply_weight_decay_to_different_param_groups: false
    save_dir: ./data/adder/checkpoints
    save_every_epoch: false
    save_best_only: true
    monitor: "valid_this_epoch_average_loss"
    mode: "min"
generator:
    max_tokens: 4
    temperature: 1.0
    greedy: true
    top_k: null
    top_p: null
```

This configuration is used to set up the entire training pipeline, specifying
details such as dataset parameters, model architecture, training
hyperparameters, and more.

The `Composer` object is a dynamic configuration manager that combines various
configurations into a single accessible object. It allows easy access to any
configuration parameter from any part of the codebase by calling
`Composer().<config_name>`. It is not a singleton and is mutable, enabling
runtime modifications for techniques like hyperparameter tuning.

To run the project with the default configuration, use the following command:

```bash
(venv) $ python omnivault/transformer/projects/adder/main.py \
            omnivault/transformer/projects/adder/config.yaml
```

and you should see the following output:

```python
Composer(
│   constants=MaybeConstant(NUM_DIGITS=2, TOKENS=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '*', '-', '=', '<BOS>', '<EOS>', '<PAD>', '<UNK>']),
│   logger=LoggerConfig(
│   │   log_file='decoder.log',
│   │   module_name=None,
│   │   propagate=False,
│   │   log_root_dir='./data/adder/logs',
│   │   rich_handler_config={'level': 'INFO', 'show_level': True, 'show_path': True, 'show_time': True, 'rich_tracebacks': True, 'markup': True, 'log_time_format': '[%Y-%m-%d %H:%M:%S]'}
│   ),
│   global_=MaybeGlobal(seed=42, debug=False, debug_samples=None),
│   data=DataConfig(
│   │   context_length=11,
│   │   dataset_name='adder_dataset',
│   │   dataset_size=10000,
│   │   dataset_path='./data/adder/adder_dataset.txt',
│   │   dataset_dir='./data/adder',
│   │   dataset_url='https://raw.githubusercontent.com/gao-hongnan/omniverse/dev/omnivault/transformer/projects/adder/assets/adder_dataset.txt',
│   │   split=[0.7, 0.2, 0.1],
│   │   collate_fn={'batch_first': True, 'pad_token_id': 16},
│   │   train_loader={'batch_size': 32, 'shuffle': True, 'num_workers': 0, 'pin_memory': False, 'drop_last': False},
│   │   valid_loader={'batch_size': 32, 'shuffle': False, 'num_workers': 0, 'pin_memory': False, 'drop_last': False},
│   │   test_loader={'batch_size': 128, 'shuffle': False, 'num_workers': 0, 'pin_memory': False, 'drop_last': False}
│   ),
│   model=DecoderConfig(
│   │   d_model=128,
│   │   vocab_size=18,
│   │   context_length=11,
│   │   num_decoder_blocks=2,
│   │   dropout=0.1,
│   │   decoder_block=DecoderBlockConfig(
│   │   │   masked_self_attention_mha=MultiHeadedAttentionConfig(attention=ScaledDotProductAttention(
  (dropout): Dropout(p=0.0, inplace=False)
), d_model=128, H=4, dropout=0.1),
│   │   │   feed_forward=PositionwiseFeedForwardConfig(d_model=128, d_ff=256, activation=GELU(approximate='tanh'), dropout=0.1, bias=True),
│   │   │   add_norm_1=AddNormConfig(feature_dim=128, dropout=0.1),
│   │   │   add_norm_2=AddNormConfig(feature_dim=128, dropout=0.1)
│   │   )
│   ),
│   optimizer=AdamConfig(name='torch.optim.Adam', lr=0.2, betas=(0.9, 0.98), eps=1e-09, weight_decay=0.0),
│   criterion=CrossEntropyLossConfig(name='torch.nn.CrossEntropyLoss', weight=None, size_average=None, ignore_index=16, reduction='mean', label_smoothing=0.0),
│   scheduler=LambdaLRConfig(name='torch.optim.lr_scheduler.LambdaLR', lr_lambda=<function main.<locals>.<lambda> at 0x1650bb9d0>),
│   trainer=TrainerConfig(
│   │   device=device(type='mps'),
│   │   max_epochs=2,
│   │   log_every_n_steps=100,
│   │   eval_every_n_steps=4,
│   │   step_scheduler_on_batch_or_epoch='epoch',
│   │   use_amp=False,
│   │   autocast_config={'enabled': False},
│   │   scaler_config={'enabled': False, 'init_scale': 65536.0, 'growth_factor': 2.0, 'backoff_factor': 0.5, 'growth_interval': 2000},
│   │   gradient_accumulation_steps=1,
│   │   clip_grad_norm={'max_norm': 1.0, 'norm_type': 2.0, 'error_if_nonfinite': False, 'foreach': None},
│   │   apply_weight_decay_to_different_param_groups=False,
│   │   save_dir='./data/adder/checkpoints/2024-02-01_13-19-05',
│   │   save_every_epoch=False,
│   │   save_best_only=True,
│   │   monitor='valid_this_epoch_average_loss',
│   │   mode='min'
│   ),
│   generator=GeneratorConfig(max_tokens=4, temperature=1.0, greedy=True, top_k=None, top_p=None)
)
```

which is a `Composer` object where I **_compose_** various configurations into a
single object. This allow me to access any configuration from any place by
calling `Composer().<config_name>`.

-   The `Composer` object is **not** a **singleton**, although it is possible to
    make it so.
-   The `Composer` object is **not immutable**. This means that once it is
    created, it can be modified. This is by design, as it allows me to modify
    the configuration at runtime. This is useful for hyperparameter tuning
    techniques.

### State

The `State` object represents the current state of the training process. It
encapsulates key components such as the model, optimizer, criterion, scheduler,
and additional metadata. The inspiration for both `Composer` and `State` comes
from [MosaicML's Composer](https://github.com/mosaicml/composer), a library that
has been beneficial in the context of pretraining Language Models (LLMs) and is
also the library that me and my team adopted for our LLM pretraining project.

Here is a snippet of the `State` object:

```python
State(
│   model=GPTDecoder(
  (tok_embed): Embedding(18, 128)
  (decoder_blocks): ModuleList(
│   (0-1): 2 x GPTDecoderBlock(
│     (masked_self_attention_mha): MultiHeadedAttention(
│   │   (W_Q): Linear(in_features=128, out_features=128, bias=False)
│   │   (W_K): Linear(in_features=128, out_features=128, bias=False)
│   │   (W_V): Linear(in_features=128, out_features=128, bias=False)
│   │   (W_O): Linear(in_features=128, out_features=128, bias=False)
│   │   (attention): ScaledDotProductAttention(
│   │     (dropout): Dropout(p=0.0, inplace=False)
│   │   )
│   │   (dropout): Dropout(p=0.1, inplace=False)
│     )
│     (feed_forward): PositionwiseFeedForward(
│   │   (ffn): ModuleDict(
│   │     (context_fc): Linear(in_features=128, out_features=256, bias=True)
│   │     (activation): GELU(approximate='tanh')
│   │     (context_projection): Linear(in_features=256, out_features=128, bias=True)
│   │     (dropout): Dropout(p=0.1, inplace=False)
│   │   )
│     )
│     (add_norm_1): AddNorm(
│   │   (dropout): Dropout(p=0.1, inplace=False)
│   │   (layer_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
│     )
│     (add_norm_2): AddNorm(
│   │   (dropout): Dropout(p=0.1, inplace=False)
│   │   (layer_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
│     )
│   )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (layer_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  (head): Linear(in_features=128, out_features=18, bias=True)
),
│   criterion=CrossEntropyLoss(),
│   optimizer=Adam (
Parameter Group 0
│   amsgrad: False
│   betas: (0.9, 0.98)
│   capturable: False
│   differentiable: False
│   eps: 1e-09
│   foreach: None
│   fused: None
│   initial_lr: 0.2
│   lr: 1.0497284228847895e-06
│   maximize: False
│   weight_decay: 0.0
),
│   scheduler=<torch.optim.lr_scheduler.LambdaLR object at 0x1652552e0>,
│   epoch_index=0,
│   train_batch_index=0,
│   step_index=0,
│   history={},
│   vocabulary=<omnivault.transformer.core.vocabulary.AdderVocabulary object at 0x16522ea00>,
│   tokenizer=<omnivault.transformer.core.tokenizer.AdderTokenizer object at 0x16522ea60>
)
```

Why is `State` useful? It allows me to easily access the model, optimizer,
criterion, scheduler, and other metadata from any part of the codebase. I can
also serialize the `State` object and save it to disk, allowing me to resume
training from a checkpoint.

Perhaps not the best design pattern, but I added a `history` attribute to the
`State` object, which is a dictionary that stores the training history. This is
to mimic what could be a separate `History` object, which is a common component
in many deep learning libraries.

### Some Quirks of the Adder Project

-   Ensure the `dataset_path` in the `config.yaml` matches the location of the
    downloaded dataset file.
-   Ensure the `context_length` in the `config.yaml` matches the context length
    of the dataset. Here it is **hardcoded** and should not be changed for this
    particular dataset. Why? Because we encode each sample as "D1+D2=SUM" where
    D1 and D2 are the two digits and SUM is the sum of the two digits. Then we
    add `<BOS>` and `<EOS>` tokens to the beginning and end of the encoded
    sample respectively. This gives us a total of 11 tokens per sample:
    `<BOS>D1+D2=SUM<EOS>`. Hence, the `context_length` is 11.
-   If you want to try other digit lengths, you can change the `NUM_DIGITS` in
    the `constants` section of the `config.yaml` file. For example, if you want
    to try 3 digits, you can change the `NUM_DIGITS` to 3 and the
    `context_length` to say, 14. But you may need to change quite a fair bit of
    code yourself. Since this is an educational project, we leave this as an
    exercise to the reader.

    ```yaml
    constants:
        NUM_DIGITS: 2
        TOKENS:
            - "0"
            - "1"
            - "2"
            - "3"
            - "4"
            - "5"
            - "6"
            - "7"
            - "8"
            - "9"
            - "+"
            - "*"
            - "-"
            - "="
            - "<BOS>"
            - "<EOS>"
            - "<PAD>"
            - "<UNK>"
    ```

### Experiments

Note that for general CPU bound experiments, I expect my seeding mechanism to
work and is reproducible almost everywhere. But for GPU bound experiments, I
expect my seeding mechanism to work only on the same class of GPU (e.g. A100)
and not across different classes of GPUs (e.g. A100 vs V100) but the difference
is almost negligible.

**_Do not ask me to support Apple Silicon MPS, thank you._**

#### Run 1. CPU Bound 3 Epochs (Debug)

```bash
python omnivault/transformer/projects/adder/main.py \
    omnivault/transformer/projects/adder/config.yaml \
    data.train_loader.batch_size=256 \
    data.valid_loader.batch_size=256 \
    trainer.max_epochs=3 \
    trainer.use_amp=False \
    trainer.autocast_config.enabled=False \
    trainer.scaler_config.enabled=False \
    trainer.device='cpu'
```

![history-cpu-3-epochs](./projects/adder/assets/history_cpu_3_epochs.png)

| Epoch | Train Avg Loss | Train Avg Perplexity | Valid Avg Loss | Valid Avg Perplexity |
| ----- | -------------- | -------------------- | -------------- | -------------------- |
| 1     | 2.42116560     | 11.25897598          | 1.72266738     | 5.59944439           |
| 2     | 1.38095001     | 3.97867942           | 1.15814416     | 3.18401861           |
| 3     | 1.08565636     | 2.96138310           | 1.00055137     | 2.71978092           |

#### Run 2. CPU Bound 20 Epochs

```bash
python omnivault/transformer/projects/adder/main.py \
    omnivault/transformer/projects/adder/config.yaml \
    data.train_loader.batch_size=256 \
    data.valid_loader.batch_size=256 \
    trainer.max_epochs=20 \
    trainer.gradient_accumulation_steps=1 \
    trainer.use_amp=False \
    trainer.autocast_config.enabled=False \
    trainer.scaler_config.enabled=False \
    trainer.device='cpu'
```

![history-cpu-20-epochs](./projects/adder/assets/history_cpu_20_epochs.png)

| Epoch | Train Avg Loss | Train Avg Perplexity | Valid Avg Loss | Valid Avg Perplexity |
| ----- | -------------- | -------------------- | -------------- | -------------------- |
| 1     | 2.42116560     | 11.25897598          | 1.72266738     | 5.59944439           |
| 2     | 1.38095001     | 3.97867942           | 1.15814416     | 3.18401861           |
| 3     | 1.08565636     | 2.96138310           | 1.00055137     | 2.71978092           |
| 4     | 0.97149544     | 2.64189243           | 0.90245709     | 2.46565390           |
| 5     | 0.84379837     | 2.32518220           | 0.73687689     | 2.08939981           |
| 6     | 0.70981541     | 2.03361583           | 0.64679867     | 1.90941834           |
| 7     | 0.62240242     | 1.86339939           | 0.50294402     | 1.65358222           |
| 8     | 0.51285011     | 1.67004418           | 0.38060882     | 1.46317518           |
| 9     | 0.38310490     | 1.46683192           | 0.25848040     | 1.29496074           |
| 10    | 0.29655651     | 1.34521854           | 0.20162616     | 1.22339058           |
| 11    | 0.24845436     | 1.28204226           | 0.15636847     | 1.16925693           |
| 12    | 0.21495370     | 1.23980451           | 0.13172702     | 1.14079690           |
| 13    | 0.18957461     | 1.20873535           | 0.13184627     | 1.14093292           |
| 14    | 0.16487235     | 1.17924261           | 0.08768568     | 1.09164488           |
| 15    | 0.14672471     | 1.15803516           | 0.08021968     | 1.08352506           |
| 16    | 0.13451685     | 1.14398396           | 0.06706810     | 1.06936824           |
| 17    | 0.11999019     | 1.12748575           | 0.06959521     | 1.07207417           |
| 18    | 0.11789565     | 1.12512672           | 0.05264705     | 1.05405760           |
| 19    | 0.10456036     | 1.11022246           | 0.04009689     | 1.04091167           |
| 20    | 0.10515899     | 1.11088717           | 0.04866121     | 1.04986465           |

#### Run 3. CPU Bound 20 Epochs with Automatic Mixed Precision

Note from
[PyTorch's Autocasting documentation](https://pytorch.org/docs/stable/amp.html)
you can also use `autocast` with `cpu` devices.

```bash
python omnivault/transformer/projects/adder/main.py \
    omnivault/transformer/projects/adder/config.yaml \
    data.train_loader.batch_size=256 \
    data.valid_loader.batch_size=256 \
    trainer.max_epochs=20 \
    trainer.use_amp=True \
    trainer.autocast_config.enabled=True \
    trainer.autocast_config.dtype=bfloat16 \
    trainer.scaler_config.enabled=False \
    trainer.device='cpu'
```

| Epoch | Train Avg Loss | Train Avg Perplexity | Valid Avg Loss | Valid Avg Perplexity |
| ----- | -------------- | -------------------- | -------------- | -------------------- |
| 1     | 2.42121410     | 11.25952148          | 1.72274739     | 5.59989262           |
| 2     | 1.38103134     | 3.97900343           | 1.15812683     | 3.18396354           |
| 3     | 1.08579203     | 2.96178484           | 1.00137327     | 2.72201729           |
| 4     | 0.96206112     | 2.61708498           | 0.87629939     | 2.40199447           |
| 5     | 0.84047511     | 2.31746769           | 0.73473563     | 2.08493066           |
| 6     | 0.69479896     | 2.00330615           | 0.61569004     | 1.85093343           |
| 7     | 0.62559632     | 1.86936045           | 0.52938478     | 1.69788742           |
| 8     | 0.56361997     | 1.75702143           | 0.50377386     | 1.65495503           |
| 9     | 0.51176658     | 1.66823566           | 0.40421455     | 1.49812531           |
| 10    | 0.41958063     | 1.52132344           | 0.28967084     | 1.33598769           |
| 11    | 0.34217327     | 1.40800428           | 0.22319898     | 1.25006926           |
| 12    | 0.28290846     | 1.32698369           | 0.17602104     | 1.19246316           |
| 13    | 0.24093854     | 1.27244282           | 0.16882722     | 1.18391562           |
| 14    | 0.21423461     | 1.23891330           | 0.12139948     | 1.12907588           |
| 15    | 0.18088057     | 1.19827211           | 0.10155004     | 1.10688531           |
| 16    | 0.15750701     | 1.17058897           | 0.08484399     | 1.08854723           |
| 17    | 0.14483654     | 1.15585065           | 0.07683181     | 1.07986045           |
| 18    | 0.12972381     | 1.13851392           | 0.05752651     | 1.05921340           |
| 19    | 0.11486193     | 1.12171853           | 0.05551502     | 1.05708492           |
| 20    | 0.11150095     | 1.11795485           | 0.04883840     | 1.05005062           |

#### Run 4. CPU Bound 20 Epochs with Automatic Mixed Precision and Gradient Scaler

```bash
python omnivault/transformer/projects/adder/main.py \
    omnivault/transformer/projects/adder/config.yaml \
    data.train_loader.batch_size=256 \
    data.valid_loader.batch_size=256 \
    trainer.max_epochs=20 \
    trainer.use_amp=True \
    trainer.autocast_config.enabled=True \
    trainer.autocast_config.dtype=bfloat16 \
    trainer.scaler_config.enabled=True \
    trainer.device='cpu'
```

| Epoch | Train Avg Loss | Train Avg Perplexity | Valid Avg Loss | Valid Avg Perplexity |
| ----- | -------------- | -------------------- | -------------- | -------------------- |
| 1     | 2.42121410     | 11.25952148          | 1.72274739     | 5.59989262           |
| 2     | 1.38103134     | 3.97900343           | 1.15812683     | 3.18396354           |
| 3     | 1.08579203     | 2.96178484           | 1.00137327     | 2.72201729           |
| 4     | 0.96206112     | 2.61708498           | 0.87629939     | 2.40199447           |
| 5     | 0.84047511     | 2.31746769           | 0.73473563     | 2.08493066           |
| 6     | 0.69479896     | 2.00330615           | 0.61569004     | 1.85093343           |
| 7     | 0.62559632     | 1.86936045           | 0.52938478     | 1.69788742           |
| 8     | 0.56361997     | 1.75702143           | 0.50377386     | 1.65495503           |
| 9     | 0.51176658     | 1.66823566           | 0.40421455     | 1.49812531           |
| 10    | 0.41958063     | 1.52132344           | 0.28967084     | 1.33598769           |
| 11    | 0.34217327     | 1.40800428           | 0.22319898     | 1.25006926           |
| 12    | 0.28290846     | 1.32698369           | 0.17602104     | 1.19246316           |
| 13    | 0.24093854     | 1.27244282           | 0.16882722     | 1.18391562           |
| 14    | 0.21423461     | 1.23891330           | 0.12139948     | 1.12907588           |
| 15    | 0.18088057     | 1.19827211           | 0.10155004     | 1.10688531           |
| 16    | 0.15750701     | 1.17058897           | 0.08484399     | 1.08854723           |
| 17    | 0.14483654     | 1.15585065           | 0.07683181     | 1.07986045           |
| 18    | 0.12972381     | 1.13851392           | 0.05752651     | 1.05921340           |
| 19    | 0.11486193     | 1.12171853           | 0.05551502     | 1.05708492           |
| 20    | 0.11150095     | 1.11795485           | 0.04883840     | 1.05005062           |

#### Run 5. GPU Bound 30 Epochs with Automatic Mixed Precision and Gradient Scaler

```bash
python omnivault/transformer/projects/adder/main.py \
    omnivault/transformer/projects/adder/config.yaml \
    data.train_loader.batch_size=256 \
    data.valid_loader.batch_size=256 \
    optimizer.lr=0.2 \
    trainer.gradient_accumulation_steps=1 \
    trainer.max_epochs=30 \
    trainer.use_amp=True \
    trainer.autocast_config.enabled=True \
    trainer.autocast_config.dtype=float16 \
    trainer.scaler_config.enabled=True \
    trainer.device='cuda'
```

![history-cpu-30-epochs](./projects/adder/assets/history_gpu_amp_30_epochs.png)

| Epoch | Train Avg Loss      | Train Avg Perplexity | Valid Avg Loss       | Valid Avg Perplexity |
| ----- | ------------------- | -------------------- | -------------------- | -------------------- |
| 1     | 2.420887033053807   | 11.255838394165039   | 1.7224298896789552   | 5.598114490509033    |
| 2     | 1.382730863571167   | 3.9857711791992188   | 1.162042067527771    | 3.196453809738159    |
| 3     | 1.080898292677743   | 2.9473259449005127   | 1.0070842652320862   | 2.73760724067688     |
| 4     | 0.9785767320224217  | 2.6606667041778564   | 0.9240238981246949   | 2.5194079875946045   |
| 5     | 0.8295312314714705  | 2.2922439575195312   | 0.6926205568313598   | 1.998947024345398    |
| 6     | 0.6710659129960196  | 1.9563214778900146   | 0.5627854623794556   | 1.7555557489395142   |
| 7     | 0.5843813615526472  | 1.793880820274353    | 0.47745774388313295  | 1.611971139907837    |
| 8     | 0.502564266034535   | 1.6529544591903687   | 0.37658693075180055  | 1.457302212715149    |
| 9     | 0.3979068990434919  | 1.4887053966522217   | 0.2713440442085266   | 1.3117262125015259   |
| 10    | 0.32561164702687945 | 1.3848774433135986   | 0.2069695371389389   | 1.2299450635910034   |
| 11    | 0.28747650814056397 | 1.333059310913086    | 0.18322225725650787  | 1.201081395149231    |
| 12    | 0.25478792377880644 | 1.290187954902649    | 0.15108466696739198  | 1.1630951166152954   |
| 13    | 0.23126019149167198 | 1.2601871490478516   | 0.13933866548538207  | 1.1495133638381958   |
| 14    | 0.21447246055943625 | 1.2392079830169678   | 0.1252987619638443   | 1.1334871053695679   |
| 15    | 0.19230408913748606 | 1.2120389938354492   | 0.11941104763746262  | 1.1268329620361328   |
| 16    | 0.1761221342257091  | 1.192583680152893    | 0.08442495411634446  | 1.0880911350250244   |
| 17    | 0.14925001897130694 | 1.1609631776809692   | 0.0765499769449234   | 1.0795561075210571   |
| 18    | 0.13658928255523953 | 1.1463571786880493   | 0.07595605623722076  | 1.0789151191711426   |
| 19    | 0.13139716700145176 | 1.14042067527771     | 0.05362189581990242  | 1.0550856590270996   |
| 20    | 0.12509043884277343 | 1.1332509517669678   | 0.058306344807147976 | 1.0600396394729614   |
| 21    | 0.11421211144753865 | 1.1209899187088013   | 0.051042843461036685 | 1.052367925643921    |
| 22    | 0.11002175408601761 | 1.1163023710250854   | 0.05413959649205208  | 1.0556319952011108   |
| 23    | 0.10133451878173011 | 1.1066467761993408   | 0.0411556151509285   | 1.042014241218567    |
| 24    | 0.10018374175684792 | 1.1053739786148071   | 0.052105383694171906 | 1.053486704826355    |
| 25    | 0.10159334559100015 | 1.1069332361221313   | 0.03665199390053749  | 1.0373319387435913   |
| 26    | 0.0877149731857436  | 1.091676950454712    | 0.03356390315294266  | 1.0341335535049438   |
| 27    | 0.08452736897127969 | 1.0882025957107544   | 0.03531740158796311  | 1.0359485149383545   |
| 28    | 0.08164276150294712 | 1.0850681066513062   | 0.04029492244124413  | 1.041117787361145    |
| 29    | 0.0783526828118733  | 1.081503987312317    | 0.028568922132253646 | 1.0289809703826904   |
| 30    | 0.07325152178747313 | 1.0760011672973633   | 0.024026787236332895 | 1.024317741394043    |

### Run 6: GPU Bound 30 Epochs with Automatic Mixed Precision, Gradient Scaler and Gradient Accumulation

```bash
python omnivault/transformer/projects/adder/main.py \
    omnivault/transformer/projects/adder/config.yaml \
    data.train_loader.batch_size=256 \
    data.valid_loader.batch_size=256 \
    optimizer.lr=0.8 \
    trainer.gradient_accumulation_steps=4 \
    trainer.max_epochs=30 \
    trainer.use_amp=True \
    trainer.autocast_config.enabled=True \
    trainer.autocast_config.dtype=float16 \
    trainer.scaler_config.enabled=True \
    trainer.device='cuda'
```

```bash
# if weight decay is 0, then it is as good as not applying custom weight decay to diff param groups:
python omnivault/transformer/projects/adder/main.py omnivault/transformer/projects/adder/config.yaml data.train_loader.batch_size=256 data.valid_loader.batch_size=256 trainer.apply_weight_decay_to_different_param_groups=True optimizer.weight_decay=1e-2
```

#### Generalization

To test the "generalization", we can ask some questions that are not in the
training set:

```bash
97+98=195
96+96=192
95+95=190
```

but we do not really need to do this since we split into `train-valid-test`
already, and in a sense, the `valid` and `test` sets are "unseen" by the model,
acting as a _rough_ holdout. Note this is very rough because there are leakage,
the split does not guarantee that equations in the `train` set do not appear in
the `valid` or `test` sets.

> Important, we must use greedy generation and not top-k or top-p (nuclues)
> sampling here because we really just want the model to output the exact
> answer, and not some other answer that is close to the correct answer in the
> distribution of the model's vocabulary.

This also yields an validation accuracy of about 97.4% over 1000 samples
(974/1000).
