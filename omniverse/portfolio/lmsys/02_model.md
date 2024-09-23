# Modelling

```{contents}
:local:
```

Given a model $\mathcal{M}$ and a dataset $\mathcal{S}$ sampled i.i.d. from a
distribution $\mathcal{D}$, we aim to classify text into $K$ different
categories $\mathcal{C} = \{c_1, c_2, \ldots, c_K\}$.

## Text Classification

### Dataset Representation

The dataset $\mathcal{S}$ consists of $N$ sequences, each sampled independently
from the distribution $\mathcal{D}$:

$$
\mathcal{S} = \left \{ \left(\mathrm{X}^{(n)}, Y^{(n)} \right) \right \}_{n=1}^N \overset{\mathrm{iid}}{\sim} \mathcal{D} \in \left(\mathcal{X} \times \mathcal{Y} \right)^N
$$

Here:

-   $\mathcal{X}$ represents the input text space.
-   $\mathcal{Y} = \mathcal{C}$ represents the label space with $K$ categories.
-   $\mathrm{X}^{(n)} \in \mathcal{X}$ is the $n$-th input sequence.
-   $Y^{(n)} \in \mathcal{Y}$ is the corresponding label.

### Model Representation

Using a decoder-only causal LLM $\mathcal{M}$, the classification task can be
framed as predicting the probability distribution over the categories
$\mathcal{C}$ given an input sequence $\mathrm{X}$:

$$
P(Y = c_k \mid \mathrm{X}; \mathcal{M}) \quad \text{for} \quad k = 1, 2, \ldots, K
$$

The model $\mathcal{M}$ generates the probability for each class $c_k$ by
processing the input sequence $\mathrm{X}$ and leveraging its autoregressive
capabilities to output logits, which are then transformed into probabilities via
a softmax function. Of course we will see in the next section how we can do it
via a pooling operation because our LLM is no longer predicting the next token
$x_t$ given $x_{<t}$. So we need to aggregate the logits of the last token to
get the final result.

## Text Generation

Given a model $\mathcal{M}$ and a dataset $\mathcal{S}$ sampled i.i.d. from a
distribution $\mathcal{D}$, the objective of text generation is to produce
coherent and contextually relevant sequences of text. Unlike classification,
where the goal is to assign a label to an input, text generation aims to
generate a sequence of tokens
$\mathrm{X} = (\mathrm{x}_1, \mathrm{x}_2, \ldots, \mathrm{x}_T)$ that is
plausible under the distribution $\mathcal{D}$.

### Dataset Representation

The dataset $\mathcal{S}$ consists of $N$ sequences, each sampled independently
from the distribution $\mathcal{D}$:

$$
\mathcal{S} = \left \{ \mathrm{X}^{(n)} \right \}_{n=1}^N \overset{\mathrm{iid}}{\sim} \mathcal{D} \in \mathcal{X}^N
$$

Here:

-   $\mathcal{X}$ represents the space of all possible token sequences.
-   $\mathrm{X}^{(n)} = (\mathrm{x}_1^{(n)}, \mathrm{x}_2^{(n)}, \ldots, \mathrm{x}_{T_n}^{(n)})$
    is the $n$-th input sequence, where $T_n$ is the length of the sequence.

### Autoregressive Modeling for Text Generation

A decoder-only causal LLM $\mathcal{M}$ generates text by modeling the
conditional probability of each token given the preceding tokens. This approach
leverages the autoregressive property, where the generation of each token
depends only on the tokens that come before it.

Formally, the probability of generating a sequence
$\mathrm{X} = (\mathrm{x}_1, \mathrm{x}_2, \ldots, \mathrm{x}_T)$ is decomposed
as:

$$
P(\mathrm{X}; \mathcal{M}) = \prod_{t=1}^T P(\mathrm{x}_t \mid \mathrm{x}_{<t}; \mathcal{M})
$$

where:

-   $\mathrm{x}_{<t} = (\mathrm{x}_1, \mathrm{x}_2, \ldots, \mathrm{x}_{t-1})$
    denotes the sequence of tokens preceding $\mathrm{x}_t$.
-   $P(\mathrm{x}_t \mid \mathrm{x}_{<t}; \mathcal{M})$ is the probability of
    token $\mathrm{x}_t$ given the previous tokens, as predicted by the model
    $\mathcal{M}$.

### Model Architecture and Token Prediction

The decoder-only causal LLM $\mathcal{M}$ processes the input sequence token by
token to generate hidden states, which are then used to predict the next token
in the sequence. For each position $t$, the model computes a hidden state $h_t$:

$$
h_t = \mathcal{M}(\mathrm{X}_{\leq t})_t
$$

where
$\mathrm{X}_{\leq t} = (\mathrm{x}_1, \mathrm{x}_2, \ldots, \mathrm{x}_t)$.

The probability distribution over the vocabulary $\mathcal{V}$ for the next
token $\mathrm{x}_{t+1}$ is obtained by applying a linear transformation
followed by a softmax function to the hidden state $h_t$:

$$
P(\mathrm{x}_{t+1} \mid \mathrm{x}_{\leq t}; \mathcal{M}) = \text{softmax}(W h_t + b)
$$

where:

-   $W \in \mathbb{R}^{|\mathcal{V}| \times d}$ is the weight matrix.
-   $b \in \mathbb{R}^{|\mathcal{V}|}$ is the bias vector.
-   $d$ is the dimensionality of the hidden state $h_t$.

### Training Objective

The model $\mathcal{M}$ is trained to maximize the likelihood of the training
data under the model's probability distribution. The objective is to maximize
the log-likelihood of the observed sequences:

$$
\mathcal{L}(\mathcal{M}) = \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} \log P\left(\mathrm{x}_t^{(n)} \mid \mathrm{x}_{<t}^{(n)}; \mathcal{M}\right)
$$

This objective encourages the model to assign high probabilities to the actual
next tokens in the training sequences.

### Generation Process

Once trained, text generation involves sampling or selecting tokens sequentially
based on the conditional probabilities predicted by the model. The generation
process can be formalized as follows:

1. **Initialization:** Start with an initial prompt or context
   $\mathrm{X}_{\leq 0} = \emptyset$ or a given sequence of tokens.
2. **Token Generation:** For each timestep $t = 1, 2, \ldots, T$:
    - Compute the hidden state $h_t = \mathcal{M}(\mathrm{X}_{\leq t})_t$.
    - Predict the probability distribution
      $P(\mathrm{x}_t \mid \mathrm{x}_{<t}; \mathcal{M})$.
    - Sample or select the next token $\mathrm{x}_t$ from the distribution.
3. **Termination:** Continue the process until a predefined stopping condition
   is met (e.g., generating a special end-of-sequence token or reaching a
   maximum length).

In our case, it is simple, we just need to generate the last token and use it as
the result because our prompt is designed to be something like a classification
where the last token in the prompt is the "category" of the input.

## Candidates

We are under constraints so we mostly experiment with the following candidates:

-   Gemma2-9B
-   Llama3-8B
-   Mistral-7B

And some models ranging from 20-40B for potential distillation. All of them
share a common config:

```python
class BaseModelConfig(BaseModel):
    """Base class for model configuration."""

    pretrained_model_name_or_path: str = Field(default=None)

    # from PretrainedConfig
    cache_dir: str | os.PathLike[str] | None = Field(default=None, examples=["./.cache/huggingface"])
    # device_map: str | Dict[str, int] | None = "auto"
    torch_dtype: (
        Literal["float16", "float32", "bfloat16", "float64", "int32", "int64", "int16", "int8", "uint8", "bool"] | None
    ) = None
    output_hidden_states: bool = Field(
        default=False,
        description="Read https://huggingface.co/docs/transformers/en/main_classes/output",
    )
    output_attentions: bool = Field(
        default=False,
        description="Read https://huggingface.co/docs/transformers/en/main_classes/output",
    )
    num_labels: int = Field(default=None, description="Number of labels.")
    problem_type: Literal["single_label_classification", "multi_label_classification", "regression"] | None = None

    # dropout shenanigans
    attention_dropout: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    hidden_dropout_prob: float = 0.0
    pooler_dropout: float = 0.0

    # custom
    model_type: Literal[
        "vanilla",
        "gemma2-9b-it-bidirectional",
        "llama3-8b-it-bidirectional",
        "llama3-8b-it",
        "llama3.1-8b-it",
        "vicuna-7b-v1.5",
        "gemma2-9b-it",
        "mistral-nemo-instruct-2407",
    ] = "vanilla"

    ## freeze and init shenanigans
    init_config: Dict[str, Any] = {}
    num_layers_to_remove: int | None = None
    reinitialize_n_layers_of_backbone: int = 0
    freeze_these_layers_indices: List[int] | None = None
    freeze_embeddings: bool = False

    # criterion
    criterion: Literal[
        "mse", "cross_entropy", "bce", "bitempered"
    ] | None = None  # ordinal-log-loss, cross-entropy, mse
    criterion_config: Dict[str, Any] = {}

    ## pooler
    pooler_type: Literal["context", "mean", "attention", "gem"] | None = None
    pooler_config: Dict[str, Any] | None = None

    ## head
    head_type: Literal["vanilla", "sequential"] | None = Field(
        default=None,
        description="""If `vanilla`, we use the default
                        head with single `linear` layer.
                        Else we use a sequential head.""",
    )

    ## pretrained LoRA adapter config
    pretrained_adapter_path: str | None = Field(default=None)
    is_trainable: bool = Field(default=True, description="Whether to continue train the adapter or not.")

    @field_validator("torch_dtype")
    @classmethod
    def convert_string_to_dtype(cls: Type[BaseModelConfig], v: str | None) -> torch.dtype:
        if v is None:
            return torch.float32

        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
            "int64": torch.int64,
            "int16": torch.int16,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }
        if v not in dtype_map:
            raise ValueError(f"Unsupported dtype: {v}")
        return dtype_map[v]

    class Config:
        """Pydantic configuration."""

        protected_namespaces = ()
        arbitrary_types_allowed = True
```

## Pooling

### [CLS-Decoder] Last Token Pooling

We mostly use the LLMs as a sequence classification model, this is easy to
implement or you can just use the logits of the last token as the classification
result - which is called Last Token Pooling or Context Pooling. Just remember,
in a causal uni-direction self attention model, the last token is the most
relevant one as it holds all information of the input sequence - since that's
the only token that can look at all other tokens.

```python
from __future__ import annotations

from typing import overload

import torch
from torch import nn

__all__ = ["LastTokenPooling"]


class LastTokenPooling(nn.Module):
    """Last token pooling layer - specifically for decoder only models to do
    fine-tuning on sequence classification tasks."""

    def __init__(self, pre_head_pooling: bool = True) -> None:
        super().__init__()
        self.pre_head_pooling = pre_head_pooling

    @overload
    def forward(self, last_hidden_state: torch.Tensor, logits: None = None) -> torch.Tensor:
        ...

    @overload
    def forward(self, last_hidden_state: None, logits: torch.Tensor) -> torch.Tensor:
        ...

    def forward(
        self, last_hidden_state: torch.Tensor | None = None, logits: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for the pooling layer.

        Parameters
        ----------
        last_hidden_state:  Hidden state of the last layer.
                            type:  torch.Tensor
                            shape: (B, T, D)
        logits:             Logits from the last layer.
                            type:  torch.Tensor
                            shape: (B, T, C)

        Notes
        -----
        In both cases, we will slice the `T` dimension to get the last token's
        hidden state or logits. For example, if `last_hidden_state` is provided,
        then we have `[B, T, D] -> [B, D]` and if `logits` is provided, then we
        have `[B, T, C] -> [B, C]`.
        """
        if self.pre_head_pooling:
            assert last_hidden_state is not None, "last_hidden_state must be provided when pre_head is True"
            pooled_hidden_state = last_hidden_state[:, -1, :]
            return pooled_hidden_state
        else:
            assert logits is not None, "logits must be provided when pre_head is False"
            pooled_logits = logits[:, -1, :]
            return pooled_logits
```

### [CLS-Encoder] Mean Pooling

This is mostly done in encoder models, we simply average the logits of all
tokens to get a single vector representation. This is easy to implement and can
capture some global information of the input sequence.

### Attention Pooling

This is also mostly done in encoder models. We did experiment briefly with
DebertaV3 where we pooled the logits via attention scores.

## Head

We used a custom head.

### [CLS] Normal Sequential Head

```python
if config.head_type is None or config.head_type == "vanilla":
    logger.info("Using vanilla head!")
    self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
elif config.head_type == "sequential":
    logger.info("Using sequential head!")
    self.score = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(config.hidden_size, config.hidden_size),
        nn.Dropout(0.1),
        nn.GELU(),
        nn.Linear(config.hidden_size, self.num_labels, bias=False),
    )
```

### [CLS] Attention Pooling Head

```python
class AttentionHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights_mask = attention_mask.unsqueeze(-1)
        att = torch.tanh(self.W(features))
        score = self.V(att)
        score[attention_mask == 0] = -1e4
        attention_weights = torch.softmax(score, dim=1)
        context_vector = torch.sum(attention_weights * weights_mask * features, dim=1)
        return context_vector
```

## Model Summary

```python
if composer.shared.show_model_summary:
    logger.info("Showing model summary.")
    logger.warning(
        "Be careful as `torchinfo` might MUTATE model init weights, so if you run without `torchinfo` your results from model may differ!"
    )
    try:
        torchinfo.summary(
            base_model,
            verbose=1,
            input_data={
                "input_ids": sample_batch_and_collated_sample_batch["collated_sample_batch"]["input_ids"],
                "attention_mask": sample_batch_and_collated_sample_batch["collated_sample_batch"]["attention_mask"],
            },
            dtypes=list[torch.LongTensor],  # type: ignore[arg-type]
            device=base_model.device,
        )
    except RuntimeError as exc:
        logger.exception(msg="Error in torchinfo.summary", exc_info=exc)
```

## Model Statistics

Yes for weight distribution and other statistics.

```python
def analyze_model(model: torch.nn.Module, verbose: bool = False) -> Dict[str, Any]:
    """Analyze and log details about the model."""
    named_modules = get_named_modules(model)
    last_module_name = next(reversed(named_modules))

    weight_stats = gather_weight_stats(model)
    total_params = total_parameters(model)
    total_trainable_params = total_trainable_parameters(model)

    if verbose:
        logger.info("Base model named modules:\n%s", jsonify(named_modules))
        logger.info("Base model weight stats:\n%s", jsonify(weight_stats))

    logger.info("Last module name: %s", last_module_name)
    logger.info(
        "Base model %s\ntotal trainable parameters: %.2fM\ntotal parameters: %.2fM",
        model.__class__.__name__,
        total_trainable_params / 1_000_000,
        total_params / 1_000_000,
    )

    return {
        "named_modules": named_modules,
        "last_module_name": last_module_name,
        "weight_stats": weight_stats,
        "total_params": total_params,
        "total_trainable_params": total_trainable_params,
    }
```

## Model Sanity Check

```python
def dry_run(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> ModelOutput:
    """Dry run the model to check if the model is correctly set up."""
```

## Low Rank Adaptation

We use LoRA. We froze $N$ layers in `layers_to_transform` and only adapted the
last $M$ layers to stabilize the training process.

```python
class AdapterConfig(BaseModel):
    """Base class for LoRA configuration."""

    peft_type: PeftType = PeftType.LORA

    task_type: TaskType = TaskType.CAUSAL_LM
    inference_mode: bool = False
    r: int = 32  # rank
    lora_alpha: int = 16  # regularization
    lora_dropout: float = 0.05  # dropout
    target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
    ]

    layers_to_transform: List[int] | None = None
    bias: Literal["none", "all", "lora_only"] = "none"
    modules_to_save: List[str] | None = None
    init_lora_weights: bool | Literal["gaussian", "pissa", "pissa_niter_[number of iters]", "loftq"] = True
    fan_in_fan_out: bool = False
    use_rslora: bool = False
    layers_pattern: List[str] | str | None = None
    rank_pattern: Dict[str, Any] | None = None
    alpha_pattern: Dict[str, Any] | None = None
    use_dora: bool = False
```
