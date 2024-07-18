import torch
from torch import nn
from transformers import GPT2ForSequenceClassification

from omnivault.transformer.config.decoder import DecoderConfig
from omnivault.transformer.decoder.core import GPTDecoder

"""
        logits:                 Output logits.
                                type:  torch.FloatTensor
                                shape: (B, S or T, V)

So if logits is of shape `[B, T, V]` where `B` is the batch size, `T` is the
sequence length, and `V` is the vocabulary size, then the output of
the model is a sequence of logits for each token in the input sequence.
Each entry in this tensor represents the logits (unnormalized log probabilities)
for each vocabulary token at each position in each sequence for each batch entry.
In other words, if `B=2`, `T=3`, and `V=4`, then the output of the model would be a
tensor of shape `[2, 3, 4]`. Let's take the first batch for example, if we slice it,
it would have a shape of `[3, 4]` where each
row (token) is a sequence of logits for each token in the input sequence. Consider
the a sequence _cats eat the mouse_, and since our sequence length is `3`, we would
then have the input sequence in question to be _cats eat the_ and the target
sequence to be _eat the mouse_. Now each word is a token here
for simplicity, then the output logits for this input sequence might look like
`[logits_for_cats, logits_for_eat, logits_for_the]` where each `logits_for_wordX` is a vector
of size `V` representing the logits for each token in the vocabulary (after processing
the word `X`). The task of our model is then to predict the target sequence
`_eat the mouse_` given the input sequence `_cats eat the_` - so ideally we would
want the model to have a high logit for the token `eat` corresponding to the
word `cats` in the input sequence, and so on. The model might predict that after
"the", the word "mouse" is highly probable, so the corresponding logit value for
"mouse" in logits_the would be relatively higher compared to other words in the vocabulary.

Now if we want to use this model for sequence classification, we would want to
extract the logits corresponding to the last token in the sequence, which is the
token that represents the entire sequence. As the mantra goes, in causal attention,
the attention is uni-directional from left to right, so the last token in the
sequence would have seen all the tokens before it, and it would behave exactly
like a token in cross-attention (it knows itself and have information of every other
token in the sequence). The last token is condensed with all the information
from the entire sequence, and it is the token that represents the entire sequence.
This form of aggregation is often termed as _pooling_ in the context of sequence
classification. Think of it like this, we have token level information, but it is
insufficient here, we would like a sequence/sentence level representation, so the
easiest way here is to take the last token in the sequence in a decoder only model.
To be more pedantic, most pooling (mean, max, gem pooling) occurs after the
backbone, before the head, in which case the embeddings output from the backbone
are further aggregated before being passed to the head. In our case, we are
pooling the logits after the head.
Therefore, our logits can assume the shape `[B, 1, V]` where `B` is the batch size
`1` is the sequence length (since we are taking the last token), and `V` is the
vocabulary size. Before we move on, I want to change the notation `V` to `C` to
represent the number of classes in the classification task. In our implementation,
it is more logical to slice the logits at the loss calculation level in trainer,
in HuggingFace, the ecosystem is designed to slice it in model level and both
ways are valid - we would use the HuggingFace way here just to be cool.

```
self.pooler =
"""

import torch
import torch.nn as nn


class ContextPooling(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # pooling to extract the last token's logits
        pooled_logits = logits[:, -1, :]
        return pooled_logits


class DecoderForSequenceClassificationConfig(DecoderConfig):
    num_labels: int


class GPTForSequenceClassification(GPTDecoder):
    def init(self, config: DecoderForSequenceClassificationConfig) -> None:
        super().__init__(config)

        self.pooler = ContextPooling()
        self.head = nn.Linear(config.d_model, config.num_labels)

        self.apply(self._init_weights)

        context_projections = ("context_projection.weight", "W_O.weight")
        # apply special scaled init to the residual projections, per GPT-2 paper
        for parameter_name, parameter in self.named_parameters():
            # NOTE: W_O is also projection but I did not have foresight to name it as such.
            if parameter_name.endswith(context_projections):
                mean = 0.0
                std_dev = 0.02 / torch.sqrt(torch.tensor(2 * config.num_decoder_blocks, dtype=torch.float))
                torch.nn.init.normal_(parameter, mean=mean, std=std_dev)

    def forward(  # type: ignore[override]
        self,
        input_tokens: torch.LongTensor,
        *,  # force keyword only arguments to prevent errors
        target_padding_masks: torch.BoolTensor | None = None,
        future_masks: torch.BoolTensor | None = None,
    ) -> torch.FloatTensor:
        batch_size: int = input_tokens.size(0)
        seq_len: int = input_tokens.size(1)  # note seq_len <= context_length in decoder
        target_masks: torch.BoolTensor = self.create_target_masks(
            batch_size=batch_size, seq_len=seq_len, target_padding_masks=target_padding_masks, future_masks=future_masks
        )

        target_masks = target_masks.to(input_tokens.device)  # type: ignore[assignment]

        z = self.tok_embed(input_tokens)  # TODO: * math.sqrt(self.d_model) for better optimization landscape
        z = z + self.pos_embed[:, :seq_len, :]
        z = self.dropout(z)

        for decoder_block in self.decoder_blocks:
            z = decoder_block(z, target_masks=target_masks)

        z = self.layer_norm(z)
        logits: torch.FloatTensor = self.head(z)
        return logits
