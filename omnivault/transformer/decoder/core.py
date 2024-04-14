from __future__ import annotations

from typing import List, cast, overload

import torch
from torch import nn
from typing_extensions import override

from omnivault._types._alias import NotGiven
from omnivault._types._sentinel import NOT_GIVEN
from omnivault.transformer.config.decoder import DecoderConfig
from omnivault.transformer.core.dataset import (
    construct_dummy_batch_future_masks,
    construct_dummy_batch_target_padding_masks,
)
from omnivault.transformer.decoder.base import BaseDecoder, BaseDecoderBlock
from omnivault.transformer.modules.attention.core import MultiHeadedAttention
from omnivault.transformer.modules.layers.addnorm import AddNorm
from omnivault.transformer.modules.layers.mlp import PositionwiseFeedForward

__all__ = ["GPTDecoderBlock", "GPTDecoder"]


class GPTDecoderBlock(BaseDecoderBlock):
    """GPTDecoderBlock focuses on masked self-attention and feed-forward layers.

    The architecture follows the GPT-style decoder, which only has masked
    self-attention and position-wise feed-forward layers, omitting the
    encoder-decoder cross-attention.
    """

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__(config)
        # fmt: off
        self.masked_self_attention_mha = MultiHeadedAttention(**config.decoder_block.masked_self_attention_mha.model_dump(mode="python"))
        # self.encoder_decoder_cross_attention_mha = MultiHeadedAttention(**config.decoder.encoder_decoder_cross_attention_mha)

        self.feed_forward              = PositionwiseFeedForward(**config.decoder_block.feed_forward.model_dump(mode="python"))

        self.add_norm_1                = AddNorm(**config.decoder_block.add_norm_1.model_dump(mode="python"))
        self.add_norm_2                = AddNorm(**config.decoder_block.add_norm_2.model_dump(mode="python"))

        # self.feed_forward.register_forward_hook(forward_hook)
        # fmt: on

    def forward(
        self,
        z: torch.Tensor,  # that's tgt in torch code base
        *,
        encoder_hidden_states: torch.Tensor | NotGiven = NOT_GIVEN,  # that's memory in torch code base
        encoder_hidden_states_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,  # that's memory_mask in torch code base
        target_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,  # that's tgt_mask in torch code base
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z:              Input sequence.
                        type:  torch.Tensor
                        shape: (B, S or T, D)
        target_masks:   Target mask.
                        type:  torch.BoolTensor
                        shape: (B, 1, S or T, S or T)

        Returns
        -------
        z:              Output tensor after masked self-attention and feed-forward layers.
                        type:  torch.Tensor
                        shape: (B, S or T, D)
        """
        assert encoder_hidden_states is NOT_GIVEN, "GPTDecoderBlock does not have encoder-decoder cross-attention"
        assert encoder_hidden_states_masks is NOT_GIVEN, "GPTDecoderBlock does not have encoder-decoder cross-attention"
        assert target_masks is not NOT_GIVEN

        z = self.add_norm_1(
            z,
            lambda z: self.masked_self_attention_mha(query=z, key=z, value=z, mask=target_masks),
        )
        z = self.add_norm_2(z, self.feed_forward)
        return z


# NOTE: seq_len <= context_length == max_seq_len
class GPTDecoder(BaseDecoder):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__(config)
        # fmt: off
        self.d_model       : int           = config.d_model
        self.tok_embed     : nn.Embedding  = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed     : nn.Parameter  = nn.Parameter(torch.zeros(1, config.context_length, config.d_model))
        self.decoder_blocks: nn.ModuleList = nn.ModuleList([GPTDecoderBlock(config) for _ in range(config.num_decoder_blocks)]) # PyTorch did not make ModuleList a proper container, maybe open a PR to make it inherit Generic[T]???

        self.dropout       : nn.Dropout    = nn.Dropout(config.dropout)
        self.layer_norm    : nn.LayerNorm  = nn.LayerNorm(config.d_model)
        self.head          : nn.Linear     = nn.Linear(config.d_model, config.vocab_size)  # last layer
        # fmt: on

        self.apply(self._init_weights)

        context_projections = ("context_projection.weight", "W_O.weight")
        # apply special scaled init to the residual projections, per GPT-2 paper
        for parameter_name, parameter in self.named_parameters():
            # NOTE: W_O is also projection but I did not have foresight to name it as such.
            if parameter_name.endswith(context_projections):
                mean = 0.0
                std_dev = 0.02 / torch.sqrt(torch.tensor(2 * config.num_decoder_blocks, dtype=torch.float))
                torch.nn.init.normal_(parameter, mean=mean, std=std_dev)

    @property
    def total_trainable_parameters(self) -> int:
        """Returns the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_parameters(self) -> int:
        """Returns the total number of parameters in the model, including non-trainable."""
        return sum(p.numel() for p in self.parameters())

    @override
    def _init_weights(self, module: nn.Module) -> None:
        normal_init_modules = (nn.Linear, nn.Embedding)
        if isinstance(module, normal_init_modules):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    @overload
    def create_target_masks(
        self, batch_size: int, seq_len: int, target_padding_masks: torch.BoolTensor, future_masks: torch.BoolTensor
    ) -> torch.BoolTensor:
        ...

    @overload
    def create_target_masks(
        self, batch_size: int, seq_len: int, target_padding_masks: torch.BoolTensor, future_masks: NotGiven
    ) -> torch.BoolTensor:
        ...

    @overload
    def create_target_masks(
        self, batch_size: int, seq_len: int, target_padding_masks: NotGiven, future_masks: torch.BoolTensor
    ) -> torch.BoolTensor:
        ...

    @overload
    def create_target_masks(
        self, batch_size: int, seq_len: int, target_padding_masks: NotGiven, future_masks: NotGiven
    ) -> torch.BoolTensor:
        ...

    def create_target_masks(
        self,
        batch_size: int,
        seq_len: int,
        target_padding_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,
        future_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,
    ) -> torch.BoolTensor:
        """
        Creates a combined target mask for use in decoder layers. If
        `target_padding_masks` is not provided, a default mask of ones is
        created. If `future_masks` is not provided, a default lower triangular
        mask is created to mask future tokens.

        The default mask created by `torch.ones_like` or `torch.tril` acts as
        a placeholder that allows operations to proceed without altering the
        behavior that the mask would impose. This is particularly useful when
        the absence of a mask should not lead to masking out or altering any
        data, but rather to a 'pass-through' behavior. For example, in the
        case of target padding masks, the default mask allows the model to
        attend to all tokens in the target sequence, since the default mask is
        all ones and thus does not mask out any tokens. What this means is if
        the user does not provide a target padding mask, the model will not
        mask out any tokens in the target sequence, which is the same behavior
        as if the user had provided a target padding mask of all ones.

        Parameters
        ----------
        batch_size : int
            The batch size of the input sequences.

        seq_len : int
            The sequence length of the input sequences.

        target_padding_masks : torch.BoolTensor | NotGiven
            The mask for the target padding, indicating which elements of the target
            should be masked out. If not provided, a default mask of ones is created
            that does not mask out anything.

        future_masks : torch.BoolTensor | NotGiven
            The mask for future tokens, typically used in self-attention mechanisms
            to prevent the model from 'seeing' future tokens. If not provided, a
            default mask of ones is created that does not prevent attending to future
            tokens.

        Returns
        -------
        torch.BoolTensor
            A combined boolean mask that is the logical AND of the target padding mask
            and future mask. The shape of the returned mask matches the input masks.
        """
        target_masks_shape = (batch_size, 1, seq_len, seq_len)
        if target_padding_masks is NOT_GIVEN and future_masks is NOT_GIVEN:
            target_padding_masks = cast(
                torch.BoolTensor, construct_dummy_batch_target_padding_masks(batch_size, seq_len)
            )
            future_masks = cast(torch.BoolTensor, construct_dummy_batch_future_masks(batch_size, seq_len))

        # FIXME: CAN SOMEONE PLEASE HELP ME WITH TYPING HERE?? I AM SO STUCK IN CASTING HELL.
        if target_padding_masks is NOT_GIVEN:
            target_padding_masks = cast(
                torch.BoolTensor, construct_dummy_batch_target_padding_masks(batch_size, seq_len)
            )

        if future_masks is NOT_GIVEN:
            future_masks = cast(torch.BoolTensor, construct_dummy_batch_future_masks(batch_size, seq_len))

        assert target_padding_masks.shape == future_masks.shape == target_masks_shape  # type: ignore[union-attr]

        return cast(
            torch.BoolTensor,
            torch.logical_and(cast(torch.Tensor, target_padding_masks), cast(torch.Tensor, future_masks)).bool(),
        )

    def forward(
        self,
        input_tokens: torch.LongTensor,
        *,  # force keyword only arguments to prevent errors
        target_padding_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,
        future_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,
        # fmt: off
        encoder_hidden_states: torch.Tensor | NotGiven = NOT_GIVEN,  # that's memory in torch code base and is ensured not used here
        encoder_hidden_states_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,
        # that's memory_mask in torch code base and is ensured not used here
        # fmt: on
    ) -> torch.FloatTensor:
        """
        Notations
        ---------
        B:      Batch size
        S or L: Source sequence length
        T or L: Target sequence length
        D:      Embedding dimension
        V:      Vocabulary size (Class size)

        Parameters
        ----------
        input_tokens:           Input sequence.
                                type:  torch.Tensor
                                shape: (B, S or T)
        target_padding_masks:   Target padding mask.
                                type:  torch.BoolTensor
                                shape: (B, 1, S or T, S or T)
        future_masks:           Future mask.
                                type:  torch.BoolTensor
                                shape: (B, 1, S or T, S or T)

        Variables
        ---------
        z:                      Input sequence after token and position embedding.
                                type:  torch.Tensor
                                shape: (B, S or T, D)
        target_masks:           Target mask.
                                type:  torch.BoolTensor
                                shape: (B, 1, S or T, S or T)
        logits:                 Output logits.
                                type:  torch.FloatTensor
                                shape: (B, S or T, V)
        """
        assert encoder_hidden_states is NOT_GIVEN, "GPTDecoderBlock does not have encoder-decoder cross-attention"
        assert encoder_hidden_states_masks is NOT_GIVEN, "GPTDecoderBlock does not have encoder-decoder cross-attention"

        # fmt: off
        batch_size  : int              = input_tokens.size(0)
        seq_len     : int              = input_tokens.size(1) # note seq_len <= context_length in decoder
        target_masks: torch.BoolTensor = self.create_target_masks(batch_size=batch_size, seq_len=seq_len, target_padding_masks=target_padding_masks, future_masks=future_masks)

        target_masks = target_masks.to(input_tokens.device) # type: ignore[assignment]

        z = self.tok_embed(input_tokens) # TODO: * math.sqrt(self.d_model) for better optimization landscape
        z = z + self.pos_embed[:, :seq_len, :]
        z = self.dropout(z)

        for decoder_block in self.decoder_blocks:
            z  = decoder_block(z, target_masks=target_masks)

        z      = self.layer_norm(z)
        logits: torch.FloatTensor = self.head(z)
        # fmt: on
        return logits

    @torch.no_grad()
    def generate(
        self,
        starting_tokens: torch.LongTensor | List[int],
        *,
        max_tokens: int = 100,  # max tokens to generate
        temperature: float = 1.0,  # temperature for sampling
        greedy: bool = False,  # if True, sample greedily
        top_k: int | None = None,  # if not None, sample from top k tokens
        top_p: float | None = None,  # neclueus sampling
    ) -> torch.LongTensor:
        """
        Generates a sequence of tokens based on the provided
        input tokens.

        This method employs a specified generation strategy that
        can be controlled through parameters like temperature,
        greedy, and top_k. It supports both greedy and probabilistic
        sampling approaches for token generation.

        Parameters
        ----------
        starting_tokens : Union[torch.LongTensor, List[int]]
            The initial tokens / starting_tokens from which the generation begins.
            Can be a list of integers or a LongTensor. It can be a batch of sequences
            too.
            Shape: (L,) or (1, L) or (B, L)
        max_tokens : int
            The maximum number of tokens to generate. Default is 100.
        temperature : float
            Controls the randomness of predictions by scaling the
            logits before applying softmax. Higher values increase
            randomness. Default is 1.0.
        greedy : bool
            If True, the generation uses a greedy approach, selecting
            the most likely next token. If False, uses probabilistic
            sampling. Default is False.
        top_k : int | None
            Limits the sampling pool to the top k most likely tokens.
            If None, no limit is applied. Default is None.
        top_p : float | None
            Limits the sampling pool to the smallest set of tokens
            whose cumulative probability exceeds the threshold p.
            If None, no limit is applied. Default is None.

        Returns
        -------
        torch.LongTensor
            The tensor containing the generated sequence of tokens.

        Notes
        -----
        1. `temperature` affects the distribution sharpness; a lower
        temperature results in a sharper distribution.
        2. **Top-K Sampling**:
            - This method involves selecting the `k` most likely next words from the
            model's output and sampling from this reduced set.
            - The logits that are not in the top `k` are set to a very large negative
            value (like `-inf`), effectively zeroing their probability.
        3. **Top-P (Nucleus) Sampling**:
            - Top-P sampling involves choosing the smallest set of words whose
            cumulative probability exceeds a threshold `p`.
            - You sort the probabilities in descending order and then cumulatively add
            them up until the sum exceeds `p`. Only these words are considered for
            sampling.
        """

        if self.training:
            # a safety check to make sure we are not in training mode
            # this generate could be called outside after training, or during
            # training as a form of validation/evaluation.
            self.eval()

        # NOTE: `starting_tokens` is a list of integers, or a torch.LongTensor of shape (S or T).
        # the distinction between this `starting_tokens` versus the one in `forward` is this is
        # not batched! It is a single sequence of tokens so in order for it to be compatible
        # with the model, we need to expand the first dimension to 1 - making it a batch.
        if isinstance(starting_tokens, list):
            starting_tokens = cast(torch.LongTensor, torch.as_tensor(starting_tokens, dtype=torch.long)[None, ...])

        if starting_tokens.dim() == 1:
            starting_tokens = cast(torch.LongTensor, torch.as_tensor(starting_tokens, dtype=torch.long)[None, ...])  # type: ignore[no-redef]
        assert starting_tokens.dim() == 2, "starting_tokens must be a 1D or 2D tensor"

        for _ in range(max_tokens):
            # if the sequence context is growing too long we must crop it at context_length
            starting_tokens_cropped = (
                starting_tokens[:, -self.config.context_length :]
                if starting_tokens.size(1) > self.config.context_length
                else starting_tokens
            )

            batch_size = starting_tokens_cropped.size(0)
            seq_len = starting_tokens_cropped.size(1)  # this must be less than or equal to self.config.context_length

            target_padding_masks = construct_dummy_batch_target_padding_masks(batch_size, seq_len)
            future_masks = construct_dummy_batch_future_masks(batch_size, seq_len)

            logits = self(
                starting_tokens_cropped,
                target_padding_masks=target_padding_masks,
                future_masks=future_masks,
            )
            assert logits.shape == (batch_size, seq_len, self.config.vocab_size)

            # NOTE: we are only interested in the last token's logits because in
            # autoregressive models, the last token's logits holds the contextual
            # information of all previous tokens (because it is the only token
            # not masked). But in any case, we need this last token's logits to
            # sample the next token.
            logits = logits[:, -1, :]  # shape: (batch_size, vocab_size)
            assert logits.shape == (batch_size, self.config.vocab_size)

            # now scale by temperature
            logits = logits / (temperature + 1e-8)  # add epsilon to prevent division by zero

            # optional cropping of logits to top k
            if top_k is not None:
                top_k_values, _ = torch.topk(logits, k=top_k)
                # The masking out to -inf is to prevent the sampling from
                # non-top k values, effectively making the sampling pool
                # to be only the top k values. We are zeroing out the
                # probabilities of non-top k values.
                logits[logits < top_k_values[:, [-1]]] = float("-inf")

            if top_p is not None:

                def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")
                    return logits

                logits = top_p_logits(logits, top_p)

            # convert logits to softmax probabilities
            probs = torch.softmax(logits, dim=-1)

            # multinomial vs greedy sampling
            # why andrej uses topk instead of torch.argmax(probs, dim=-1, keepdim=True)?
            next_token = (
                torch.multinomial(probs, num_samples=1) if not greedy else torch.topk(probs, k=1, dim=-1).indices
            )

            # append the next token to the input tokens, aka append sampled index
            # to the running sequence context and continue the generation
            starting_tokens = torch.cat([starting_tokens, next_token], dim=1)  # type: ignore[assignment]
        return starting_tokens
