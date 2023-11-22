from typing import List, Tuple

import torch

# FIXME
EQUAL_SIGN = 13
PAD = 16


def construct_future_mask(seq_len: int) -> torch.BoolTensor:
    """
    Construct a binary mask that contains 1's for all valid connections and 0's for
    all outgoing future connections. This mask will be applied to the attention
    logits in decoder self-attention such that all logits with a 0 mask are set to
    -inf.

    Parameters
    ----------
    seq_len : int
        Length of the input sequence.

    Returns
    -------
    torch.BoolTensor
        (seq_len, seq_len) mask
    """
    future_masks = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).to(torch.bool)
    future_masks = future_masks.contiguous()
    return future_masks == 0


def construct_input_tensors(x: torch.Tensor) -> torch.LongTensor:
    return x[:, :-1].long()


def construct_target_tensors(
    equations: torch.Tensor, equal_sign_locs: List[int], pad_token_id: int
) -> torch.LongTensor:
    """
    Create target tensors by masking out the tokens before the equal sign.

    Parameters
    ----------
    equations : torch.Tensor
        Batch of equations.
    equal_sign_locs : List[int]
        Locations of the EQUAL_SIGN in each equation.
    pad_token_id : int
        ID of the pad token.

    Returns
    -------
    torch.Tensor
        Target tensors.
    """
    targets = [
        torch.cat(
            (
                torch.tensor([pad_token_id] * loc, dtype=torch.long),
                equations[i, loc + 1 :],
            )
        )
        for i, loc in enumerate(equal_sign_locs)
    ]
    return torch.stack(targets).long()


def construct_padding_mask(inputs: torch.Tensor, pad_token_id: int) -> torch.BoolTensor:
    """
    Create a padding mask for the inputs.

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensors.

    Returns
    -------
    torch.BoolTensor
        Padding mask tensor.
    """
    batch_size, seq_len = inputs.size()
    padding_mask = inputs != pad_token_id
    # fmt: off
    return padding_mask.view(batch_size, 1, 1, seq_len).expand(batch_size, 1, seq_len, seq_len)
    # fmt: on


def construct_batches(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # drops last column of the inputs, meaning remove last element of each row
    inputs = construct_input_tensors(x)
    batch_size, seq_len = inputs.size()

    equal_sign_loc: List[int] = [
        (equation == EQUAL_SIGN).nonzero(as_tuple=True)[0].item() for equation in x
    ]

    # Mask out the tokens before the equal sign
    targets = construct_target_tensors(x, equal_sign_loc, pad_token_id=PAD)

    future_masks = construct_future_mask(seq_len)
    # future mask has shape (L, L) but we want it to be (B, L, L) then (B, 1, L, L)
    future_masks = (
        future_masks.view(1, seq_len, seq_len)
        .expand(size=(batch_size, -1, -1))
        .unsqueeze(1)
    )

    # padding_masks before view has shape: (batch_size, seq_len)
    # we want it to be (B, L, L) then (B, 1, L, L)
    padding_masks = construct_padding_mask(inputs, pad_token_id=PAD)
    return inputs, targets, padding_masks, future_masks
