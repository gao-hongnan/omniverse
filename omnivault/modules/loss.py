from enum import Enum

import torch


class Reduction(Enum):
    NONE = "none"
    MEAN = "mean"
    SUM = "sum"


class CrossEntropyLoss:
    def __init__(self, reduction: Reduction = Reduction.MEAN) -> None:
        """
        Initialize the CrossEntropyLoss.

        Parameters
        ----------
        reduction : Reduction, optional
            Specifies the reduction to apply to the output.
            Defaults to Reduction.MEAN.
        """
        self.reduction = reduction

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross entropy loss for a given set of predicted logits and targets.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits tensor of shape (batch_size, ..., vocab_size)
            where ... represents an arbitrary number of dimensions. Note
            `vocab_size` is the last dimension but is used in context of transformer models.
        targets : torch.Tensor
            Target tensor of shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            The computed cross entropy loss, reduced according to the specified reduction strategy.
        """
        if logits.shape[:-1] != targets.shape:
            raise ValueError(
                f"Logits and targets must have compatible shapes. "
                f"Received logits shape: {logits.shape} and targets shape: {targets.shape}"
            )

        if len(logits.shape) > 3:
            raise ValueError(f"Only 1D, 2D, and 3D logits are supported. " f"Received logits shape: {logits.shape}")

        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits_stable = logits - logits_max

        log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1))
        target_logits = logits_stable.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        neg_log_likelihood = -target_logits + log_sum_exp

        if self.reduction == Reduction.NONE:
            return neg_log_likelihood
        elif self.reduction == Reduction.MEAN:
            return torch.mean(neg_log_likelihood)
        elif self.reduction == Reduction.SUM:
            return torch.sum(neg_log_likelihood)
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
