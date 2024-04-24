import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from typing_extensions import TypeAlias

# from torch.optim.optimizer import params_t as ParamsT

__all__ = ["AdamW"]

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        self.validate_params(lr, betas, eps, weight_decay)

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

        # NOTE: state/param_groups are stated from pytorch optim source for clarity
        # self.state: DefaultDict[torch.Tensor, Any] = defaultdict(dict)
        # self.param_groups: List[Dict[str, Any]] = []

    @staticmethod
    def validate_params(lr: float, betas: Tuple[float, float], eps: float, weight_decay: float) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}, must be >= 0.")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}, must be >= 0.")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}, must be >= 0.")
        if not 0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta values: {betas}, must be in [0, 1).")

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss
