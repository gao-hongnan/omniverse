def noam_lr_decay(step: int, d_model: int, warmup_steps: int) -> float:
    """
    Calculate the learning rate based on the current step, model dimensionality,
    and the number of warmup steps.

    Args:
        step (int): The current step in the training process.
        d_model (int): The dimensionality of the model.
        warmup_steps (int): The number of warmup steps.

    Returns:
        float: The calculated learning rate.
    """
    return d_model ** (-0.5) * min(
        (step + 1) ** (-0.5), (step + 1) * warmup_steps ** (-1.5)
    )

def get_linear_schedule_with_warmup() -> None:
    ...
