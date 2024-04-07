def noam_lr_decay(step: int, *, d_model: int, warmup_steps: int) -> float:
    """
    Calculate the learning rate based on the current step, model dimensionality,
    and the number of warmup steps.

    Parameters
    ----------
    step : int
        The current step in the training process.
    d_model : int
        The dimensionality of the model.
    warmup_steps : int
        The number of steps during which the learning rate is increased linearly.
        After reaching `warmup_steps`, the learning rate decays proportionally to the
        inverse square root of the step number.

    Returns:
        float: The calculated learning rate.
    """

    return float(d_model ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * warmup_steps ** (-1.5)))
