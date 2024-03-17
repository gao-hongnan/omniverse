"""See MosaicML's Composer Speed Monitor Callback here:
https://github.com/mosaicml/composer/blob/dev/composer/callbacks/speed_monitor.py
"""

def estimate_mfu(
    num_decoder_blocks: int,
    num_heads: int,
    d_model: int,
    context_length: int,
    model_total_parameters: int,
    effective_batch_size_per_iter: int,
    time_taken_per_step: float,
    gpu_promised_flops: float = 312e12, # A100 GPU bfloat16/float16 peak flops is 312 TFLOPS
) -> float:
    """
    Estimate Model FLOPs Utilization (MFU) as a ratio of achieved FLOPs to
    the A100 GPU's peak FLOPs capability.

    Parameters
    ----------
    num_decoder_blocks : int
        Number of decoder blocks in the Transformer model.
    num_heads : int
        Number of attention heads in each Transformer block.
    d_model : int
        Dimension of the model's embeddings.
    context_length : int
        Number of tokens in each input sequence.
    model_total_parameters : int
        Total number of learnable parameters in the model.
    effective_batch_size_per_iter : int
        Effective batch size processed in one iteration, accounting for
        gradient accumulation.
    time_taken_per_step : float
        Time taken per training iteration in seconds.
    gpu_promised_flops : float, optional
        Theoretical peak performance of the GPU in FLOPs (default is
        312e12 for A100 GPU bfloat16/float16 operations).

    Returns
    -------
    mfu : float
        Estimated MFU, indicating the percentage of the GPU's computational
        capacity effectively utilized by the model during training.

    Example
    -------
    >>> estimate_mfu(
    ...     num_decoder_blocks=6,
    ...     num_heads=8,
    ...     d_model=512,
    ...     context_length=1024,
    ...     model_total_parameters=1_000_000,
    ...     effective_batch_size_per_iter=20 * 8,  # 20 sequences per GPU, 8 GPUs
    ...     time_taken_per_step=0.1, # 0.1 seconds per iteration
    ...     gpu_promised_flops=312e12, # A100 GPU bfloat16/float16 peak flops
    ... )

    Notes
    -----
    This function utilizes the formula from the PaLM paper Appendix B
    (https://arxiv.org/abs/2204.02311) for estimating the FLOPs required
    for one forward and backward pass of a single token and scales it up to
    the effective batch size and the given model architecture to calculate MFU.
    You can likely use it as a callback in your `Trainer` to log the MFU during training.
    """
    # fmt: off
    N, L, H, Q, T = model_total_parameters, num_decoder_blocks, num_heads, d_model // num_heads, context_length
    flops_per_token_per_fwdbwd = 6 * N + 12 * L * H * Q * T # 1 token forward and backward flops
    flops_per_sequence_per_fwdbwd = flops_per_token_per_fwdbwd * T # 1 sequence = T tokens
    flops_per_iter_per_fwdbwd = flops_per_sequence_per_fwdbwd * effective_batch_size_per_iter # 1 iter means if batch size is 100, then 100 sequences are processed in 1 iter
    # express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved_per_second = flops_per_iter_per_fwdbwd * (1.0 / time_taken_per_step)  # per second
    mfu = flops_achieved_per_second / gpu_promised_flops
    # fmt: on
    return mfu
