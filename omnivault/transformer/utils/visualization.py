from __future__ import annotations

import math
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def save_plot_history(history: Dict[str, List[float]], plot: bool = False, save_path: str | None = None) -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"

    num_metrics = len(history.keys())
    num_cols = 2
    num_rows = math.ceil(num_metrics / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))

    axs = axs.flatten()

    for i, metric in enumerate(history.keys()):
        axs[i].plot(history[metric])
        axs[i].set_title(metric)
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Remove unused subplots
    for i in range(num_metrics, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()

    if plot:
        plt.show()  # type: ignore[no-untyped-call]

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)


def show_attention_heatmaps(
    attention_weights: torch.Tensor,
    xlabel: str = "Keys",
    ylabel: str = "Queries",
    xticks: List[str] | None = None,
    yticks: List[str] | None = None,
    show_title: bool = False,
    show_values: bool = False,
    value_dp: int = 2,
    figure_kwargs: Dict[str, Any] | None = None,
    plot_kwargs: Dict[str, Any] | None = None,
) -> plt.Figure:
    """
    Visualizes attention heatmaps for a given attention weight tensor.

    Parameters
    ----------
    attention_weights : torch.Tensor
        Attention weight tensor of shape (B, H, _, _).
    xlabel : str, optional
        Label for the x-axis. Default is "Keys".
    ylabel : str, optional
        Label for the y-axis. Default is "Queries".
    xticks : list of str, optional
        Labels for the x-axis ticks. If provided, must have the same length as the number of keys.
    yticks : list of str, optional
        Labels for the y-axis ticks. If provided, must have the same length as the number of queries.
    show_title : bool, optional
        Whether to show titles for each subplot. Default is False.
    show_values : bool, optional
        Whether to display attention weight values on the heatmap. Default is False.
    value_dp : int, optional
        Number of decimal places to display for the attention weight values. Default is 2.
    figure_kwargs : dict, optional
        Additional keyword arguments for the figure creation.
    plot_kwargs : dict, optional
        Additional keyword arguments for the heatmap plot.

    Returns
    -------
    fig : plt.Figure
        The figure object containing the attention heatmaps.

    Raises
    ------
    TypeError
        If `attention_weights` is not a PyTorch tensor.
    ValueError
        If `attention_weights` does not have shape (B, H, _, _).
    """
    if not isinstance(attention_weights, torch.Tensor):
        raise TypeError(f"Attention weights must be a PyTorch tensor, but got {type(attention_weights)}.")

    if len(attention_weights.shape) != 4:
        raise ValueError(f"Attention weights must have shape (B, H, _, _), but got {attention_weights.shape}.")

    B, H, num_queries, num_keys = attention_weights.shape

    if xticks is not None and len(xticks) != num_keys:
        raise ValueError(f"Length of xticks must match the number of keys. Expected {num_keys}, but got {len(xticks)}.")

    if yticks is not None and len(yticks) != num_queries:
        raise ValueError(
            f"Length of yticks must match the number of queries. Expected {num_queries}, but got {len(yticks)}."
        )

    attention_weights = attention_weights.cpu().detach().numpy()

    figure_kwargs = figure_kwargs or {"figsize": (15, 15), "sharex": True, "sharey": True, "squeeze": False}
    fig, axes = plt.subplots(B, H, **figure_kwargs)

    plot_kwargs = plot_kwargs or {"cmap": "viridis"}
    if B == 1:
        attention_weight = attention_weights[0]
        for h, (ax, head_attention) in enumerate(zip(axes, attention_weight)):
            sns.heatmap(head_attention, ax=ax, annot=show_values, fmt=f".{value_dp}f", **plot_kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            if xticks is not None:
                ax.set_xticks(np.arange(num_keys) + 0.5)
                ax.set_xticklabels(xticks, rotation=45, ha="right")
            if yticks is not None:
                ax.set_yticks(np.arange(num_queries) + 0.5)
                ax.set_yticklabels(yticks)
            if show_title:
                ax.set_title(f"Head {h + 1}")
    else:
        for b, (row_axes, attention_weight) in enumerate(zip(axes, attention_weights)):
            for h, (ax, head_attention) in enumerate(zip(row_axes, attention_weight)):
                sns.heatmap(head_attention, ax=ax, annot=show_values, fmt=f".{value_dp}f", **plot_kwargs)
                if b == B - 1:
                    ax.set_xlabel(xlabel)  # Only the last batch will have the xlabel
                if h == 0:
                    ax.set_ylabel(ylabel)  # Only the first head will have the ylabel

                if xticks is not None:
                    ax.set_xticks(np.arange(num_keys) + 0.5)
                    ax.set_xticklabels(xticks, rotation=45, ha="right")
                if yticks is not None:
                    ax.set_yticks(np.arange(num_queries) + 0.5)
                    ax.set_yticklabels(yticks)

                if show_title:
                    ax.set_title(f"Batch {b + 1}, Head {h + 1}")

    fig.subplots_adjust(wspace=0.2, hspace=0.2)  # Adjust subplot spacing

    plt.show()  # type: ignore[no-untyped-call]

    return fig
