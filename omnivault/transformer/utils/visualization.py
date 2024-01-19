from __future__ import annotations

import math
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns


def save_plot_history(history: Dict[str, List[float]], plot: bool = False, save_path: str | None = None) -> None:
    sns.set(style="whitegrid")
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
