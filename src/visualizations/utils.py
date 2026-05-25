"""
Shared plotting helpers for the `visualizations` package.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(
    data: np.ndarray,
    bins: int = 50,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Count",
    out_path: str | None = None,
) -> None:
    """
    Plot a histogram of `data`. Pass any 1-D array-like (e.g. log prices) and
    optionally an `out_path` to save the figure to disk in addition to showing
    it.
    """
    arr = np.asarray(data).ravel()
    arr = arr[np.isfinite(arr)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins=bins, edgecolor="black")
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=200)
    plt.show()
