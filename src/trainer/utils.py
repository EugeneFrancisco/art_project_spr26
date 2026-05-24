"""
Shared helpers for trainer implementations.
"""

import numpy as np


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float,
    test_frac: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle rows of `X` and `y` together and partition them into
    train / val / test splits.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    n = len(X)
    if len(y) != n:
        raise ValueError(f"X and y length mismatch: {n} vs {len(y)}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )
