"""
A KNN-based Vec2Price implementation that predicts a token's price from the K
most cosine-similar artists in the training set.
"""

from datetime import datetime
from pathlib import Path

import numpy as np

from src.tokenizers.tokenizer import Token, Tokenizer
from src.trainer.trainer import Vec2Price
from src.trainer.utils import train_val_test_split


class KNNVec2Price(Vec2Price):
    """
    Standard KNN regression over the training set, using cosine similarity
    between tokenizer embeddings.

    For a query token, predictions come from the K training artists whose
    embeddings are most cosine-similar to the query. `weighted` picks between a
    plain mean of those neighbors' prices and a mean weighted by cosine
    similarity.

    There is no fitting step. `train()` just snapshots the (X_train, y_train)
    "model" to disk so `load_model` can restore it later.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        names: list[str | Token],
        prices: np.ndarray,
        k: int = 5,
        weighted: bool = False,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 0,
        model_dir: str = "models/knns",
        model_name: str | None = None,
    ):
        super().__init__(tokenizer)
        self.k = k
        self.weighted = weighted
        self.model_dir = Path(model_dir)
        self.model_name = model_name or f"knn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        prices = np.asarray(prices, dtype=np.float32)
        if len(prices) != len(names):
            raise ValueError(
                f"names and prices length mismatch: {len(names)} vs {len(prices)}"
            )

        X = tokenizer.get_vectors(names, include_all=True).astype(np.float32)
        y = prices

        (
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test,
        ) = train_val_test_split(X, y, val_frac, test_frac, seed=seed)

        # Cache unit-normalized training rows so cosine similarity reduces to a
        # single matmul at query time.
        self._X_train_unit = self._unit_rows(self.X_train)

    @staticmethod
    def _unit_rows(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return X / norms

    def _predict_from_vecs(self, Q: np.ndarray) -> np.ndarray:
        k = min(self.k, len(self.X_train))
        Q_unit = self._unit_rows(Q.astype(np.float32))
        sims = Q_unit @ self._X_train_unit.T

        topk_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(len(Q))[:, None]
        topk_sims = sims[rows, topk_idx]
        topk_prices = self.y_train[topk_idx]

        if self.weighted:
            return (
                (topk_sims * topk_prices).sum(axis=1) / topk_sims.sum(axis=1)
            ).astype(np.float32)
        return topk_prices.mean(axis=1).astype(np.float32)

    def _mse(self, X: np.ndarray, y: np.ndarray) -> float:
        if len(X) == 0:
            return float("nan")
        preds = self._predict_from_vecs(X)
        return float(np.mean((preds - y) ** 2))

    def train(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.model_dir / f"{self.model_name}.npz"
        np.savez(save_path, X_train=self.X_train, y_train=self.y_train)
        print(
            f"k={self.k} weighted={self.weighted} "
            f"train_mse={self._mse(self.X_train, self.y_train):.4f} "
            f"val_mse={self._mse(self.X_val, self.y_val):.4f}"
        )
        print(f"saved model to {save_path}")

    def test(self) -> float:
        return self._mse(self.X_test, self.y_test)

    def test_baseline(self) -> float:
        if len(self.y_test) == 0:
            return float("nan")
        return float(np.var(self.y_test))

    def get_pred(self, name: str | Token) -> float:
        vec = self.tokenizer.get_vector(name)
        if vec is None:
            raise KeyError(f"Token {name!r} is not in the tokenizer vocabulary.")
        return float(self._predict_from_vecs(vec[None, :])[0])

    def get_preds(self, names: list[str | Token]) -> np.ndarray:
        Q = self.tokenizer.get_vectors(names, include_all=True).astype(np.float32)
        return self._predict_from_vecs(Q)

    def load_model(self, path: str) -> None:
        data = np.load(path)
        self.X_train = data["X_train"].astype(np.float32)
        self.y_train = data["y_train"].astype(np.float32)
        self._X_train_unit = self._unit_rows(self.X_train)
