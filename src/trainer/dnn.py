"""
A simple DNN-based Vec2Price implementation.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.tokenizers.tokenizer import Token, Tokenizer
from src.trainer.trainer import Vec2Price
from src.trainer.utils import train_val_test_split


class _PriceDNN(nn.Module):
    """Small feed-forward net: embedding vector -> scalar prediction."""

    def __init__(self, embedding_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = embedding_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DNNVec2Price(Vec2Price):
    """
    Maps tokenizer embedding vectors to predicted prices via a small MLP.
    Trained with Adam + MSE; L2 regularization comes from Adam's `weight_decay`
    argument.

    The class is agnostic to how the caller scales prices: the targets are used
    verbatim, and predictions are returned in whatever space the caller trained
    on. If you want to fit on log-prices, log them before passing them in and
    exponentiate the predictions yourself.

    The constructor embeds the names and splits `(X, y)` into train/val/test
    arrays stored as member variables.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        names: list[str | Token],
        prices: np.ndarray,
        hidden_dims: tuple[int, ...] = (128, 64),
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 100,
        batch_size: int = 32,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 0,
        device: str = "cpu",
        model_dir: str = "models/dnns",
        model_name: str | None = None,
    ):
        super().__init__(tokenizer)
        self.hidden_dims = tuple(hidden_dims)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model_dir = Path(model_dir)
        self.model_name = model_name or f"dnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

        torch.manual_seed(seed)
        self.model = _PriceDNN(self.embedding_dim, self.hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.loss_fn = nn.MSELoss()

    def _tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).to(self.device)

    def _mse(self, X: np.ndarray, y: np.ndarray) -> float:
        if len(X) == 0:
            return float("nan")
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self._tensor(X))
            return float(self.loss_fn(preds, self._tensor(y)).item())

    def train(self) -> None:
        dataset = TensorDataset(
            self._tensor(self.X_train), self._tensor(self.y_train)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            self.model.train()
            for xb, yb in loader:
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
            print(
                f"epoch {epoch + 1:3d}/{self.num_epochs} "
                f"train_mse={self._mse(self.X_train, self.y_train):.4f} "
                f"val_mse={self._mse(self.X_val, self.y_val):.4f}"
            )

        self.model_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.model_dir / f"{self.model_name}.pt"
        torch.save(self.model.state_dict(), save_path)
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
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0).to(self.device)
            return float(self.model(x).item())

    def get_preds(self, names: list[str | Token]) -> np.ndarray:
        X = self.tokenizer.get_vectors(names, include_all=True).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(self._tensor(X)).cpu().numpy()

    def load_model(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
