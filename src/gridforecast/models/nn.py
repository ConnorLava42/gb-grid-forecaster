"""Rung 3: a PyTorch network that predicts all three quantiles at once.

Install the optional extra first: `uv sync --extra nn`.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from gridforecast.config import QUANTILES
from gridforecast.models.base import QCOLS, time_split

EMBEDDED = ["hh", "dow"]  # categorical inputs, learned as embeddings


class QuantileMLP(nn.Module):
    """Numeric features + embeddings for half-hour and weekday -> (q10, q50, q90).

    The network outputs the median plus two positive gaps, so q10 < q50 < q90
    by construction and the quantiles can never cross.
    """

    def __init__(self, n_numeric: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.hh_emb = nn.Embedding(48, 8)
        self.dow_emb = nn.Embedding(7, 4)
        self.body = nn.Sequential(
            nn.Linear(n_numeric + 12, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, x_num, hh, dow):
        z = torch.cat([x_num, self.hh_emb(hh), self.dow_emb(dow)], dim=1)
        out = self.body(z)
        median = out[:, 1]
        lower = median - nn.functional.softplus(out[:, 0])
        upper = median + nn.functional.softplus(out[:, 2])
        return torch.stack([lower, median, upper], dim=1)


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, quantiles=QUANTILES) -> torch.Tensor:
    """Average pinball loss over the quantiles. pred: (batch, 3), y: (batch,)."""
    losses = []
    for i, q in enumerate(quantiles):
        err = y - pred[:, i]
        losses.append(torch.maximum(q * err, (q - 1) * err))
    return torch.stack(losses).mean()


class MLPQuantile:
    """Scikit-learn-style wrapper: standardises inputs, trains, early-stops."""

    name = "mlp"

    def __init__(
        self,
        hidden: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 1024,
        max_epochs: int = 60,
        patience: int = 6,
        seed: int = 42,
    ):
        self.hidden, self.dropout, self.lr = hidden, dropout, lr
        self.batch_size, self.max_epochs, self.patience, self.seed = (
            batch_size,
            max_epochs,
            patience,
            seed,
        )

    # --- Preparing inputs ---
    def _prepare(self, X: pd.DataFrame):
        num = X[self.numeric_].astype(float)
        num = (num - self.mean_) / self.std_
        num = num.fillna(0.0)  # a missing lag becomes "average" after standardising
        return (
            torch.tensor(num.to_numpy(), dtype=torch.float32),
            torch.tensor(X["hh"].to_numpy(), dtype=torch.long),
            torch.tensor(X["dow"].to_numpy(), dtype=torch.long),
        )

    def _train(self, X, y, epochs=None, X_val=None, y_val=None):
        """Train for `epochs`, or early-stop on the validation set if given."""
        torch.manual_seed(self.seed)
        model = QuantileMLP(len(self.numeric_), self.hidden, self.dropout)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        n_epochs = epochs or self.max_epochs
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        xn, hh, dow = self._prepare(X)
        yt = torch.tensor(((y - self.y_mean_) / self.y_std_).to_numpy(), dtype=torch.float32)
        if X_val is not None:
            vn, vhh, vdow = self._prepare(X_val)
            vy = torch.tensor(
                ((y_val - self.y_mean_) / self.y_std_).to_numpy(), dtype=torch.float32
            )

        best, best_epoch, best_state, wait = np.inf, 0, None, 0
        gen = torch.Generator().manual_seed(self.seed)
        for epoch in range(n_epochs):
            model.train()
            for idx in torch.randperm(len(yt), generator=gen).split(self.batch_size):
                opt.zero_grad()
                loss = pinball_loss(model(xn[idx], hh[idx], dow[idx]), yt[idx])
                loss.backward()
                opt.step()
            sched.step()
            if X_val is None:
                continue
            model.eval()
            with torch.no_grad():
                val = pinball_loss(model(vn, vhh, vdow), vy).item()
            if val < best:
                best, best_epoch, wait = val, epoch + 1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_epoch

    def fit(self, X: pd.DataFrame, y: pd.Series):
        keep = y.notna()
        X, y = X[keep], y[keep]
        self.numeric_ = [c for c in X.columns if c not in EMBEDDED]
        self.mean_, self.std_ = X[self.numeric_].mean(), X[self.numeric_].std().replace(0, 1)
        self.y_mean_, self.y_std_ = y.mean(), y.std()
        X_fit, y_fit, X_val, y_val = time_split(X, y)
        _, self.best_epochs_ = self._train(X_fit, y_fit, X_val=X_val, y_val=y_val)
        # Retrain on everything for the number of epochs that worked best
        self.model_, _ = self._train(X, y, epochs=max(self.best_epochs_, 1))
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(*self._prepare(X)).numpy()
        out = out * self.y_std_ + self.y_mean_
        return pd.DataFrame(out, columns=QCOLS, index=X.index)
