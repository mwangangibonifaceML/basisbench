"""Train the Taylor polynomial model on Bike Sharing data.

The model uses exactly one input: a normalized time coordinate. The default
preprocessing maps time to [-1, 1] using statistics from the training split and
applies log1p to the non-negative ``cnt`` target. Both choices help prevent
large polynomial features and target magnitudes from destabilizing training.

Example:
    python scripts/train_taylor.py --degree 5

For the raw Bike Sharing files, use a time-like numeric column such as
``instant`` or ``dteday``. For the existing processed files, ``timenormalized``
is used by default when present.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from basisbench.models.taylor import Taylor


@dataclass(frozen=True)
class TimeScaler:
    """Affine transform that maps a time column to approximately [-1, 1]."""

    minimum: float
    maximum: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        span = self.maximum - self.minimum
        if span <= 0:
            raise ValueError("the training time column must contain at least two values")
        return 2.0 * (values - self.minimum) / span - 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/proccessed/train_set.csv"))
    parser.add_argument("--validation", type=Path, default=Path("data/proccessed/val_set.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/proccessed/test_set.csv"))
    parser.add_argument(
        "--time-column",
        default=None,
        help="Numeric time column. Defaults to timenormalized, or instant if unavailable.",
    )
    parser.add_argument("--target-column", default="cnt")
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/taylor"))
    return parser.parse_args()


def choose_time_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"time column {requested!r} is not present in the CSV")
        return requested

    for candidate in ("timenormalized", "instant"):
        if candidate in frame.columns:
            return candidate

    raise ValueError(
        "could not find a time column; pass --time-column with a numeric column "
        "such as 'instant'"
    )


def numeric_column(frame: pd.DataFrame, column: str, split_name: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"{column!r} is missing from the {split_name} CSV")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{column!r} contains missing or non-numeric values in {split_name}")
    return values


def make_tensors(
    frame: pd.DataFrame,
    split_name: str,
    time_column: str,
    target_column: str,
    scaler: TimeScaler,
) -> tuple[torch.Tensor, torch.Tensor]:
    time = scaler.transform(numeric_column(frame, time_column, split_name))
    target = numeric_column(frame, target_column, split_name)
    if (target < 0).any():
        raise ValueError("cnt must be non-negative when using the log1p target transform")

    # log1p reduces the effect of the highly skewed bike-count target.
    return (
        torch.from_numpy(time.astype(np.float32)),
        torch.from_numpy(np.log1p(target).astype(np.float32)),
    )


def evaluate(model: nn.Module, x: torch.Tensor, y_log: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        prediction_log = model(x)
        mse_log = nn.functional.mse_loss(prediction_log, y_log).item()
        prediction = torch.expm1(prediction_log).clamp_min(0.0)
        actual = torch.expm1(y_log)
        mae = torch.mean(torch.abs(prediction - actual)).item()
    return mse_log, mae


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.patience < 0:
        raise ValueError("epochs must be positive and patience must be non-negative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.validation)
    test_df = pd.read_csv(args.test)

    time_column = choose_time_column(train_df, args.time_column)
    train_time = numeric_column(train_df, time_column, "training")
    scaler = TimeScaler(float(train_time.min()), float(train_time.max()))

    x_train, y_train = make_tensors(train_df, "training", time_column, args.target_column, scaler)
    x_val, y_val = make_tensors(val_df, "validation", time_column, args.target_column, scaler)
    x_test, y_test = make_tensors(test_df, "test", time_column, args.target_column, scaler)

    model = Taylor(degree=args.degree)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = loss_fn(model(x_train), y_train)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        val_loss, val_mae = evaluate(model, x_val, y_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"epoch={epoch:04d} train_mse_log={train_loss.item():.6f} "
                f"val_mse_log={val_loss:.6f} val_mae={val_mae:.3f}"
            )

        if epochs_without_improvement >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    test_mse_log, test_mae = evaluate(model, x_test, y_test)
    print(f"test_mse_log={test_mse_log:.6f} test_mae={test_mae:.3f}")

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "degree": args.degree,
            "time_column": time_column,
            "target_column": args.target_column,
            "target_transform": "log1p",
            "time_scaler": asdict(scaler),
        },
        args.output / "model.pt",
    )
    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "test_mse_log": test_mse_log,
                "test_mae": test_mae,
                "best_validation_mse_log": best_val_loss,
                "time_column": time_column,
                "target_column": args.target_column,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
