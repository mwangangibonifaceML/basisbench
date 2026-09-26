from __future__ import annotations

import pandas as pd
import numpy as np
import torch
import argparse
import torch.nn as nn

from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class TimeScaler:
    """Affine transform that maps a time column to approximately [-1, 1]."""
    
    minimum: float
    maximum: float
    
    def transform(self, values: np.ndarray) -> np.ndarray:
        gap = self.maximum - self.minimum
        if gap <=0:
            raise ValueError(
                'The training time column must have atleast two values'
            )
        return 2.0 * (values - self.minimum) / gap - 1
    
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
    parser.add_argument("--max-frequency", type=int, default=5)
    parser.add_argument('-hidden-size', type=int, default=10)
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

    #* log1p reduces the effect of the highly skewed bike-count target.
    return (
        torch.from_numpy(time.astype(np.float32)),
        torch.from_numpy(np.log1p(target).astype(np.float32)),
    )

def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y_log: torch.Tensor,
) -> tuple[float, float, float, float]:

    model.eval()

    with torch.no_grad():
        prediction_log = model(x)

        mse_log = nn.functional.mse_loss(
            prediction_log,
            y_log
        ).item()

        prediction = torch.expm1(prediction_log).clamp_min(0.0)
        actual = torch.expm1(y_log)

        mae = torch.mean(
            torch.abs(prediction - actual)
        ).item()

        mse = torch.mean(
            (prediction - actual) ** 2
        ).item()

        rmse = torch.sqrt(
            torch.mean((prediction - actual) ** 2)
        ).item()

    return mse_log, mae, rmse, mse

def load_data():
    args = parse_args()
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.validation)
    test_df = pd.read_csv(args.test)
    
    return (
        train_df,
        val_df,
        test_df
    )