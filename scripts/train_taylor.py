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

import copy
import json
import pandas as pd
import numpy as np

import torch
from torch import nn

from basisbench.models.taylor import Taylor
from basisbench.utils.utils import (make_tensors,
                                TimeScaler, load_data,
                                numeric_column,
                                choose_time_column,
                                evaluate,parse_args)

def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.patience < 0:
        raise ValueError("epochs must be positive and patience must be non-negative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_df, val_df, test_df = load_data()

    time_column = choose_time_column(train_df, args.time_column)
    train_time = numeric_column(train_df, time_column, "training")
    scaler = TimeScaler(float(train_time.min()), float(train_time.max()))

    x_train, y_train = make_tensors(train_df, "training", time_column, args.target_column, scaler)
    x_val, y_val = make_tensors(val_df, "validation", time_column, args.target_column, scaler)
    x_test, y_test = make_tensors(test_df, "test", time_column, args.target_column, scaler)

    x_val = 2.0 * (x_val - x_val.min()) / (x_val.max() - x_val.min()) - 1
    y_val = 2.0 * (y_val - y_val.min()) / (y_val.max() - y_val.min()) - 1
    x_test = 2.0 * (x_test - x_test.min()) / (x_test.max() - x_test.min()) - 1
    y_test = 2.0 * (y_test - y_test.min()) / (y_test.max() - y_test.min()) - 1
    
    model = Taylor(degree=args.degree)
    with torch.no_grad():
        model.coefficients.zero_()
        model.coefficients[0] = y_train.mean(dim=-1)

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
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        val_mse_log, val_mae, val_rmse, val_mse = evaluate(model, x_val, y_val)
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"epoch={epoch:04d} train_mse_log={train_loss.item():.6f} "
                f"val_mse_log={val_mse_log:.6f} val_mae={val_mae:.3f}"
                f" val_rmse= {val_rmse:.4f} val_mse={val_mse:.4f}"
            )

        if epochs_without_improvement >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    test_mse_log, test_mae, test_rmse, test_mse = evaluate(model, x_test, y_test)
    print(f"test_mse_log={test_mse_log:.6f} test_mae={test_mae:.3f} test_rmse={test_rmse:.5f} test_mse={test_mse:.5f}")

    import matplotlib.pyplot as plt

    x_probe = torch.linspace(-1.0, 1.9, 500)
    with torch.no_grad():
        y_probe_log = model(x_probe)
        y_probe = torch.expm1(y_probe_log).clamp_min(0)
        
    plt.figure(figsize=(10, 5))
    plt.plot(x_probe.numpy(), y_probe_log.numpy())
    plt.axvline(-1, linestyle="--")
    plt.axvline(1, linestyle="--")
    plt.xlabel("Normalized time")
    plt.ylabel("Predicted log(count + 1)")
    plt.title("Taylor polynomial extrapolation in log space")
    plt.show()
    
    with torch.no_grad():
        y_probe = torch.expm1(y_probe_log).clamp_min(0)

    plt.figure(figsize=(10, 5))
    plt.plot(x_probe.numpy(), y_probe.numpy())
    plt.axvline(-1, linestyle="--")
    plt.axvline(1, linestyle="--")
    plt.ylim(bottom=0)
    plt.xlabel("Normalized time")
    plt.ylabel("Predicted bike rentals")
    plt.title("Taylor polynomial extrapolation in original scale")
    plt.show()
    # args.output.mkdir(parents=True, exist_ok=True)
    # torch.save(
    #     {
    #         "model_state_dict": model.state_dict(),
    #         "degree": args.degree,
    #         "time_column": time_column,
    #         "target_column": args.target_column,
    #         "target_transform": "log1p",
    #         "time_scaler": asdict(scaler),
    #     },
    #     args.output / "model.pt",
    # )
    # (args.output / "metrics.json").write_text(
    #     json.dumps(
    #         {
    #             "test_mse_log": test_mse_log,
    #             "test_mae": test_mae,
    #             "best_validation_mse_log": best_val_loss,
    #             "time_column": time_column,
    #             "target_column": args.target_column,
    #         },
    #         indent=2,
    #     )
    #     + "\n",
    #     encoding="utf-8",
    # )


if __name__ == "__main__":
    main()
