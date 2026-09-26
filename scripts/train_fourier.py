from __future__ import annotations

import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from basisbench.models.fourier import Fourier
from basisbench.utils.utils import (
    load_data,
    numeric_column,
    parse_args, 
    TimeScaler,
    choose_time_column,
    make_tensors,
    evaluate
)

def main():
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
    
    model = Fourier(max_frequency= args.max_frequency)
    with torch.no_grad():
        model.cos_coefficients.zero_()
        model.sin_coefficients.zero_()
        model.bias.copy_(y_train.mean(dim=-1).to(model.bias.device, dtype=model.bias.dtype))

    optimizer = torch.optim.AdamW(model.parameters(), 
                            lr=args.learning_rate,
                            weight_decay= args.weight_decay)
    
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        train_loss = nn.functional.mse_loss(model(x_train), y_train)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip)
        optimizer.step()
        
        val_mse_log, val_mae, val_rmse, val_loss= evaluate(model, x_val, y_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"epoch={epoch:04d} train_mse_log={train_loss.item():.6f} "
                f"val_mse_log={val_mse_log:.6f} val_mae={val_mae:.3f}"
                f" val_rmse= {val_rmse:.4f} val_mse={val_loss:.4f}"
            )
        if epochs_without_improvement >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    test_mse_log, test_mae, test_rmse, test_loss = evaluate(model, x_test, y_test)
    print(f"test_mse_log={test_mse_log:.6f} test_mae={test_mae:.3f} test_rmse={test_rmse:.5f} test_mse={test_loss:.5f}")

if __name__ == '__main__':
    main()