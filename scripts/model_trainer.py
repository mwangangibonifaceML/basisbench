from __future__ import annotations

import torch
import pandas as pd
import matplotlib.pyplot as plt

from basisbench.models.taylor import Taylor
from basisbench.models.fourier import Fourier
from basisbench.training.trainer import Trainer
from basisbench.models.neural_network import NeuralNetwork
from basisbench.utils.utils import parse_args
from basisbench.data.datasets import create_split_dataloader

args = parse_args()

neural_net = NeuralNetwork(hidden_size=args.hidden_size)
taylor_model = Taylor(degree=args.degree)
Fourier_model = Fourier(max_frequency=args.max_frequency)

histories = []
results = []

for model in [
    neural_net,
    taylor_model,
    Fourier_model
]:
    trainer = Trainer(
        model=model,
        arguments=args,
        dataloader_factory=create_split_dataloader
    )
    
    result =  trainer.train()
    results.append(
        {
            'model': model.__class__.__name__,
            'best_epoch': result.best_epoch,
            'best_val_rmse': result.best_val_rmse,
            'best_val_mse': result.best_val_mse,
            'test_mse_log': result.test_mse_log,
            'test_mse': result.test_mse,
            'test_rmse': result.test_rmse,
            'test_mae': result.test_mae
        }
    )
    
    history_df = pd.DataFrame(result.history)
    history_df['model'] = model.__class__.__name__
    histories.append(history_df)

all_histories = pd.concat(histories, ignore_index=False)
all_histories.to_csv(args.output_path/'histories.csv', index=False)

results_df = pd.DataFrame(results)
results_df.to_csv(args.output_path/'training_results.csv', index=False)


plt.figure(figsize=(10, 6))

for model_name, group in all_histories.groupby("model"):
    plt.plot(
        group["epoch"],
        group["val_mse_log"],
        label=model_name,
    )

    best_idx = group["val_mse_log"].idxmin()
    best_row = group.loc[best_idx]

    plt.scatter(
        best_row["epoch"],
        best_row["val_mse_log"],
        s=60,
        zorder=3,
    )

    plt.annotate(
        f"epoch {int(best_row['epoch'])}",
        xy=(best_row["epoch"], best_row["val_mse_log"]),
        xytext=(5, 5),
        textcoords="offset points",
    )

plt.xlabel("Epoch")
plt.ylabel("Validation MSE (log target)")
plt.title("Validation MSE by Epoch")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()