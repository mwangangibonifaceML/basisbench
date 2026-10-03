from __future__ import annotations

import copy
import json
import torch
import torch.nn as nn
import argparse
import pandas as pd
import logging

from collections.abc import Callable
from dataclasses import dataclass, field
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from basisbench.data.datasets import create_split_dataloader
from basisbench.models.neural_network import NeuralNetwork
from basisbench.models.fourier import Fourier
from basisbench.models.taylor import Taylor
from basisbench.utils.utils import (
    evaluate,
    TimeScaler,
    load_data
)
from basisbench.data.proccess_data import (
    load_raw_data,
    choose_time_column,
    numeric_column,
    preprocess_data)

from torch.optim.optimizer import Optimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingResult:
    """Class for keeping track of training results."""
    best_epoch: int
    epochs_trained: int
    
    best_val_mse_log: float
    best_val_mae: float
    best_val_rmse: float
    best_val_mse: float
    
    test_mse_log: float
    test_mae: float
    test_rmse: float
    test_mse: float
    
    parameter_count: int

    history: list[dict[str, float]] = field(default_factory=list)

class Trainer:
    def __init__(self,
                model: nn.Module,
                arguments: argparse.Namespace,
                dataloader_factory: Callable[[pd.DataFrame, str, TimeScaler, int], DataLoader]
                ) -> None:
        if arguments.epochs <= 0 or arguments.patience <0:
            raise ValueError(
                'Epochs be non-zero and patience must be non-negative'
            )
            
        self.model = model
        self.args = arguments
        self.dataloader_factory = dataloader_factory
        self.loss_fn = nn.MSELoss(reduction='mean')
        
        #* optimizer selection based on user input
        optimizer_registry: dict[str, type[Optimizer]] = {
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
        }
        
        if arguments.optimizer not in optimizer_registry:
            raise ValueError(
                f"Invalid optimizer choice: {arguments.optimizer}. "
                f"Valid options are: {list(optimizer_registry.keys())}"
            )
        
        optimizer_cls = optimizer_registry[arguments.optimizer]

        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.args.learning_rate, 
            weight_decay=self.args.weight_decay
        )
        
        
    def create_dataloaders(self: Trainer) -> tuple[DataLoader, DataLoader, DataLoader]:
        #* load the data, process it and create the dataloaders
        preprocess_data(self.args.input_path, self.args.output_path)
        train_df, val_df, test_df = load_data(self.args)
        time_column = choose_time_column(train_df, self.args.time_column)
        train_time = numeric_column(train_df, time_column, "training")
        scaler = TimeScaler(float(train_time.min()), float(train_time.max()))
        
        #* create dataloaders for training, validation, and testing
        train_loader = self.dataloader_factory(train_df, 
                                    split_name='train',
                                    scaler=scaler,
                                    batch_size=self.args.batch_size)
        val_loader = self.dataloader_factory(val_df, 
                                    split_name='val',
                                    scaler=scaler,
                                    batch_size=self.args.batch_size)
        test_loader = self.dataloader_factory(test_df, 
                                    split_name='test',
                                    scaler=scaler,
                                    batch_size=self.args.batch_size)
        return train_loader, val_loader, test_loader
    
    def _train_batch(self, x_batch, y_batch):
        """
        Process batch by running the forward pass
        backward pass and parameter update
        """
        self.optimizer.zero_grad(set_to_none=True)                  #* reset the previous gradients
        train_loss = self.loss_fn(self.model(x_batch), y_batch)     #* compute the training loss
        train_loss.backward()                                       #* compute the gradients via backpropagation
    
        grad_norm = clip_grad_norm_(                                #* clip the gradients to prevent exploding gradients
            self.model.parameters(), 
            self.args.gradient_clip)
        
        self.optimizer.step()                                       #* update the model parameters based on the gradients
        return train_loss, grad_norm
    
    def _process_data_loader(self, loader: DataLoader):
        #* reset training metrics for this epoch
        running_loss = 0.0
        num_samples = 0
        
        for x_batch, y_batch in loader:                             #* iterate over the training data
            train_loss,_ = self._train_batch(x_batch, y_batch)      #* process one batch at time
            running_loss += train_loss.item() * x_batch.size(0)     #* accumulate the training loss
            num_samples += x_batch.size(0)                          #* accumulate the number of samples processed
            
        return running_loss, num_samples
    
    def train(self) -> TrainingResult:    
        #* load the data and create the dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders()
        
        best_state = copy.deepcopy(self.model.state_dict())
        best_val_mse_log: float = float("inf")
        best_val_mae: float = float("inf")
        best_val_rmse: float = float("inf")
        best_val_mse: float = float("inf")
        best_epoch: int = 0
        
        epochs_trained: int = 0
        epochs_without_improvement = 0
        
        history  = []  # Optional: Store training history if needed

        for epoch in range(1, self.args.epochs + 1):
            #* -------------------------------------
            #* Training Phase
            #* -------------------------------------
            self.model.train()                                      #* set the model to training mode
            running_loss, num_samples = self._process_data_loader(train_loader)
            train_loss_epoch = running_loss / num_samples           #* compute the average training loss for this epoch
                
            
            #* -------------------------------------
            #* Validation Phase
            #* -------------------------------------
            val_mse_log,val_mse, val_rmse, val_mae = evaluate(     
                self.model, val_loader, self.loss_fn)                    #* evaluate the model on the validation set
            epochs_trained = epoch                                   #* update the number of epochs trained
            
            history.append(
                {
                    'epoch': epoch,
                    'train_mse_log': train_loss_epoch,
                    'val_mse_log': val_mse_log,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'val_mse': val_mse
                    }
                )
            
            #* -------------------------------------
            #* Early Stopping Check
            #* -------------------------------------
            if val_mse_log < best_val_mse_log:                            #* check if the validation loss improved
                best_val_mse_log = val_mse_log                            #* update the best validation loss
                best_val_mae = val_mae
                best_val_rmse = val_rmse
                best_val_mse = val_mse
                best_epoch = epoch
                
                best_state = copy.deepcopy(self.model.state_dict())  #* save the model state
                epochs_without_improvement = 0                       #* reset the counter for epochs without improvement
            else:
                epochs_without_improvement += 1                      #* increment the counter for epochs without improvement
                
            #* -------------------------------------
            #* Logging
            #* -------------------------------------
            if epoch == 1 or epoch % 5 == 0:                         #* log the metrics every 5 epochs
                logger.info(
                    f" epoch={epoch:04d} train_mse_log={train_loss_epoch:.6f} "
                    f" val_mse_log={val_mse_log:.6f} val_mae={val_mae:.3f} "
                    f" val_rmse= {val_rmse:.4f} val_mse={val_mse:.4f}"
                )
                
            if epochs_without_improvement >= self.args.patience:    #* check for early stopping
                logger.info(f"early stopping at epoch {epoch}")
                break
            
        #* -------------------------------------
        #* Load the best model state and evaluate on the test set
        #* -------------------------------------
        self.model.load_state_dict(best_state)                       #* load the best model state
        
        #* -------------------------------------
        #* Test Phase
        #* evaluate the model on the test set
        #* -------------------------------------
        test_mse_log,test_mse, test_rmse, test_mae = evaluate(      #* evaluate the model on the test set
            self.model, test_loader, self.loss_fn)
        
        logger.info('Training completed.\n')
        logger.info('Test set evaluation metrics:')
        logger.info(f"test_mse_log={test_mse_log:.6f} "
                    f"test_mae={test_mae:.3f} "
                    f"test_rmse={test_rmse:.5f} test_mse={test_mse:.5f}")
        
        parameter_count = sum(
            p.numel()
            for p in self.model.parameters() if p.requires_grad)    #* count the number of trainable parameters

        
        logger.info(
            f'best_epoch={best_epoch} '
            f'best_val_mse_log={best_val_mse_log:.6f} '
            f'test_mae={test_mae:.3f} '
            f'test_rmse={test_rmse:.5f} '
            f'test_mse={test_mse:.5f} '
            f'test_mse_log={test_mse_log:.6f} '
        )
        
        return TrainingResult(
            best_epoch=epoch,
            epochs_trained=epochs_trained,
            best_val_mse_log=best_val_mse_log,
            best_val_mae=best_val_mae,
            best_val_rmse=best_val_rmse,
            best_val_mse=best_val_mse,
            test_mse_log=test_mse_log,
            test_mae=test_mae,
            test_rmse=test_rmse,
            test_mse=test_mse,
            parameter_count=parameter_count,
            history= history
        )