import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Type

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer: Type[optim.Optimizer], optimizer_kwargs: Dict[str, Any], metrics=None, predict_fn=None, log_dir='./logs', device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.device = device

        # Default metrics if none are provided
        if metrics is None:
            self.metrics = {
                'accuracy': accuracy_score,
                'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
                'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
                'f1_score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
            }
        else:
            self.metrics = metrics

        # Default prediction function if none is provided
        if predict_fn is None:
            self.predict_fn = self.default_predict
        else:
            self.predict_fn = predict_fn

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_metrics = self._run_epoch(self.train_loader, training=True)
            val_metrics = self._run_epoch(self.val_loader, training=False)

            # Log all metrics
            for metric_name, metric_value in train_metrics.items():
                self.writer.add_scalar(f'{metric_name}/Train', metric_value, epoch)
            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f'{metric_name}/Val', metric_value, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Train Metrics: {train_metrics}")
            print(f"Val Metrics: {val_metrics}")

        self.writer.close()

    def _run_epoch(self, data_loader, training=True):
        mode = 'train' if training else 'val'
        if training:
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_predictions = []

        for inputs, target in tqdm(data_loader, desc=f"{mode.capitalize()} Epoch"):
            if isinstance(inputs, tuple) or isinstance(inputs, list):
                inputs = tuple([x.to(self.device) for x in inputs])
            elif isinstance(inputs, torch.Tensor):
                inputs = tuple([inputs.to(self.device)])
            else:
                raise ValueError(f"unknown type {type(inputs)}")
            target = target.to(self.device)
            if training:
                self.optimizer.zero_grad()

            outputs = self.model(*inputs)
            loss = self.criterion(outputs, target)
            if training:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(self.predict_fn(outputs).cpu().numpy())

        avg_loss = running_loss / len(data_loader)
        metrics = self._calculate_metrics(all_targets, all_predictions)
        metrics['loss'] = avg_loss

        return metrics

    @staticmethod
    def default_predict(outputs):
        _, predicted = torch.max(outputs, 1)
        return predicted

    def _calculate_metrics(self, targets, predictions):
        return {metric_name: metric_fn(targets, predictions) for metric_name, metric_fn in self.metrics.items()}
