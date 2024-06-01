import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Type
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from caser_datasets.sequential_recommender import SequentialRecommenderDataModule
from utils.trainer import Trainer
from models.caser import Caser, CaserCriterion
from caser_datasets.movielens import MovielensDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from sklearn.base import BaseEstimator, RegressorMixin

def get_early_stopping_callback() -> EarlyStopping:
    return EarlyStopping(
        monitor='val_loss', patience=self.hparams.training_config.es_patience, mode='min', 
        verbose=True, min_delta=self.hparams.training_config.es_min_delta
    )

class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Collect the training metrics
        self.train_metrics.append(trainer.callback_metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Collect the validation metrics
        self.val_metrics.append(trainer.callback_metrics)

    def on_test_epoch_end(self, trainer, pl_module):
        # Collect the test metrics
        self.test_metrics.append(trainer.callback_metrics)

class GSRunner(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            recommender_data_module: SequentialRecommenderDataModule,
            model: Type[Caser],
            model_fixed_kwargs: Dict[str, Any],
            trainer_kwargs: Dict[str, Any]
        ): 
        self.data_module = recommender_data_module
        self.model_fixed_kwargs = model_fixed_kwargs
        self.model = model
        self.trainer_kwargs = trainer_kwargs

    def run(self):
        pass

    def single_run(self, gs_kwargs: Dict[str, Any]) -> float:
        model = self.model.from_kwargs(**self.model_fixed_kwargs, **gs_kwargs)
        trainer = Trainer(
            callbacks=[get_early_stopping_callback(), metrics_callback],
            **self.trainer_kwargs,
            gpus=1 if self.device_ == "cuda" else 0
        )
        trainer.fit(model, self.data_module)
        return metrics_callback.val_aggregated_metric

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
