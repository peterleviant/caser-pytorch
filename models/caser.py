from typing import Optional, Dict, Any, List, NamedTuple, Optional
import torch
from torch.nn import (
    Embedding, ReLU, Sigmoid, Identity, Tanh, Conv2d, Module, Linear, ModuleList, Dropout
)
from utils.metrics import calculate_top_N_metrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
from collections import namedtuple

class TrainingConfig(NamedTuple):
    learning_rate: float = 0.001
    lr_patience: int = 5
    lr_factor: float = 0.1
    es_patience: int = 10
    es_min_delta: float = 0.0
    l2_lambda: float = 0.01

class ModelConfig(NamedTuple):
    L: int = 0
    U: int = 0
    I: int = 0
    d: int = 0
    F_h: int = 0
    F_v: int = 0
    N: int = 0
    activation_type: str = ""
    dropout_prob: float = 0.5
    items_to_predict: int = 10

class LossConfig(NamedTuple):
    normalize_loss: bool = True
    num_negative_samples: int = 3
    seed: Optional[int] = None

class CaserModel(Module):
    """
    Caser model for sequence-based recommendation.
    """
    ACTIVATION_TYPES = {
        "relu": ReLU,
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "identity": Identity
    }

    def __init__(self, config: ModelConfig):
        super(CaserModel, self).__init__()
        self._config = config
        self.user_embedding = Embedding(config.U, config.d)
        self.item_embedding = Embedding(config.I, config.d)
        self.horizontal_conv = ModuleList([
            Conv2d(in_channels=1, out_channels=config.F_h, kernel_size=(h, config.d))
            for h in range(1, config.L + 1)
        ])
        self.vertical_conv = Conv2d(in_channels=1, out_channels=config.F_v, kernel_size=(config.L, 1))
        self.item_linear_layer = Linear(config.F_v * config.d + config.F_h * config.L, config.d)
        self.item_ll_dropout = Dropout(config.dropout_prob)
        self.prediction_layer = Linear(2 * config.d, config.I)
        self.prediction_ll_dropout = Dropout(config.dropout_prob)
        self.activation = CaserModel.ACTIVATION_TYPES[config.activation_type]()
        
    def forward(self, user_ids: torch.LongTensor, item_history: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            user_ids (torch.LongTensor): User IDs, shape (B)
            item_history (torch.LongTensor): Item history, shape (B, L)
        """
        B = user_ids.shape[0]
        P_u = self.user_embedding(user_ids)  # shape (B, d)
        E_u_t = self.item_embedding(item_history).unsqueeze(1)  # shape (B, 1, L, d)
        
        horizontal_convs = [torch.max(self.activation(conv(E_u_t)), dim=2)[0].squeeze(-1) for conv in self.horizontal_conv]
        o = torch.cat(horizontal_convs, dim=1)  # shape (B, F_h * L)
        
        o_tilde = self.activation(self.vertical_conv(E_u_t)).reshape(B, -1)  # shape (B, F_v * d)
        
        oo = torch.cat([o, o_tilde], dim=1)  # shape (B, F_v * d + F_h * L)
        z = self.item_ll_dropout(self.activation(self.item_linear_layer(oo)))  # shape (B, d)
        
        user_item_features = torch.cat([z, P_u], dim=1)  # shape (B, 2d)
        y = self.prediction_ll_dropout(self.prediction_layer(user_item_features))  # shape (B, I)
        
        return y

class CaserCriterion(Module):
    """
    Criterion for the Caser model.
    """
    def __init__(self, config: LossConfig, device: torch.device):
        super(CaserCriterion, self).__init__()
        self._bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self._num_negative_samples = config.num_negative_samples
        self._normalize_loss = config.normalize_loss
        self._random_generator = torch.Generator(device)
        if config.seed is not None:
            self._random_generator.manual_seed(config.seed)

    def forward(self, y_hat: torch.Tensor, y_pos: torch.LongTensor, y_neg: torch.LongTensor) -> torch.Tensor:
        """
        Compute the loss.
        Args:
            y_hat (torch.Tensor): Predicted values, shape (B, I)
            y_pos (torch.LongTensor): Positive items, shape (B, T + 1)
            y_neg (torch.LongTensor): Negative items, shape (B, T + 1, num_negative_samples)
        """
        pos_scores = torch.gather(y_hat, dim=1, index=y_pos)  # shape (B, T + 1)
        neg_scores = torch.gather(y_hat, dim=1, index=y_neg.view(y_neg.size(0), -1)).view(y_neg.size())  # shape (B, T + 1, num_negative_samples)

        pos_loss = self._bce_loss(pos_scores, torch.ones_like(pos_scores))
        neg_loss = self._bce_loss(neg_scores, torch.zeros_like(neg_scores))
        
        if self._normalize_loss:
            neg_loss = neg_loss.mean(dim=2)
        else:
            neg_loss = neg_loss.sum(dim=2)
        
        loss = pos_loss.mean() + neg_loss.mean()
        return loss

    def get_negative_samples(self, B: int, I: int, T: int) -> torch.Tensor:
        """
        Generate negative samples.
        Args:
            B (int): Batch size
            I (int): Number of items
            T (int): Sequence length
        """
        return torch.randint(
            low=0, high=I, size=(B, T + 1, self._num_negative_samples),
            generator=self._random_generator
        )

class Caser(pl.LightningModule):
    """
    Caser model in PyTorch Lightning.
    """
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig, loss_config: LossConfig):
        super(Caser, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for easy checkpointing
        self._model = CaserModel(model_config)
        self._criterion = CaserCriterion(loss_config, self.device)
        self._l2_lambda = training_config.l2_lambda
    
    @staticmethod
    def from_kwargs(**kwargs) -> 'Caser':
        model_config = ModelConfig({k.split("model_config_")[1]:v for  k,v in kwargs.items() if "model_config_" in k})
        training_config = TrainingConfig({k.split("training_config_")[1]:v for  k,v in kwargs.items() if "training_config_" in k})
        loss_config = LossConfig({k.split("loss_config_")[1]:v for  k,v in kwargs.items() if "loss_config_" in k})
        return Caser(model_config, training_config, loss_config)

    def forward(self, user_ids: torch.LongTensor, item_history: torch.LongTensor) -> torch.Tensor:
        return self._model(user_ids, item_history)
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.get_train_metrics(batch, "train", batch_idx)
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.get_train_metrics(batch, "val", batch_idx)
        self.get_test_metrics(batch, "val", batch_idx)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = Adam(self.parameters(), lr=self.hparams.training_config.learning_rate, weight_decay=self._l2_lambda)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=self.hparams.training_config.lr_factor, patience=self.hparams.training_config.lr_patience, verbose=True),
            'monitor': 'val_loss',
            'frequency': 1,
            'strict': True
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def get_train_metrics(self, batch: Any, split: str, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        user_ids, item_history, target, _, _ = batch
        y = self(user_ids, item_history)
        y_neg = self._criterion.get_negative_samples(y.size(0), self.hparams.model_config.I, target.size(1))
        loss = self._criterion(y, target, y_neg)
        self.log(f'{split}_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def get_test_metrics(self, batch: Any, split: str, batch_idx: int, dataloader_idx: int = 0) -> None:
        (user_ids, recent_item_histories), _ , items_encountered_previously, items_yet_to_be_interacted_with = batch
        _, sorted_indices = torch.sort(self(user_ids, recent_item_histories), dim=1, descending=True)
        metrics = calculate_top_N_metrics(sorted_indices, items_yet_to_be_interacted_with, items_encountered_previously, self.hparams.model_config.items_to_predict)

        for metric_name, metric_value in metrics.items():
            self.log(f'{split}_{metric_name}', metric_value, on_epoch=True, prog_bar=True, logger=True)
    
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """
        The test set contains each user once, and their history up to that point which is used to predict next k items
        which are compared to the ground truth consisting of the final items the user interacted with.
        each value in the batch is a tuple of (user_id, recent_item_history, items_encountered_previously, items_yet_to_be_interacted_with)
        where the last 2 are boolean masks of size I indicating which items the user has interacted with and which are yet to be interacted with.
        """
        self.get_test_metrics(batch, "test", batch_idx, dataloader_idx)
        return 0