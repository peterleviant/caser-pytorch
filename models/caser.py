import torch
from torch.nn import Embedding, ReLU, Sigmoid, Identity, Tanh, Conv2d, Module, Linear, ModuleList, Dropout
from icecream import ic
from typing import Optional

class Caser(Module):
    """
    Caser model
    """

    ACTIVATION_TYPES = {
        "relu": ReLU,
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "identity": Identity
    }

    def __init__(self, L: int, U: int, I: int, d: int, F_h: int, F_v: int, N: int, activation_type: str, dropout_prob:float=0.5):
        """
        Caser model
        Args:
            L (int): sequence length
            I (int): number of items
            U (int): number of users
            N (int): number of items to be predicted during inference
            d (int): embedding dimension
            F_h (int): number of filters for each kernel size in the horizontal convolution
            F_v (int): number of filters in the vertical convolution
            activation_type (str): activation function type
        """
        super().__init__()
        self.d: int = d
        self.L: int = L
        self.N: int = N
        self.user_embedding: Embedding = Embedding(U, d)
        self.item_embedding: Embedding = Embedding(I, d)
        self.horizontal_conv: ModuleList[Conv2d] = ModuleList([Conv2d(in_channels=1, out_channels=F_h, kernel_size=(h, d)) for h in range(1, L + 1)])
        self.vertical_conv: Conv2d = Conv2d(in_channels=1, out_channels=F_v, kernel_size=(L, 1))
        self.item_linear_layer: Linear = Linear(F_v * d + F_h * L, d)
        self.item_ll_dropout = Dropout(dropout_prob)
        self.prediction_layer: Linear = Linear(2 * d, I)
        self.prediction_ll_dropout = Dropout(dropout_prob)
        self.activation: Module = Caser.ACTIVATION_TYPES[activation_type]()
    
    def forward(self, user_ids: torch.LongTensor, item_history: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            user_ids (LongTensor): user ids - shape (B) (batch size)
            item_history (LongTensor): item_history - shape (B, L)
        """
        B = user_ids.shape[0]
        P_u = self.user_embedding(user_ids) # shape (B, d)
        E_u_t = self.item_embedding(item_history).unsqueeze(1) # shape (B, 1, L, d)
        o = torch.cat([
            torch.max(self.activation(conv(E_u_t)),dim=2)[0].squeeze(-1) for conv in self.horizontal_conv
        ], dim = 1) # shape (B, F_h * L)
        o_tilde = self.activation(self.vertical_conv(E_u_t)) # shape (B, F_v, 1, d)
        o_tilde = o_tilde.reshape(B, -1) # shape (B, F_v * d)
        oo = torch.cat([o, o_tilde], dim=1) # shape (B, F_v * d + F_h * L)
        z = self.item_ll_dropout(self.activation(self.item_linear_layer(oo))) # shape (B, d) - convolutional sequence embedding
        user_item_features = torch.cat([z, P_u], dim=1) # shape (B, 2d)
        y = self.prediction_layer(user_item_features) # shape (B, I) - prediction
        y = self.prediction_ll_dropout(y)
        return y

    def predict(self, user_ids: torch.LongTensor, item_history: torch.LongTensor) -> torch.LongTensor:
        """
        Predict top N items for each user
        Args:
            user_ids (LongTensor): user ids - shape (B) (batch size)
            item_history (LongTensor): item_history - shape (B, L)
        """
        B = user_ids.shape[0]
        with torch.no_grad():
            y = self.forward(user_ids, item_history) # shape (B, I)
            return torch.topk(y, self.N, dim=1, largest=True, sorted=True).indices # shape (B, N)

class CaserCriterion(Module):
    """
    Caser loss/criterion
    Binary cross-entropy loss for item prediction calculated on the next T+1 items.
    S_t^u,...,S_{t+T}^u are the items to be predicted, with negative sampling
    """
    def __init__(self, num_negative_samples:int, normalize: bool = True, seed: Optional[int] = None, device:torch.device=torch.device("cpu")):
        """
        Args:
            normalize (bool): whether to normalize the negative sampling in the loss function
        """
        super().__init__()
        self._bce_loss: torch.nn.Module = torch.nn.BCEWithLogitsLoss(reduction='none')
        self._normalize: bool = normalize
        self._num_negative_samples = num_negative_samples
        self._random_generator = torch.Generator(device=device)
        self._device = device
        self._initial_seed: Optional[int] = seed
        if seed is not None:
            self.generator.manual_seed(seed)

    def _set_device(self, device:torch.device):
        self._random_generator = torch.Generator(device)
        if self._initial_seed is not None:
            self._random_generator.manual_seed(self._initial_seed)
        self._device = device

    def _get_negative_samples(self, B: int, I: int, T: int):
        negative_samples = torch.randint(low=0,high=I,size=(B, T+1, self._num_negative_samples), \
        device=self._device, generator=self._random_generator)
        return negative_samples

    def to(self, device: torch.device):
        """
        Override the default .to() method to include the random generator.
        """
        self._set_device(device)
        super().to(device)
        return self

    def forward(self, y_hat: torch.Tensor, y_pos: torch.LongTensor) -> torch.nn.Module:
        """
        Forward pass
        Args:
            y_hat (Tensor): predicted values - shape (B, I)
            y_pos (LongTensor): positive items - shape (B, T + 1)
            y_neg (LongTensor): negative items - shape (B, T + 1, neg)
        """
        B, I = y_hat.shape
        y_neg = self._get_negative_samples(B, I, y_pos.shape[1])
        pos_scores = torch.gather(y_hat, dim=1, index=y_pos) # shape (B, T + 1)
        neg_scores = torch.gather(y_hat, dim=1, index=y_neg.reshape(B, -1)).reshape(y_neg.shape) # shape (B, T + 1, neg)

        pos_loss = self._bce_loss(pos_scores, torch.ones_like(pos_scores))
        neg_loss = self._bce_loss(neg_scores, torch.zeros_like(neg_scores))
        if self._normalize:
            neg_loss = neg_loss.mean(dim=2)
        else:
            neg_loss = neg_loss.sum(dim=2)
        loss = pos_loss.mean() + neg_loss.mean()
        return loss
