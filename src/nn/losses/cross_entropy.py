from nn.core.tensor import Tensor
from nn.core import LossFunction

import numpy as np


class CrossEntropyLoss(LossFunction):
    @staticmethod
    def forward(y_pred: Tensor, y: Tensor) -> tuple[int, dict]:
        y_pred = np.clip(y_pred, 1e-12, 1)
        cache = {"y": y, "y_pred": y_pred}
        return -np.mean(np.sum(y * np.log(y_pred), axis=1)), cache

    @staticmethod
    def backward(cache) -> Tensor:
        y = cache["y"]
        y_pred = cache["y_pred"]
        return -y / y_pred


class CrossEntropyWithLogits(LossFunction):
    """
    This loss function is used as a fusion of Cross-Entropy loss and Softmax.
    """

    @staticmethod
    def forward(y_pred: Tensor, y: Tensor) -> tuple[float, dict]:
        logits = y_pred
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)

        cache = {"logits": logits_shifted, "y": y}

        logsumexp = np.log(np.sum(np.exp(logits_shifted), axis=1, keepdims=True))
        return -np.mean(np.sum(y * (logits_shifted - logsumexp), axis=1)), cache

    @staticmethod
    def backward(cache) -> Tensor:
        logits = cache["logits"]
        y = cache["y"]

        exp = np.exp(logits)
        p = exp / np.sum(exp, axis=1, keepdims=True)
        return (p - y) / y.shape[0]
