from abc import ABC

from nn.core.tensor import Tensor


class LossFunction(ABC):
    @staticmethod
    def forward(y_pred: Tensor, y: Tensor) -> tuple[float, dict]:
        raise NotImplementedError()

    @staticmethod
    def backward(cache) -> Tensor:
        raise NotImplementedError()
