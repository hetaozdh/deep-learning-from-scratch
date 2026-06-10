from nn.core import layer, tensor
import numpy as np

Tensor = tensor.Tensor


class ReLU(layer.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X: Tensor) -> Tensor:
        self.X = X
        return np.maximum(X, 0)

    def backward(self, grad_output: Tensor) -> Tensor:
        return grad_output * (self.X > 0).astype(float)
