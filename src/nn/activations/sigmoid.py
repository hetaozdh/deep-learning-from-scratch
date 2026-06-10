from nn.core import layer, tensor
import numpy as np

Tensor = tensor.Tensor


class Sigmoid(layer.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X: Tensor) -> Tensor:
        self.a = 1 / (1 + np.exp(-X))
        return self.a

    def backward(self, grad_output: Tensor) -> Tensor:
        return grad_output * self.a * (1 - self.a)
