from nn.core import layer, tensor
import numpy as np

Tensor = tensor.Tensor


class Tanh(layer.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X: Tensor) -> Tensor:
        self.a = np.tanh(X)
        return self.a

    def backward(self, grad_output: Tensor) -> Tensor:
        return 1 - self.a**2 * grad_output


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)
