from typing import Callable

from nn.core import layer, tensor
import numpy as np

from nn.core.param import Parameter
from nn.utils import initializers

Tensor = tensor.Tensor


class Linear(layer.Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        initializer: Callable = initializers.xavier_init,
    ):
        self.w = Parameter(initializer((in_features, out_features)))
        self.b = Parameter(np.zeros(out_features))

    def forward(self, X: Tensor) -> Tensor:
        self.X = X
        return X @ self.w.data + self.b.data

    def backward(self, grad_output: Tensor) -> Tensor:
        self.w.grad += self.X.T @ grad_output
        self.b.grad += np.sum(grad_output, axis=0)
        return grad_output @ self.w.data.T

    def zero_grad(self):
        self.w.grad = np.zeros_like(self.w.grad)
        self.b.grad = np.zeros_like(self.b.grad)
