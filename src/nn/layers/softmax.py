from nn.core import layer, tensor
import numpy as np

Tensor = tensor.Tensor


class SoftMax(layer.Layer):
    def __init__(self):
        super().__init__()
        self.p = None

    def forward(self, X: Tensor) -> Tensor:
        exp = np.exp(X - np.max(X))
        self.p = exp / np.sum(exp, axis=1, keepdims=True)
        return self.p

    def backward(self, grad_output: Tensor) -> Tensor:
        return self.p * (
            grad_output - np.sum(grad_output * self.p, axis=1, keepdims=True)
        )
