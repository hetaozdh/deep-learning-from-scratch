import numpy as np

from .tensor import Tensor


class Parameter:
    def __init__(self, initial_value: Tensor) -> None:
        self.data: Tensor = initial_value
        self.grad: Tensor = np.zeros_like(initial_value)

    def zero_grad(self):
        self.grad.fill(0)
