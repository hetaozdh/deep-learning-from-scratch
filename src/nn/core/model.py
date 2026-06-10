from typing import Generator

from nn.core import LossFunction
from nn.core.layer import Layer
from nn.core.param import Parameter
from nn.core.tensor import Tensor


class NeuralNetwork:
    def __init__(self, layers: list[Layer], loss_fn: LossFunction):
        self.loss_fn = loss_fn
        self.layers = layers
        self.loss = None
        self.output = None

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer.forward(x)
        self.output = x
        return x

    def backward(self, grad: Tensor):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self) -> Generator[Parameter, None, None]:
        for layer in self.layers:
            for param in layer.parameters():
                yield param

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
