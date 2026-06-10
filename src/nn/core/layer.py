from typing import Generator
from nn.core.param import Parameter
from nn.core.tensor import Tensor
from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        return X

    @abstractmethod
    def backward(self, grad_output: Tensor) -> Tensor:
        return grad_output

    def parameters(self) -> Generator[Parameter, None, None]:
        for attr in self.__dict__.values():
            if isinstance(attr, Parameter):
                yield attr

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()
