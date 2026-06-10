import numpy as np

from nn.core.model import NeuralNetwork
from nn.core.optim import Optim


class MomentemSGD(Optim):
    def __init__(self, model: NeuralNetwork, lr, beta=0.9, weight_decay=0) -> None:
        self.model = model
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.v = {}

    def step(self) -> None:
        for param in self.model.parameters():
            grad = param.grad

            if param not in self.v:
                self.v[param] = np.zeros_like(grad)

            self.v[param] = (self.beta) * self.v[param] + grad

            if self.weight_decay > 0:
                param.data -= self.lr * self.weight_decay * param.data

            param.data -= self.lr * self.v[param]
        self.model.zero_grad()
