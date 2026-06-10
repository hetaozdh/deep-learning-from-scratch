import numpy as np

from nn.core.model import NeuralNetwork
from nn.core.optim import Optim


class RMSprop(Optim):
    def __init__(
        self, model: NeuralNetwork, lr, beta=0.9, epsilon=1e-8, weight_decay=0
    ) -> None:
        self.model = model
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.s = {}

    def step(self) -> None:
        for param in self.model.parameters():
            if param not in self.s:
                self.s[param] = np.zeros_like(param.data)
            self.s[param] = self.beta * self.s[param] + param.grad**2
            if self.weight_decay > 0:
                param.data -= self.lr * self.weight_decay * param.data
            param.data -= self.lr * param.grad / (np.sqrt(self.s[param]) + self.epsilon)

        self.model.zero_grad()
