import numpy as np

from nn.core.model import NeuralNetwork
from nn.core.optim import Optim


class Adam(Optim):
    def __init__(
        self, model: NeuralNetwork, lr, beta1=0.9, beta2=0.999, epsilon=1e-8
    ) -> None:
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.v = {}
        self.s = {}
        self.epsilon = epsilon
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for param in self.model.parameters():
            grad = param.grad
            if param not in self.v:
                self.v[param] = np.zeros_like(param.data)
            if param not in self.s:
                self.s[param] = np.zeros_like(param.data)
            self.v[param] = self.beta1 * self.v[param] + (1 - self.beta1) * grad
            self.s[param] = self.beta2 * self.s[param] + (1 - self.beta2) * grad**2
            v_hat = self.v[param] / (1 - self.beta1**self.t)
            s_hat = self.s[param] / (1 - self.beta2**self.t)
            param.data -= self.lr * v_hat / (np.sqrt(s_hat) + self.epsilon)
        self.model.zero_grad()


class AdamW(Adam):
    def __init__(
        self,
        model: NeuralNetwork,
        lr,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=1e-4,
    ) -> None:
        super().__init__(model, lr, beta1, beta2, epsilon)
        self.weight_decay = weight_decay

    def step(self) -> None:
        self.t += 1
        for param in self.model.parameters():
            grad = param.grad
            if param not in self.v:
                self.v[param] = np.zeros_like(param.data)
            if param not in self.s:
                self.s[param] = np.zeros_like(param.data)
            self.v[param] = self.beta1 * self.v[param] + (1 - self.beta1) * grad
            self.s[param] = self.beta2 * self.s[param] + (1 - self.beta2) * grad**2
            v_hat = self.v[param] / (1 - self.beta1**self.t)
            s_hat = self.s[param] / (1 - self.beta2**self.t)

            param.data -= self.lr * self.weight_decay * param.data

            param.data -= self.lr * v_hat / (np.sqrt(s_hat) + self.epsilon)
        self.model.zero_grad()
