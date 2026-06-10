from nn.core.model import NeuralNetwork
from nn.core.optim import Optim


class SGD(Optim):
    def __init__(self, model: NeuralNetwork, lr, l2_lambda=0) -> None:
        self.model = model
        self.lr = lr
        self.l2_lambda = l2_lambda

    def step(self) -> None:
        for param in self.model.parameters():
            if self.l2_lambda > 0:
                param.grad -= self.l2_lambda * param.data
            param.data -= self.lr * param.grad
        self.model.zero_grad()
