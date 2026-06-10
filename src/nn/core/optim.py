from abc import ABC, abstractmethod


class Optim(ABC):
    @abstractmethod
    def step(self):
        raise NotImplementedError()
