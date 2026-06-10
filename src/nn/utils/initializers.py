import numpy as np

from nn.core.tensor import Tensor


def calc_in_features(shape: tuple[int, ...]) -> int:
    if len(shape) < 1:
        raise AttributeError("The tuple is empty.")
    if len(shape) == 2:
        return shape[1]
    if len(shape) >= 3:
        return int(np.prod(shape[1:]))

    raise AttributeError("The tuple has more than 2 dimensions.")


def calc_out_features(shape: tuple[int, ...]) -> int:
    if len(shape) < 1:
        raise AttributeError("The tuple is empty.")
    if len(shape) == 2:
        return shape[0]
    if len(shape) >= 3:
        return int(shape[0] * np.prod(shape[2:]))

    raise AttributeError("The tuple has more than 2 dimensions.")


def xavier_init(shape: tuple) -> Tensor:
    return np.random.randn(*shape) * np.sqrt(
        2 / (calc_in_features(shape) + calc_out_features(shape))
    )


def he_init(shape: tuple) -> Tensor:
    return np.random.randn(*shape) * np.sqrt(2 / calc_in_features(shape))
