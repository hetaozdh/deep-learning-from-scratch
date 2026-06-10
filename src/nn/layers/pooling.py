from typing import Literal

from numba import int32, njit, prange
from nn.core.layer import Layer
from nn.core.tensor import Tensor
from .conv import pad

import numpy as np


@njit(parallel=True)
def maxpool_backward(
    dout: Tensor, argmax: Tensor, x_shape: tuple, k: int, stride: int
) -> Tensor:
    b, c, yH, yW = dout.shape

    dX: Tensor = np.zeros(x_shape)

    for n in prange(b * c):
        b_idx = n // c
        ch = n % c
        for i in range(yH):
            for j in range(yW):
                h_start = i * stride
                w_start = j * stride

                idx = argmax[b_idx, ch, i, j]

                kh = idx // k
                kw = idx % k

                dX[b_idx, ch, h_start + kh, w_start + kw] += dout[b_idx, ch, i, j]

    return dX


@njit(parallel=True)
def avgpool_backward(dout, x_shape, k, stride) -> Tensor:
    b, c, yH, yW = dout.shape

    dX: Tensor = np.zeros(x_shape)

    for n in prange(b * c):
        b_idx = n // c
        ch = n % c
        for i in range(yH):
            for j in range(yW):
                h_start = i * stride
                w_start = j * stride

                for ik in range(k):
                    for jk in range(k):
                        dX[b_idx, ch, h_start + ik, w_start + jk] += dout[
                            b_idx, ch, i, j
                        ] / (k**2)

    return dX


class Pooling2d(Layer):
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        padding=0,
        mode: Literal["max", "avg"] = "max",
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mode: Literal["max", "avg"] = mode

    def forward(self, X: Tensor) -> Tensor:
        # X: (batch, channels, H, W)
        X_padded = pad(X, "constant", self.padding)
        b, c, H, W = X_padded.shape
        yH = (H - self.kernel_size) // self.stride + 1
        yW = (W - self.kernel_size) // self.stride + 1
        y = np.zeros((*X_padded.shape[:2], yH, yW))
        self.x_shape = X_padded.shape

        if self.mode == "max":
            self.arg_max = np.zeros((b, c, yH, yW), dtype=int)

        for i in range(yH):
            for j in range(yW):
                patch = X_padded[
                    :,
                    :,
                    i * self.stride : i * self.stride + self.kernel_size,
                    j * self.stride : j * self.stride + self.kernel_size,
                ]

                if self.mode == "max":
                    self.arg_max[:, :, i, j] = np.argmax(
                        patch.reshape(b, c, -1), axis=2
                    )
                    y[:, :, i, j] = np.max(patch, axis=(2, 3))
                elif self.mode == "avg":
                    y[:, :, i, j] = np.mean(patch, axis=(2, 3))
        return y

    def backward(self, grad_output: Tensor) -> Tensor:
        if self.mode == "max":
            if self.padding == 0:
                return maxpool_backward(
                    grad_output,
                    self.arg_max,
                    self.x_shape,
                    self.kernel_size,
                    self.stride,
                )
            return maxpool_backward(
                grad_output, self.arg_max, self.x_shape, self.kernel_size, self.stride
            )[:, :, self.padding : -self.padding, self.padding : -self.padding]
        else:
            if self.padding == 0:
                return avgpool_backward(
                    grad_output, self.x_shape, self.kernel_size, self.stride
                )
            return avgpool_backward(
                grad_output, self.x_shape, self.kernel_size, self.stride
            )[:, :, self.padding : -self.padding, self.padding : -self.padding]
