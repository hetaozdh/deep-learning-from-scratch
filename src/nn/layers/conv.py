from typing import Callable, Literal


from nn.core.layer import Layer
from nn.core.param import Parameter
from nn.core.tensor import Tensor

import numpy as np
from numba import njit, prange

from nn.utils import initializers


@njit(parallel=True)
def convolve2d(
    x: Tensor, w: Tensor, stride: int = 1, dilation: int = 1, groups: int = 1
):
    # x: (b, in_channel, H , W)
    # w: (out_channel, in_channel_per_group, kH, kW)
    # y: (b, out_channel, fH, fW)
    b, ic, H, W = x.shape
    oc, _, kH, kW = w.shape

    in_per_group = ic // groups
    out_per_group = oc // groups

    kH_d = (kH - 1) * dilation + 1
    kW_d = (kW - 1) * dilation + 1

    fH = (H - kH_d) // stride + 1
    fW = (W - kW_d) // stride + 1
    y = np.zeros((b, oc, fH, fW), dtype=x.dtype)
    for idx in prange(b * oc):
        b_idx = idx // oc
        out_idx = idx % oc

        g_idx = out_idx // out_per_group
        base_in_idx = g_idx * in_per_group

        for i in range(fH):
            for j in range(fW):
                acc = x.dtype.type(0)

                base_i = i * stride
                base_j = j * stride

                for in_idx in range(base_in_idx, base_in_idx + in_per_group):
                    for ik in range(kH):
                        for jk in range(kW):
                            acc += (
                                w[out_idx, in_idx - base_in_idx, ik, jk]
                                * x[
                                    b_idx,
                                    in_idx,
                                    base_i + ik * dilation,
                                    base_j + jk * dilation,
                                ]
                            )
                y[b_idx, out_idx, i, j] = acc
    return y


def pad(
    X: Tensor,
    padding_mode: Literal["constant", "reflect", "symmetric", "edge"] = "constant",
    padding: int = 0,
    padding_value: int = 0,
) -> Tensor:
    match padding_mode:
        case "constant":
            return np.pad(
                X,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                "constant",
                constant_values=padding_value,
            )
        case "edge":
            return np.pad(
                X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), "edge"
            )
        case "reflect":
            return np.pad(
                X,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                "reflect",
            )
        case "symmetric":
            return np.pad(
                X,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                "symmetric",
            )


@njit(parallel=True)
def calc_w_grad(
    grad_output: Tensor,
    x: Tensor,
    w: Tensor,
    stride: int,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    # grad_output: (b, out_channel, fH, fW)
    # x: (b, in_channel, H, W)
    # w_grad: (out_channel, in_channel_per_group, kH, kW)
    dw = np.zeros_like(w)
    b, oc, fH, fW = grad_output.shape
    b, ic = x.shape[:2]
    kH, kW = dw.shape[2:]

    in_per_group = ic // groups
    out_per_group = oc // groups

    for idx in prange(b * oc):
        b_idx = idx // oc
        out_idx = idx % oc
        g_idx = out_idx // out_per_group
        base_in_idx = g_idx * in_per_group
        for i in range(fH):
            for j in range(fW):
                g = grad_output[b_idx, out_idx, i, j]
                base_i = i * stride
                base_j = j * stride
                for in_idx in range(base_in_idx, base_in_idx + in_per_group):
                    for ik in range(kH):
                        for jk in range(kW):
                            dw[out_idx, in_idx - base_in_idx, ik, jk] += (
                                g
                                * x[
                                    b_idx,
                                    in_idx,
                                    base_i + ik * dilation,
                                    base_j + jk * dilation,
                                ]
                            )

    return dw


def calc_b_grad(grad_output: Tensor) -> Tensor:
    # grad_output: (b, out_channel, fH, fW)
    return np.sum(grad_output, axis=(0, 2, 3), dtype=grad_output.dtype)


@njit(parallel=True)
def calc_X_grad(
    grad_output: Tensor,
    w: Tensor,
    X_padded: Tensor,
    stride: int,
    padding: int,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    # grad_output: (b, out_channel, fH, fW)
    # w: (out_channel, in_channel_per_group, kH, kW)
    # grad_X: (b, in_channel, H, W)
    b, oc, fH, fW = grad_output.shape
    _, ic, kH, kW = w.shape
    dX = np.zeros_like(X_padded)

    in_per_group = ic // groups
    out_per_group = oc // groups

    for idx in prange(b * oc):
        b_idx = idx // oc
        out_idx = idx % oc
        group_idx = out_idx // out_per_group
        base_i_idx = group_idx * in_per_group
        for i in range(fH):
            for j in range(fW):
                base_i_X = i * stride
                base_j_X = j * stride

                for in_idx_g in range(in_per_group):
                    in_idx = base_i_idx + in_idx_g
                    for ik in range(kH):
                        for jk in range(kW):
                            dX[
                                b_idx,
                                in_idx,
                                base_i_X + ik * dilation,
                                base_j_X + jk * dilation,
                            ] += (
                                grad_output[b_idx, out_idx, i, j]
                                * w[out_idx, in_idx_g, ik, jk]
                            )
    if padding == 0:
        return dX
    return dX[:, :, padding:-padding, padding:-padding]


class Conv2d(Layer):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_shape: tuple[int, int],
        stride: int = 1,
        padding_mode: Literal["constant", "reflect", "symmetric", "edge"] = "constant",
        padding: int = 0,
        padding_value: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        initializer: Callable = initializers.xavier_init,
    ) -> None:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding_mode: Literal["constant", "reflect", "symmetric", "edge"] = (
            padding_mode
        )
        self.padding = padding
        self.padding_value = padding_value
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        if in_channel % self.groups != 0 or out_channel % self.groups != 0:
            raise AttributeError("The number of groups cannot devide channels.")

        in_channel_per_group = in_channel // groups

        self.w = Parameter(
            initializer((out_channel, in_channel_per_group, *kernel_shape))
        )

        if self.bias:
            self.b = Parameter(np.zeros(out_channel))

    def forward(self, X: Tensor) -> Tensor:
        X_padded = pad(X, self.padding_mode, self.padding, self.padding_value)
        self.X_padded = X_padded

        return convolve2d(
            X_padded, self.w.data, self.stride, self.dilation, self.groups
        ) + (self.b.data[None, :, None, None] if self.bias else 0)

    def backward(self, grad_output: Tensor) -> Tensor:
        self.w.grad += calc_w_grad(
            grad_output,
            self.X_padded,
            self.w.data,
            self.stride,
            self.dilation,
            self.groups,
        )
        if self.bias:
            self.b.grad += calc_b_grad(grad_output)
        return calc_X_grad(
            grad_output,
            self.w.data,
            self.X_padded,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def zero_grad(self):
        self.w.grad = np.zeros_like(self.w.data)
        if self.bias:
            self.b.grad = np.zeros_like(self.b.data)
