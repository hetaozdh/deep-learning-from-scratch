from nn.core import layer, tensor

Tensor = tensor.Tensor


class Flatten(layer.Layer):
    def forward(self, X: Tensor) -> Tensor:
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad_output: Tensor) -> Tensor:
        return grad_output.reshape(self.input_shape)
