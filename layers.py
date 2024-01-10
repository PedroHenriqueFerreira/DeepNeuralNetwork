from math import floor, ceil

from typing import Optional, Literal

from numpy.typing import NDArray
from numpy import float64
import numpy as np


class Layer:
    ''' Base class for all layers '''

    def __init__(self):
        self.input_shape: tuple[int, ...] = ()
        self.output_shape: tuple[int, ...] = ()

    def initialize(self) -> None:
        ''' Initializes the layer '''

        return None

    def set_input_shape(self, shape: tuple[int, ...]) -> None:
        ''' Sets the input shape of the layer '''

        self.input_shape = shape

    def parameters(self) -> int:
        ''' Returns the number of trainable parameters in the layer '''

        return 0

    def forward(self, input_value: NDArray[float64], training: bool = True) -> NDArray[float64]:
        ''' Propogates input forward through the layer '''

        raise NotImplementedError()

    def backward(self, output_gradient: NDArray[float64], training: bool = True) -> NDArray[float64]:
        ''' Propogates output error backwards through the layer '''

        raise NotImplementedError()

    def get_output_shape(self) -> tuple[int, ...]:
        ''' Returns the output shape of the layer '''

        raise NotImplementedError()


class Dense(Layer):
    ''' Fully connected layer '''

    def __init__(self, units: int, input_shape: Optional[tuple[int]] = None):
        self.units = units

        if input_shape is not None:
            self.set_input_shape(input_shape)

    def initialize(self):
        limit = 1 / np.sqrt(self.input_shape[0])

        self.weights = np.random.uniform(-limit,
                                         limit, (self.input_shape[0], self.units))
        self.bias = np.zeros((1, self.units))

        self.output_shape = self.get_output_shape()

    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.bias.shape)

    def forward(self, input_value, training=True):
        self.input_value = input_value

        return input_value @ self.weights + self.bias

    def backward(self, output_gradient, training=True):
        if training:
            self.weights_gradient = self.input_value.T @ output_gradient
            self.bias_gradient = output_gradient.sum(axis=0, keepdims=True)

        return output_gradient @ self.weights.T

    def get_output_shape(self):
        return (self.units, )


class Conv2D(Layer):
    ''' 2D convolutional layer '''

    def __init__(
        self,
        filters: int,
        kernel_shape: tuple[int, int],
        stride: int = 1,
        padding: Literal['valid', 'same'] = 'same',
        input_shape: Optional[tuple[int, int, int]] = None,
    ):
        self.filters = filters
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding

        if input_shape is not None:
            self.set_input_shape(input_shape)

    def initialize(self):
        kernel_height, kernel_width = self.kernel_shape
        channels = self.input_shape[0]

        limit = 1 / np.sqrt(channels * kernel_height * kernel_width)

        self.weights = np.random.uniform(-limit, limit,
                                         (self.filters, channels, kernel_height, kernel_width))
        self.bias = np.zeros((self.filters, 1))

        self.pad_h, self.pad_w = self.determine_padding()

        self.output_shape = self.get_output_shape()

    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.bias.shape)

    def determine_padding(self):
        ''' Determine the padding based on the padding type '''

        if self.padding == 'valid':
            return (0, 0), (0, 0)

        kernel_height, kernel_width = self.kernel_shape

        vertical = floor((kernel_height - 1) /
                         2), ceil((kernel_height - 1) / 2)
        horizontal = floor((kernel_width - 1) /
                           2), ceil((kernel_width - 1) / 2)

        return vertical, horizontal

    def img_to_col_indices(self) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
        ''' Calculate the indices for the image to column operation '''

        channels, height, width = self.input_shape
        kernel_height, kernel_width = self.kernel_shape

        _, output_height, output_width = self.output_shape

        i0 = np.tile(
            np.repeat(np.arange(kernel_height), kernel_width), channels)
        i1 = self.stride * np.repeat(np.arange(output_height), output_width)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)

        j0 = np.tile(np.arange(kernel_width), kernel_height * channels)
        j1 = self.stride * np.tile(np.arange(output_width), output_height)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(channels), kernel_height *
                      kernel_width).reshape(-1, 1)

        return k, i, j

    def img_to_col(self, images: NDArray[float64]) -> NDArray[float64]:
        ''' Converts the images to columns '''

        k, i, j = self.img_to_col_indices()

        images_padded = np.pad(
            images, ((0, 0), (0, 0), self.pad_h, self.pad_w))

        cols = images_padded[:, k, i, j]

        kernel_height, kernel_width = self.kernel_shape
        channels = self.input_shape[0]

        return cols.transpose(1, 2, 0).reshape(channels * kernel_height * kernel_width, -1)

    def col_to_img(self, cols: NDArray[float64]) -> NDArray[float64]:
        ''' Converts the columns back to images '''

        channels, height, width = self.input_shape

        height_padded = height + np.sum(self.pad_h)
        width_padded = width + np.sum(self.pad_w)

        images_padded = np.zeros(
            (self.batch_size, channels, height_padded, width_padded))

        k, i, j = self.img_to_col_indices()

        kernel_height, kernel_width = self.kernel_shape
        cols = cols.reshape(channels * kernel_height *
                            kernel_width, -1, self.batch_size).transpose(2, 0, 1)

        np.add.at(images_padded, (slice(None), k, i, j), cols)  # type: ignore

        pad_h0, pad_w0 = self.pad_h[0], self.pad_w[0]

        return images_padded[:, :, pad_h0: height + pad_h0, pad_w0: width + pad_w0]

    def forward(self, input_value, training=True):
        self.batch_size = input_value.shape[0]

        self.input_col = self.img_to_col(input_value)
        self.weights_col = self.weights.reshape((self.filters, -1))

        output_value = self.weights_col @ self.input_col + self.bias

        return output_value.reshape((*self.output_shape, self.batch_size)).transpose(3, 0, 1, 2)

    def backward(self, output_gradient, training=True):
        output_gradient_col = output_gradient.transpose(
            1, 2, 3, 0).reshape(self.filters, -1)

        if training:
            self.weights_gradient = (
                output_gradient_col @ self.input_col.T).reshape(self.weights.shape)
            self.bias_gradient = np.sum(
                output_gradient_col, axis=1, keepdims=True)

        return self.col_to_img(self.weights_col.T @ output_gradient_col)

    def get_output_shape(self):
        _, input_height, input_width = self.input_shape
        kernel_height, kernel_width = self.kernel_shape

        output_height: int = (input_height - kernel_height +
                              np.sum(self.pad_h)) // self.stride + 1
        output_width: int = (input_width - kernel_width +
                             np.sum(self.pad_w)) // self.stride + 1

        return self.filters, output_height, output_width


class MaxPooling2D(Layer):
    ''' 2D max pooling layer '''

    def __init__(self,
                 pool_shape: tuple[int, int] = (2, 2),
                 stride: int = 2,
                 padding: Literal['valid', 'same'] = 'valid',
                 input_shape: Optional[tuple[int, int, int]] = None
                 ):
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding

        if input_shape is not None:
            self.set_input_shape(input_shape)

    def initialize(self):
        self.pad_h, self.pad_w = self.determine_padding()

        self.output_shape = self.get_output_shape()

    def determine_padding(self):
        ''' Determine the padding based on the padding type '''

        if self.padding == 'valid':
            return (0, 0), (0, 0)

        pool_height, pool_width = self.pool_shape

        vertical = floor((pool_height - 1) / 2), ceil((pool_height - 1) / 2)
        horizontal = floor((pool_width - 1) / 2), ceil((pool_width - 1) / 2)

        return vertical, horizontal

    def forward(self, input_value, training=True):
        batch_size = input_value.shape[0]
        channels, output_height, output_width = self.output_shape
        pool_height, pool_width = self.pool_shape

        input_value = input_value.reshape(
            batch_size,
            channels,
            output_height,
            pool_height,
            output_width,
            pool_width
        )

        output_value = input_value.max(axis=(3, 5), keepdims=True)

        self.mask = (input_value == output_value)

        return output_value.reshape(batch_size, *self.output_shape)

    def backward(self, output_gradient, training=True):
        batch_size = output_gradient.shape[0]
        channels, output_height, output_width = self.output_shape

        output_gradient = output_gradient.reshape(
            batch_size, channels, output_height, 1, output_width, 1)

        input_gradient = self.mask * output_gradient

        return input_gradient.reshape(batch_size, *self.input_shape)

    def get_output_shape(self):
        pool_height, pool_width = self.pool_shape
        channels, input_height, input_width = self.input_shape

        height = (input_height + sum(self.pad_h) -
                  pool_height) // self.stride + 1
        width = (input_width + sum(self.pad_w) - pool_width) // self.stride + 1

        return channels, height, width


class UpSampling2D(Layer):
    ''' 2D upsampling layer '''

    def __init__(self, size: tuple[int, int] = (2, 2), input_shape: Optional[tuple[int, int, int]] = None):
        self.size = size

        if input_shape is not None:
            self.set_input_shape(input_shape)

    def initialize(self):
        self.output_shape = self.get_output_shape()

    def forward(self, input_value, training=True):
        size_height, size_width = self.size

        return input_value.repeat(size_height, axis=2).repeat(size_width, axis=3)

    def backward(self, output_gradient, training=True):
        batch_size = output_gradient.shape[0]
        channels, input_height, input_width = self.input_shape

        size_height, size_width = self.size

        input_gradient = output_gradient.reshape(
            batch_size,
            channels,
            input_height,
            size_height,
            input_width,
            size_width
        )

        return input_gradient.sum(axis=(3, 5), keepdims=True).reshape(batch_size, *self.input_shape)

    def get_output_shape(self):
        channels, height, width = self.input_shape

        size_height, size_width = self.size

        return channels, height * size_height, width * size_width


class Flatten(Layer):
    ''' Flattens the input '''

    def __init__(self, input_shape: Optional[tuple[int, ...]] = None):

        if input_shape is not None:
            self.set_input_shape(input_shape)

    def initialize(self):
        self.output_shape = self.get_output_shape()

    def forward(self, input_value, training=True):
        batch_size = input_value.shape[0]

        return input_value.reshape(batch_size, -1)

    def backward(self, output_gradient, training=True):
        batch_size = output_gradient.shape[0]

        return output_gradient.reshape(batch_size, *self.input_shape)

    def get_output_shape(self):
        return np.prod(self.input_shape),


class Reshape(Layer):
    ''' Reshapes the input '''

    def __init__(self, shape: tuple[int, ...], input_shape: Optional[tuple[int, ...]] = None):
        self.shape = shape

        if input_shape is not None:
            self.set_input_shape(input_shape)

    def initialize(self):
        self.output_shape = self.get_output_shape()

    def forward(self, input_value, training=True):
        batch_size = input_value.shape[0]

        return input_value.reshape((batch_size, *self.shape))

    def backward(self, output_gradient, training=True):
        batch_size = output_gradient.shape[0]

        return output_gradient.reshape(batch_size, *self.input_shape)

    def get_output_shape(self):
        return self.shape


class Dropout(Layer):
    ''' Randomly drops a percentage of the input '''

    def __init__(self, p: float = 0.2, input_shape: Optional[tuple[int, ...]] = None):
        self.p = p

        self.scale = 1 / (1 - self.p)

        if input_shape is not None:
            self.set_input_shape(input_shape)

    def initialize(self):
        self.output_shape = self.get_output_shape()

    def forward(self, input_value, training=True):
        if training:
            self.mask = self.scale * \
                (np.random.uniform(0, 1, input_value.shape) > self.p)
        else:
            self.mask = 1

        return input_value * self.mask

    def backward(self, output_gradient, training=True):
        return output_gradient * self.mask

    def get_output_shape(self):
        return self.input_shape


class BatchNormalization(Layer):
    ''' Normalizes the input to have zero mean and unit variance '''

    def __init__(self, momentum: float = 0.9, input_shape: Optional[tuple[int, ...]] = None):
        self.momentum = momentum
        self.eps = 1e-10

        self.running_mean: Optional[NDArray[float64]] = None
        self.running_var: Optional[NDArray[float64]] = None

        if input_shape is not None:
            self.set_input_shape(input_shape)

    def initialize(self):
        self.weights = np.ones(self.input_shape)
        self.bias = np.zeros(self.input_shape)

        self.output_shape = self.get_output_shape()

    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.bias.shape)

    def forward(self, input_value, training=True):
        if self.running_mean is None or self.running_var is None:
            self.running_mean = np.mean(input_value, axis=0)
            self.running_var = np.var(input_value, axis=0)

        if training:
            mean = np.mean(input_value, axis=0)
            var = np.var(input_value, axis=0)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean  # type: ignore
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var  # type: ignore

        else:
            mean = self.running_mean
            var = self.running_var

        self.ivar = 1 / (var + self.eps)
        self.istd = np.sqrt(self.ivar)

        self.input_centered = input_value - mean
        self.input_normalized = self.input_centered * self.istd

        return self.input_normalized * self.weights + self.bias

    def backward(self, output_gradient, training=True):
        self.bias_gradient = output_gradient.sum(axis=0)

        if training:
            self.weights_gradient = np.sum(
                self.input_normalized * output_gradient, axis=0)

        batch_size = output_gradient.shape[0]

        #

        dxhat = output_gradient * self.weights

        distd = np.sum(dxhat * self.input_centered, axis=0)

        dxmu1 = dxhat * self.istd

        dstd = -1 * self.ivar * distd

        dvar = 0.5 * self.ivar * dstd

        dsq = (1 / batch_size) * dvar * np.ones_like(output_gradient)

        dxmu2 = 2 * self.input_centered * dsq

        dx1 = dxmu1 + dxmu2

        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)

        dx2 = (1 / batch_size) * dmu * np.ones_like(output_gradient)

        dx = dx1 + dx2

        dxx = (
            (output_gradient * self.weights * self.istd) + 
            (self.input_centered * (1 / batch_size) * self.ivar * -1 * self.ivar * np.sum(output_gradient * self.weights * self.input_centered, axis=0) * np.ones_like(output_gradient)
            )
        ) + dx2

        dx_ = (1 / batch_size) * self.weights * self.istd * (
            batch_size * output_gradient
            - self.bias_gradient
            - self.input_centered * self.ivar
            * np.sum(output_gradient * self.input_centered, axis=0)
        )

        return dx

    def get_output_shape(self):
        return self.input_shape


class Activation(Layer):
    ''' Activation layer '''

    def initialize(self):
        self.output_shape = self.get_output_shape()

    def __call__(self, x: NDArray[float64]) -> NDArray[float64]:
        ''' Applies the activation function '''

        raise NotImplementedError()

    def gradient(self, x: NDArray[float64]) -> NDArray[float64]:
        ''' Applies the gradient of the activation function '''

        raise NotImplementedError()

    def forward(self, input_value, training=True):
        self.input_value = input_value

        return self(input_value)

    def backward(self, output_gradient, training=True):
        return output_gradient * self.gradient(self.input_value)

    def get_output_shape(self):
        return self.input_shape
