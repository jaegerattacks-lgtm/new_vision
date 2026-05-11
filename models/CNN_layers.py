import numpy as np
from models.CNN_base import Layer

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Renamed to 'weights' and 'biases' to match the Logger's expectations
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros((out_channels, 1))

    def forward(self, input_data):
        self.input = input_data
        batch_size, in_c, in_h, in_w = input_data.shape
        
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1
        
        self.output = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        for b in range(batch_size):
            for f in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        
                        patch = self.input[b, :, h_start:h_end, w_start:w_end]
                        self.output[b, f, i, j] = np.sum(patch * self.weights[f]) + self.biases[f]
                        
        return self.output

    def backward(self, output_gradient, optimizer):
        batch_size, out_c, out_h, out_w = output_gradient.shape
        d_weights = np.zeros_like(self.weights)
        d_biases  = np.zeros_like(self.biases)
        d_input   = np.zeros_like(self.input)

        for b in range(batch_size):
            for f in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end   = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end   = w_start + self.kernel_size

                        patch = self.input[b, :, h_start:h_end, w_start:w_end]
                        grad  = output_gradient[b, f, i, j]

                        d_weights[f] += patch * grad
                        d_input[b, :, h_start:h_end, w_start:w_end] += self.weights[f] * grad
                        d_biases[f] += grad

        self.weights = optimizer.update((id(self), 'weights'), self.weights, d_weights)
        self.biases  = optimizer.update((id(self), 'biases'),  self.biases,  d_biases)
        return d_input

class MaxPooling2D(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_data):
        self.input = input_data
        b, c, h, w = input_data.shape
        
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        
        self.output = np.zeros((b, c, out_h, out_w))
        
        for b_idx in range(b):
            for c_idx in range(c):
                for i in range(out_h):
                    for j in range(out_w):
                        h_s = i * self.stride
                        w_s = j * self.stride
                        patch = self.input[b_idx, c_idx, h_s:h_s+self.pool_size, w_s:w_s+self.pool_size]
                        self.output[b_idx, c_idx, i, j] = np.max(patch)
        return self.output

    def backward(self, output_gradient, optimizer):
        # Pooling has no learnable parameters, so optimizer is ignored here
        d_input = np.zeros_like(self.input)
        b, c, out_h, out_w = output_gradient.shape
        
        for b_idx in range(b):
            for c_idx in range(c):
                for i in range(out_h):
                    for j in range(out_w):
                        h_s = i * self.stride
                        w_s = j * self.stride
                        patch = self.input[b_idx, c_idx, h_s:h_s+self.pool_size, w_s:w_s+self.pool_size]
                        
                        max_val = np.max(patch)
                        mask = (patch == max_val)
                        d_input[b_idx, c_idx, h_s:h_s+self.pool_size, w_s:w_s+self.pool_size] += mask * output_gradient[b_idx, c_idx, i, j]
                        
        return d_input    
    
class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, optimizer):
        
        d_weights = np.dot(self.input.T, output_gradient)
        d_biases  = np.sum(output_gradient, axis=0, keepdims=True)
        d_input   = np.dot(output_gradient, self.weights.T)

        self.weights = optimizer.update((id(self), 'weights'), self.weights, d_weights)
        self.biases  = optimizer.update((id(self), 'biases'),  self.biases,  d_biases)
        return d_input

    
class Flatten(Layer):
    def forward(self, input_data):
        self.input = input_data
        self.batch_size = input_data.shape[0]
        return np.reshape(input_data, (self.batch_size, -1))

    def backward(self, output_gradient, optimizer):
        # Flatten has no parameters, ignore optimizer
        return np.reshape(output_gradient, self.input.shape)