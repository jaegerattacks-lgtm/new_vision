import numpy as np
from models.CNN_base import Layer

class ReLU(Layer):
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)