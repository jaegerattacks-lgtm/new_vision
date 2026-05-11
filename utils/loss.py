import numpy as np
from models.CNN_base import Layer

def categorical_cross_entropy(y_pred, y_true):
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

class SoftmaxCrossEntropy(Layer):
    def forward(self, input_data):
        # Shift values for numerical stability
        exps = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    # ---> ADD THIS METHOD <---
    def compute_loss(self, y_pred, y_true):
        """Calculates the loss value for the trainer."""
        return categorical_cross_entropy(y_pred, y_true)

    def backward(self, y_true, learning_rate=None):
        # Gradient of loss with respect to inputs of softmax
        batch_size = y_true.shape[0]
        return (self.output - y_true) / batch_size