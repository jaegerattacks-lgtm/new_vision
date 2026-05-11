import numpy as np
from .CNN_layers import Conv2D,MaxPooling2D,Flatten,Dense 
from .activations import ReLU



def my_cnn(input_shape, num_classes):
    # This defines the "Architecture"
    model = [
        Conv2D(in_channels=input_shape[0], out_channels=16, kernel_size=3),
        ReLU(),
        MaxPooling2D(pool_size=2),
        
        Conv2D(in_channels=16, out_channels=32, kernel_size=3),
        ReLU(),
        Flatten(),
        
        Dense(input_size=32 * 29 * 29, output_size=128), # Size depends on resizing
        ReLU(),
        Dense(input_size=128, output_size=num_classes)
    ]
    return model