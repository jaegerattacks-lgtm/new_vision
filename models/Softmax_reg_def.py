from models.CNN_layers import Dense

def build_softmax_regression(input_features, num_classes):
    """
    Constructs a Softmax Regression model compatible with the Trainer.
    
    In the context of this architecture, Softmax Regression is simply 
    a single Dense layer mapping flattened features directly to class logits.
    
    Parameters:
    -----------
    input_features : int
        The flattened size of the input image or feature vector (e.g., 64*64*3)
    num_classes : int
        The number of target classifications.
        
    Returns:
    --------
    list
        A list containing the initialized layer(s), allowing it to plug 
        directly into: `for layer in self.model:` inside trainer.py.
    """
    
    model = [
        Dense(input_size=input_features, output_size=num_classes)
    ]
    
    return model