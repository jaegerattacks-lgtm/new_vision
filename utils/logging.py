import os
import csv
import json
import numpy as np
from datetime import datetime
from utils.plotting import plot_confusion_matrix

class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder to automatically convert NumPy types to Python types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # Converts NumPy array to Python list
        if isinstance(obj, np.integer):
            return int(obj)     # Converts np.int32/64 to int
        if isinstance(obj, np.floating):
            return float(obj)   # Converts np.float32/64 to float
        return super(NumpyEncoder, self).default(obj)

class ExperimentLogger:
    def __init__(self, base_dir="experiment_records", resume_dir=None):
        self.base_dir = base_dir
        
        if resume_dir:
            # If resuming, point to the existing experiment folder
            self.exp_dir = resume_dir
        else:
            # Create a new experiment folder incrementally
            os.makedirs(self.base_dir, exist_ok=True)
            exp_num = 1
            while os.path.exists(os.path.join(self.base_dir, f"experiment{exp_num}")):
                exp_num += 1
            self.exp_dir = os.path.join(self.base_dir, f"experiment{exp_num}")
            os.makedirs(self.exp_dir)
            
        self.log_file = os.path.join(self.exp_dir, "logs.csv")
        self.config_file = os.path.join(self.exp_dir, "config.json")
        self.checkpoint_file = os.path.join(self.exp_dir, "best_checkpoint.npz")

        # Initialize CSV header if starting fresh
        if not resume_dir:
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "learning_rate"])

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        """Appends metrics to the logs.csv file."""
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc, lr])

    def save_config(self, config_dict):
        """Saves the run configuration to a machine-readable JSON."""
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)

    def save_checkpoint(self, model, optimizer, epoch, val_loss):
        """
        Saves model weights, biases, and optimizer state to an NPZ file.
        """
        state_dict = {
            'epoch': epoch,
            'val_loss': val_loss
        }

        # Extract weights and biases from parameterized layers
        for i, layer in enumerate(model):
            if hasattr(layer, 'weights'):
                state_dict[f'layer_{i}_weights'] = layer.weights
                state_dict[f'layer_{i}_biases'] = layer.biases

        # Extract optimizer state (e.g., m and v buffers for Adam)
        if hasattr(optimizer, 'm'): state_dict['opt_m'] = optimizer.m
        if hasattr(optimizer, 'v'): state_dict['opt_v'] = optimizer.v
        if hasattr(optimizer, 't'): state_dict['opt_t'] = optimizer.t

        np.savez(self.checkpoint_file, **state_dict)

    def load_checkpoint(self, model, optimizer):
        """
        Restores model and optimizer state from the NPZ file.
        Returns the epoch to resume from and the best validation loss recorded.
        """
        if not os.path.exists(self.checkpoint_file):
            raise FileNotFoundError(f"No checkpoint found at {self.checkpoint_file}")

        state = np.load(self.checkpoint_file, allow_pickle=True)
        
        # Restore layers
        for i, layer in enumerate(model):
            if hasattr(layer, 'weights'):
                layer.weights = state[f'layer_{i}_weights']
                layer.biases = state[f'layer_{i}_biases']

        # Restore optimizer
        if hasattr(optimizer, 'm') and 'opt_m' in state: optimizer.m = state['opt_m']
        if hasattr(optimizer, 'v') and 'opt_v' in state: optimizer.v = state['opt_v']
        if hasattr(optimizer, 't') and 'opt_t' in state: optimizer.t = state['opt_t']

        # Read config to get the last known learning rate/setup
        with open(self.config_file, 'r') as f:
            config = json.load(f)

        return int(state['epoch']), float(state['val_loss']), config
    
    
def save_evaluation_report(metrics_dict, model_name, base_dir="tests_summary/evaluation_reps"):
        """Creates a dedicated folder for the test and saves the results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"eval_{timestamp}_{model_name}"
        save_path = os.path.join(base_dir, folder_name)
        
        os.makedirs(save_path, exist_ok=True)
        
        # Extract confusion matrix so we can save the rest as JSON
        cm = metrics_dict.pop("confusion_matrix")
        
        # Save text metrics
        with open(os.path.join(save_path, "metrics_report.json"), "w") as f:
            json.dump(metrics_dict, f, indent=4, cls=NumpyEncoder)
            
        # Put cm back in case the user needs it in memory
        metrics_dict["confusion_matrix"] = cm 
        
        # Here you would also call your plotting utility:
        plot_confusion_matrix(cm, save_path=os.path.join(save_path, "confusion_matrix.png"))
        
        print(f"Evaluation saved to {save_path}")