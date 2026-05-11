import numpy as np

# ==========================================
# Optimizers
# ==========================================

class SGD:
    def __init__(self, lr=0.01, weight_decay=0.0):
        """
        Stochastic Gradient Descent (Baseline)
        """
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, param_key, param, grad):
        """
        Updates a parameter using SGD.
        param_key: Unique identifier (like id(param)) to track state if needed (not needed for plain SGD, but kept for API consistency).
        """
        # Apply L2 Regularization (Weight Decay) to the gradient
        if self.weight_decay > 0.0:
            grad = grad + self.weight_decay * param
            
        return param - self.lr * grad


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        """
        Adam Optimizer (Advanced)
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # State dictionaries to hold momentum and velocity for each parameter
        self.m = {}
        self.v = {}
        self.t = {} # Timestep per parameter

    def update(self, param_key, param, grad):
        """
        Updates a parameter using Adam.
        param_key: Unique identifier (use id(param) in the layer) to fetch correct m and v buffers.
        """
        # Apply L2 Regularization (Weight Decay)
        if self.weight_decay > 0.0:
            grad = grad + self.weight_decay * param

        # Initialize states for new parameters
        if param_key not in self.m:
            self.m[param_key] = np.zeros_like(param)
            self.v[param_key] = np.zeros_like(param)
            self.t[param_key] = 0

        self.t[param_key] += 1
        t = self.t[param_key]

        # 1. Update biased first moment estimate
        self.m[param_key] = self.beta1 * self.m[param_key] + (1 - self.beta1) * grad
        
        # 2. Update biased second raw moment estimate
        self.v[param_key] = self.beta2 * self.v[param_key] + (1 - self.beta2) * (grad ** 2)

        # 3. Compute bias-corrected first moment estimate
        m_hat = self.m[param_key] / (1 - self.beta1 ** t)
        
        # 4. Compute bias-corrected second raw moment estimate
        v_hat = self.v[param_key] / (1 - self.beta2 ** t)

        # 5. Update parameters
        param_updated = param - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return param_updated


# ==========================================
# Learning Rate Schedules
# ==========================================

class StepDecay:
    def __init__(self, initial_lr, drop_factor=0.5, epochs_drop=10):
        """Drops the learning rate by a factor every N epochs."""
        self.initial_lr = initial_lr
        self.drop_factor = drop_factor
        self.epochs_drop = epochs_drop

    def get_lr(self, epoch):
        # Epochs usually start at 0, so we use math.floor equivalent
        return self.initial_lr * (self.drop_factor ** (epoch // self.epochs_drop))


class ExponentialDecay:
    def __init__(self, initial_lr, decay_rate=0.95):
        """Gradually decays learning rate every epoch."""
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def get_lr(self, epoch):
        return self.initial_lr * (self.decay_rate ** epoch)


class CosineAnnealing:
    def __init__(self, initial_lr, min_lr=1e-6, max_epochs=50):
        """Oscillates or drops LR following a cosine curve."""
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_epochs = max_epochs

    def get_lr(self, epoch):
        if epoch >= self.max_epochs:
            return self.min_lr
            
        # Cosine formula: lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * epoch / max_epochs))
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / self.max_epochs))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay