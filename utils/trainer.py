import numpy as np


class Trainer:
    def __init__(self, model, loss_layer, optimizer, logger, lr_schedule=None, config=None):
        """
        Orchestrates the training of CNN and Softmax Regression models.
        
        Parameters:
        -----------
        model : list
            List of initialized layer objects from models/CNN_layers.py
        loss_layer : object
            Initialized loss object from utils/loss.py
        optimizer : object
            Initialized optimizer from utils/optimizer.py
        logger : ExperimentLogger
            Initialized logger from utils/logging.py
        lr_schedule : object, optional
            Learning rate scheduler from utils/optimizer.py
        config : dict, optional
            Run configuration parameters.
        """
        self.model = model
        self.loss_layer = loss_layer
        self.optimizer = optimizer
        self.logger = logger
        self.lr_schedule = lr_schedule
        self.config = config if config else {}
        
        # Hyperparameters for Safety Features (Loaded from config or defaults)
        self.l2_lambda = self.config.get('l2_lambda', 0.0)
        self.clip_value = self.config.get('clip_value', 1.0)
        
        # Tracking variables for Resumability
        self.start_epoch = 0
        self.best_val_loss = float('inf')

    def calculate_accuracy(self, y_pred, y_true):
        """Calculates accuracy for one-hot encoded labels."""
        predictions = np.argmax(y_pred, axis=1)
        truths = np.argmax(y_true, axis=1)
        return np.mean(predictions == truths)

    def _compute_l2_penalty(self):
        """Calculates L2 Regularization penalty across all parameterized layers."""
        l2_penalty = 0.0
        if self.l2_lambda > 0:
            for layer in self.model:
                if hasattr(layer, 'weights'):
                    l2_penalty += np.sum(layer.weights ** 2)
        return 0.5 * self.l2_lambda * l2_penalty

    def resume_from_checkpoint(self):
        """Injects saved states from checkpoint.npz back into the model and optimizer."""
        resume_epoch, best_val, saved_config = self.logger.load_checkpoint(self.model, self.optimizer)
        self.start_epoch = resume_epoch  # Start at the next epoch
        self.best_val_loss = best_val
        self.config.update(saved_config)
        print(f"Resuming training from Epoch {self.start_epoch + 1} | Best Val Loss: {self.best_val_loss:.4f}")

    def evaluate(self, X_val, y_val, batch_size):
        """Runs the validation pass without updating gradients."""
        val_loss = 0.0
        val_acc = 0.0
        num_batches = int(np.ceil(X_val.shape[0] / batch_size))

        for i in range(0, X_val.shape[0], batch_size):
            X_batch = X_val[i : i + batch_size]
            y_batch = y_val[i : i + batch_size]

            # Forward pass only
            output = X_batch
            for layer in self.model:
                output = layer.forward(output)
            
            predictions = self.loss_layer.forward(output)
            
            # Base loss + L2 Penalty
            batch_loss = self.loss_layer.compute_loss(predictions, y_batch) + self._compute_l2_penalty()
            val_loss += batch_loss
            val_acc += self.calculate_accuracy(predictions, y_batch)

        return val_loss / num_batches, val_acc / num_batches

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, patience=5):
        # Save overarching config to experiments_records/experimentX/config.json
        self.config.update({
            'epochs': epochs,
            'batch_size': batch_size,
            'patience': patience,
            'l2_lambda': self.l2_lambda,
            'clip_value': self.clip_value
        })
        self.logger.save_config(self.config)

        no_improvement_count = 0

        for epoch in range(self.start_epoch, epochs):
            # 1. Update Learning Rate Schedule
            current_lr = self.optimizer.lr
            if self.lr_schedule:
                current_lr = self.lr_schedule.get_lr(epoch)
                self.optimizer.lr = current_lr

            # 2. Mini-batch Shuffling (Milestone Requirement)
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            train_loss = 0.0
            train_acc = 0.0
            num_batches = int(np.ceil(X_train.shape[0] / batch_size))

            # 3. Training Loop
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # --- Forward Pass ---
                output = X_batch
                for layer in self.model:
                    output = layer.forward(output)

                predictions = self.loss_layer.forward(output)
                
                # Base loss + L2 Penalty
                batch_loss = self.loss_layer.compute_loss(predictions, y_batch) + self._compute_l2_penalty()
                train_loss += batch_loss
                train_acc += self.calculate_accuracy(predictions, y_batch)

                # --- Backward Pass ---
                gradient = self.loss_layer.backward(y_batch)
                
                for layer in reversed(self.model):
                    # Milestone Requirement: Gradient Clipping
                    # Prevents exploding gradients during backprop
                    if self.clip_value > 0.0:
                        gradient = np.clip(gradient, -self.clip_value, self.clip_value)
                        
                    # L2 Regularization derivative is pushed to the optimizer or layer backward
                    gradient = layer.backward(gradient, self.optimizer)

            train_loss /= num_batches
            train_acc /= num_batches

            # 4. Validation Pass
            val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)

            # 5. CSV Logging via utils/logging.py
            print(f"Epoch {epoch+1:03d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f}")
            self.logger.log_epoch(epoch + 1, train_loss, val_loss, train_acc, val_acc, current_lr)

            # 6. Checkpoint & Early Stopping Logic (Milestone Requirement)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                no_improvement_count = 0
                # Saves parameters, optimizer state, and epoch to checkpoint.npz
                self.logger.save_checkpoint(self.model, self.optimizer, epoch + 1, val_loss)
                print(f"  -> Val loss improved! Best Checkpoint saved to {self.logger.exp_dir}")
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}. No improvement for {patience} epochs.")
                    break