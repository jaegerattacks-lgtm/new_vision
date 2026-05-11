import numpy as np

class KNNClassifier:
    def __init__(self):
        """
        K-Nearest Neighbors Classifier from scratch.
        Note: KNN doesn't actually 'train' weights. It just stores the dataset.
        """
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Stores the training features and their one-hot encoded labels."""
        self.X_train = X_train
        # Convert one-hot encoded labels to 1D class indices for easier voting
        if y_train.ndim > 1:
            self.y_train = np.argmax(y_train, axis=1)
        else:
            self.y_train = y_train

    def _compute_l2_distance(self, x_test_row):
        """
        Computes the Euclidean (L2) distance between a single test sample 
        and ALL training samples from scratch using NumPy broadcasting.
        """
        # Distance formula: sqrt(sum((x2 - x1)^2))
        squared_diff = (self.X_train - x_test_row) ** 2
        distances = np.sqrt(np.sum(squared_diff, axis=1))
        return distances

    def predict(self, X_test, k):
        """Predicts the class labels for a set of test data."""
        predictions = []
        
        for i in range(X_test.shape[0]):
            # 1. Calculate distances from this test point to all training points
            distances = self._compute_l2_distance(X_test[i])
            
            # 2. Get the indices of the k smallest distances
            k_nearest_indices = np.argsort(distances)[:k]
            
            # 3. Get the labels of those k nearest neighbors
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            # 4. Majority voting (find the most frequent label)
            # np.bincount counts occurrences, argmax finds the highest count
            most_common_label = np.argmax(np.bincount(k_nearest_labels))
            predictions.append(most_common_label)
            
        return np.array(predictions)

    def k_sweep(self, X_val, y_val, k_list):
        """
        Sweeps through a list of K values and reports the best one based on validation accuracy.
        
        Parameters:
        -----------
        X_val : np.ndarray
            Validation feature matrix.
        y_val : np.ndarray
            Validation labels.
        k_list : list
            List of integers to test (e.g., [1, 3, 5, 7, 9]).
            
        Returns:
        --------
        int
            The best K value.
        """
        if y_val.ndim > 1:
            y_val_1d = np.argmax(y_val, axis=1)
        else:
            y_val_1d = y_val

        best_k = k_list[0]
        best_acc = -1.0
        
        print("Starting K-Sweep on Validation Set...")
        for k in k_list:
            predictions = self.predict(X_val, k)
            accuracy = np.mean(predictions == y_val_1d)
            print(f"  -> K={k} | Validation Accuracy: {accuracy:.4f}")
            
            if accuracy > best_acc:
                best_acc = accuracy
                best_k = k
                
        print(f"\nOptimal K found: {best_k} with {best_acc:.4f} accuracy.")
        return best_k