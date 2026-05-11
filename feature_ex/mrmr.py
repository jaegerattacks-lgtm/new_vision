import pandas as pd
import numpy as np
from mrmr import mrmr_classif

class MRMRSelector:
    def __init__(self, k_features):
        """
        Minimum Redundancy Maximum Relevance Feature Selector.
        
        Parameters:
        -----------
        k_features : int
            The target number of top features to retain.
        """
        self.k_features = k_features
        self.selected_feature_names = []
        self.selected_indices = []

    def fit_transform(self, X_features, y_labels, feature_names):
        """
        Calculates MRMR scores, selects top K features, and reduces the dataset.
        
        Parameters:
        -----------
        X_features : np.ndarray
            The full 2D feature matrix (N_samples, N_features) from FeaturePool.
        y_labels : np.ndarray
            The 1D array of target class labels.
        feature_names : list
            The list of string names corresponding to the columns in X_features.
            
        Returns:
        --------
        np.ndarray
            The reduced feature matrix (N_samples, K).
        """
        # Convert to Pandas DataFrame for the mrmr library to track names
        df_X = pd.DataFrame(X_features, columns=feature_names)
        
        # If labels are one-hot encoded, convert back to 1D class indices for MRMR
        if y_labels.ndim > 1:
            y_1d = np.argmax(y_labels, axis=1)
        else:
            y_1d = y_labels

        series_y = pd.Series(y_1d)

        # Run MRMR Classification
        print(f"Running MRMR to select top {self.k_features} out of {len(feature_names)} features...")
        self.selected_feature_names = mrmr_classif(X=df_X, y=series_y, K=self.k_features)
        
        print("\nTop Features Selected:")
        for i, name in enumerate(self.selected_feature_names):
            print(f"{i+1}. {name}")

        # Map selected names back to column indices for future transforms
        self.selected_indices = [feature_names.index(name) for name in self.selected_feature_names]

        # Return the truncated numpy array
        return df_X[self.selected_feature_names].values

    def transform(self, X_features):
        """
        Applies the previously learned feature selection to new data (e.g., Validation/Test sets).
        """
        if not self.selected_indices:
            raise ValueError("Selector has not been fitted. Call fit_transform first.")
            
        return X_features[:, self.selected_indices]