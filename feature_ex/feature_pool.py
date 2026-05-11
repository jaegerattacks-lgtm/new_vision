import numpy as np
from tqdm import tqdm
from minicv.features import (
    global_feature_vector,
    color_moments,
    gradient_features,
    gradient_descriptor, compute_histogram
)

class FeaturePool:
    def __init__(self, use_grayscale_gradients=True):
        """
        Orchestrates the extraction of multiple feature families.
        Maintains a strict naming scheme to satisfy traceability requirements.
        """
        self.use_grayscale_gradients = use_grayscale_gradients

    def _get_channel_prefix(self, is_rgb):
        return ['R', 'G', 'B'] if is_rgb else ['Gray']

    def extract_features(self, img_rgb, img_gray):
        """
        Extracts features and their corresponding names.
        
        Parameters:
        -----------
        img_rgb : np.ndarray
            The original color image (or grayscale if dataset is purely gray)
        img_gray : np.ndarray
            The grayscale version of the image for gradient processing.
            
        Returns:
        --------
        tuple : (np.ndarray, list)
            A 1D array of all concatenated features, and a list of their string names.
        """
        features = []
        feature_names = []
        is_rgb = img_rgb.ndim == 3 and img_rgb.shape[2] == 3

        # --- 1. Global Intensity Features ---
        f_global = global_feature_vector(img_rgb)
        features.extend(f_global)
        feature_names.extend(['global_mean', 'global_std', 'global_min', 'global_max'])

        # --- 2. Color/Intensity Moments ---
        f_moments = color_moments(img_rgb)
        features.extend(f_moments)
        
        prefixes = self._get_channel_prefix(is_rgb)
        for prefix in prefixes:
            feature_names.extend([f'{prefix}_mean', f'{prefix}_variance', f'{prefix}_skewness'])

        # --- 3. Gradient Features Family 1 & 2 ---
        # Gradients are typically computed on grayscale to capture structure over color
        f_grad_1 = gradient_features(img_gray)
        features.extend(f_grad_1)
        feature_names.extend(['grad_mean_magnitude', 'grad_max_magnitude'])

        f_grad_2 = gradient_descriptor(img_gray)
        features.extend(f_grad_2)
        feature_names.extend(['grad_mean_abs_gx', 'grad_mean_abs_gy', 'grad_mean_mag_desc', 'grad_std_mag'])

        # Optional: Add Histogram Bins (Uncomment if you want a massive feature vector)
        hist = compute_histogram(img_gray, bins=50)
        features.extend(hist)
        feature_names.extend([f'hist_bin_{i}' for i in range(50)])

        return np.array(features, dtype=np.float32), feature_names

    def process_dataset(self, X_rgb, X_gray, desc="Extracting Features"):
        """
        Processes an entire dataset (N images).
        Returns a 2D matrix (N, num_features) and the list of feature names.
        """
        dataset_features = []
        names = None
        
        for i in tqdm(range(len(X_rgb)), desc=desc):
            feats, names = self.extract_features(X_rgb[i], X_gray[i])
            dataset_features.append(feats)
            
        return np.array(dataset_features), names