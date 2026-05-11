import os
import numpy as np
import random
from tqdm import tqdm  
from minicv.io import read_image
from minicv.transforms import (
    resize_bilinear, rotate_image, translate_image, 
    flip_horizontal, adjust_brightness
)

class DataLoader:
    def __init__(self, data_dir, target_size=(64, 64), batch_size=32):
        self.data_dir = data_dir # where raw initial data exists
        self.target_size = target_size 
        self.batch_size = batch_size 
        
        # Intel Dataset classes
        self.class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        self.label_map = {name: i for i, name in enumerate(self.class_names)}
        
    # Made public by removing the underscore!
    def get_samples_from_dir(self, subset_dir, shuffle=True, seed=42):
        """
        Scans the 'seg_train' or 'seg_test' folders and builds a list of (path, label).
        subset_dir: e.g., 'seg_train/seg_train'
        """
        samples = [] # initialize empty list
        full_path = os.path.join(self.data_dir, subset_dir) 
        for label_name in self.class_names:
            class_folder = os.path.join(full_path, label_name) 
            if not os.path.exists(class_folder):
                continue
                
            label_idx = self.label_map[label_name]
            for img_name in os.listdir(class_folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples.append((os.path.join(class_folder, img_name), label_idx))
                    
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(samples)
            
        return samples

    def _preprocess(self, img):
        img_resized = resize_bilinear(img, self.target_size)
        return img_resized.astype(np.float32)

    def _augment(self, img):
        # 1. Random Rotation
        img = rotate_image(img, random.uniform(-15, 15), interpolation="bilinear")
        # 2. Random Translation
        img = translate_image(img, random.randint(-4, 4), random.randint(-4, 4))
        # 3. Horizontal Flip
        if random.random() > 0.5: img = flip_horizontal(img)
        # 4. Brightness
        img = adjust_brightness(img, random.uniform(0.8, 1.2))
        # 5. Noise
        noise = np.random.normal(0, 0.02, img.shape)
        return np.clip(img + noise, 0.0, 1.0)

    def load_all_data(self, subset_dir, augment=False, limit=None, shuffle=True, seed=42):
        """
        Loads the entire subset into memory as NumPy arrays.
        Useful for KNN and Softmax Regression.
        """
        samples = self.get_samples_from_dir(subset_dir) 
        
        # FIXED: Shuffle must happen BEFORE the limit!
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(samples)
            
        if limit: 
            samples = samples[:limit]
            
        x_data = [] # initialize array
        y_data = [] # initialize array
        
        print(f"Loading {len(samples)} images from {subset_dir}...")
        for path, label in tqdm(samples, desc=f"Loading {subset_dir}"):
            try:
                img = read_image(path)              # read_image expects a path string
                img = self._preprocess(img)         # resize all images to 64*64
                if augment:                         # augment true or false for train or test
                    img = self._augment(img)
                x_data.append(img)                  # fill x data with numpy array of images
                y_data.append(label)                # fill y data with corresponding labels in same row
            except:
                continue # Skip broken images
                
        return np.array(x_data), np.array(y_data)   

    def get_batches(self, subset_dir, augment=False, shuffle=True, seed=42):
        """Generator for training CNNs in mini-batches."""
        samples = self.get_samples_from_dir(subset_dir)
        
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(samples)
        
        for i in range(0, len(samples), self.batch_size):
            batch_samples = samples[i : i + self.batch_size]
            x_batch, y_batch = [], [] 
            
            for path, label in batch_samples:
                img = read_image(path)
                img = self._preprocess(img)
                if augment: img = self._augment(img)
                x_batch.append(img)                
                y_batch.append(label)              
            
            yield np.array(x_batch), np.array(y_batch)