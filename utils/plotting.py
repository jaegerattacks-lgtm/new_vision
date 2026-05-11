import matplotlib.pyplot as plt
import numpy as np
import os
from minicv.io import read_image # Make sure to import this directly!

def plot_augmentation_panel(data_loader, sample_list, num_examples=5, save_path="tests_summary/aug_panels"):
    """
    Provides a panel showing original (preprocessed) vs augmented images.
    Requirement: Before and after augmentation panels.
    """
    plt.figure(figsize=(15, 6))
    
    for i in range(num_examples):
        path, label = sample_list[i]
        
        # Read using the imported function, not through data_loader
        img = read_image(path) 
        original = data_loader._preprocess(img)
        augmented = data_loader._augment(original.copy())
        
        # Before (Original/Preprocessed)
        plt.subplot(2, num_examples, i + 1)
        plt.imshow(original, cmap='gray' if original.ndim == 2 else None)
        plt.title(f"Original {i+1}")
        plt.axis('off')
        
        # After (Augmented)
        plt.subplot(2, num_examples, num_examples + i + 1)
        plt.imshow(augmented, cmap='gray' if augmented.ndim == 2 else None)
        plt.title(f"Augmented {i+1}")
        plt.axis('off')
        
    plt.tight_layout()
    
    # Ensure the directory exists before trying to save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    
def plot_class_distribution(sample_list, class_names, save_path="tests_summary/class_dist.png"):
    """Requirement: Class Distribution Plot."""
    labels = [s[1] for s in sample_list]
    counts = [labels.count(i) for i in range(len(class_names))]
    
    plt.figure(figsize=(10, 5))
    plt.bar(class_names, counts, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Dataset Class Distribution')
    plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(cm, class_names=None, save_path="tests_summary/confusion_matrix.png"):
    """
    Plots a heatmap for the confusion matrix and saves it.
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
        
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Set labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='left')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.xaxis.set_label_position('top')
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()