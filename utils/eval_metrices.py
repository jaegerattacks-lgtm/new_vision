import numpy as np

def compute_confusion_matrix(y_true, y_pred, num_classes):
    """
    Computes the confusion matrix from scratch.
    Rows: Actual classes, Columns: Predicted classes.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def compute_metrics(y_true, y_pred, num_classes):
    """
    Computes accuracy, precision, recall, F1 per class, macro-F1, and weighted-F1.
    
    Returns:
    --------
    dict: A dictionary containing all computed metrics.
    """
    # Ensure inputs are 1D arrays of class indices
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
        
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    
    total_samples = len(y_true)
    accuracy = np.trace(cm) / total_samples if total_samples > 0 else 0.0
    
    precisions = np.zeros(num_classes)
    recalls = np.zeros(num_classes)
    f1_scores = np.zeros(num_classes)
    supports = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])
        
        supports[i] = support
        
        # Add epsilon (1e-9) to avoid division by zero
        precisions[i] = tp / (tp + fp + 1e-9)
        recalls[i] = tp / (tp + fn + 1e-9)
        
        if (precisions[i] + recalls[i]) > 0:
            f1_scores[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
            
    macro_f1 = np.mean(f1_scores)
    weighted_f1 = np.sum(f1_scores * supports) / total_samples if total_samples > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "per_class": {
            "precision": precisions,
            "recall": recalls,
            "f1": f1_scores,
            "support": supports
        },
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }