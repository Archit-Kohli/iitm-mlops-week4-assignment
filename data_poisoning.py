import numpy as np
import pandas as pd
import random

def poison_labels(y_train: pd.Series, poison_level: float) -> pd.Series:
    """
    Applies a label-flipping attack to a "y_train" dataset.

    Args:
        y_train: The original Series of training labels.
        poison_level: The fraction of labels to poison (e.g., 0.05 for 5%).

    Returns:
        A new Series with the poisoned labels.
    """
    if poison_level == 0.0:
        return y_train.copy()

    # Get unique classes
    unique_classes = y_train.unique()
    if len(unique_classes) < 2:
        # Cannot flip labels if there's only one class
        return y_train.copy()
        
    y_poisoned = y_train.copy()
    
    # Calculate number of samples to poison
    n_samples = len(y_poisoned)
    n_to_poison = int(n_samples * poison_level)
    
    if n_to_poison == 0:
        print(f"Warning: Poison level {poison_level} is too small to poison any samples.")
        return y_poisoned

    # Get random indices to poison
    poison_indices = np.random.choice(y_poisoned.index, n_to_poison, replace=False)
    
    print(f"Poisoning {n_to_poison} of {n_samples} training labels...")

    # Flip the labels
    for idx in poison_indices:
        current_label = y_poisoned.loc[idx]
        
        # Create a list of all *other* possible classes
        possible_new_labels = [cls for cls in unique_classes if cls != current_label]
        
        # Randomly select a new (incorrect) label
        new_label = random.choice(possible_new_labels)
        
        # Apply the poison
        y_poisoned.loc[idx] = new_label
        
    return y_poisoned