import numpy as np
import torch

def calculate_pos_weights(y_path='data/processed/y.npy'):
    """
    Calcola i pesi positivi per BCEWithLogitsLoss in un setting multilabel.
    """
    y = np.load(y_path)
    
    num_samples, num_classes = y.shape
    pos_counts = np.sum(y, axis=0)  # quante volte ogni classe è 1
    neg_counts = num_samples - pos_counts

    # Evita divisioni per zero con eps
    eps = 1e-8
    pos_weight = neg_counts / (pos_counts + eps)

    return torch.tensor(pos_weight, dtype=torch.float32)
