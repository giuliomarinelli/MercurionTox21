import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from mercurion.model import MercurionMLP
from mercurion.labels import tox21_labels  # Lista delle 12 label
import os

def load_test_data():
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    return torch.tensor(X_test, dtype=torch.float32), y_test

def plot_roc_curves(model, X_test, y_test, save_dir='outputs/roc_curves'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        logits = model(X_test).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))  # sigmoid

    for i, label in enumerate(tox21_labels):
        y_true = y_test[:, i]
        y_score = probs[:, i]

        # Skip se label tutta 0 o tutta 1
        if len(np.unique(y_true)) < 2:
            print(f"⚠️ Skipping {label}: not enough label variety.")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {label}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/roc_{label}.png')
        plt.close()
        print(f"✅ Salvata ROC per {label}")

if __name__ == "__main__":
    model = MercurionMLP()
    model.load_state_dict(torch.load("outputs/models/best_model.pt", map_location='cpu'))

    X_test_tensor, y_test_np = load_test_data()
    plot_roc_curves(model, X_test_tensor, y_test_np)
