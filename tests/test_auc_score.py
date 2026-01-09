import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from mercurion.model import MercurionMLP
from mercurion.labels import tox21_labels

def load_test_data():
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    return torch.tensor(X_test, dtype=torch.float32), y_test

def evaluate_auc_per_label(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))  # sigmoid

    auc_scores = {}
    for i, label in enumerate(tox21_labels):
        y_true = y_test[:, i]
        y_score = probs[:, i]

        if len(np.unique(y_true)) < 2:
            auc_scores[label] = None
        else:
            auc_scores[label] = roc_auc_score(y_true, y_score)

    return auc_scores

if __name__ == "__main__":
    model = MercurionMLP()
    model.load_state_dict(torch.load("outputs/models/best_model.pt", map_location='cpu'))

    X_test_tensor, y_test_np = load_test_data()
    auc_scores = evaluate_auc_per_label(model, X_test_tensor, y_test_np)

    print("\n📊 ROC-AUC scores per label:\n")
    for label, score in sorted(auc_scores.items(), key=lambda x: (x[1] is not None, x[1]), reverse=True):
        if score is None:
            print(f"{label:<25} : ⚠️ Non calcolabile (label costante)")
        else:
            print(f"{label:<25} : {score:.2%}")
