# Load Torch before sklearn's native libraries to avoid Windows DLL conflicts.
import torch
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from mercurion.model import MercurionMLP
from mercurion.early_stopping import EarlyStopping
from mercurion.focal_loss import FocalLoss
from mercurion.labels import tox21_labels
import json
import random
from typing import Any, Literal, Protocol, cast


class _F1Score(Protocol):
    def __call__(
        self, y_true: Any, y_pred: Any, *,
        average: Literal['micro', 'macro'], zero_division: Literal[0],
    ) -> float: ...


# sklearn accepts numeric zero_division; its unannotated default is a string.
_f1_score = cast(_F1Score, f1_score)


def find_best_threshold(y_true, y_probs):
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0.0
    best_thresh = 0.5
    for t in thresholds:
        y_pred = (y_probs > t).astype(int)
        f1 = _f1_score(y_true, y_pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    return best_thresh, best_f1


def find_per_label_thresholds(all_targets, all_probs):
    thresholds = {}
    for label in ('SR-ATAD5', 'NR-AhR', 'SR-MMP', 'SR-p53'):
        index = tox21_labels.index(label)
        threshold, _ = find_best_threshold(all_targets[:, index], all_probs[:, index])
        thresholds[label] = threshold
    return thresholds

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def load_data(batch_size=64):
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    y_val = np.load('data/processed/y_val.npy')

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def train_model(epochs=20, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    best_thresh = 0.5
    print(f"Using device: {device}")
    train_loader, val_loader = load_data()

    model = MercurionMLP().to(device)
    criterion = FocalLoss(alpha=0.75)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    early_stopper = EarlyStopping(patience=15, verbose=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        
        model.eval()
        val_loss = 0
        all_targets = []
        all_preds = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                all_preds.append(outputs.cpu())
                all_targets.append(y_batch.cpu())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        # Concatena i batch
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        # Sigmoid per convertire logits in probabilità
        all_probs = 1 / (1 + np.exp(-all_preds))

        # Binarizza per F1
        all_bin = (all_probs > 0.5).astype(int)

        # F1 score
        f1_micro = _f1_score(all_targets, all_bin, average='micro', zero_division=0)
        f1_macro = _f1_score(all_targets, all_bin, average='macro', zero_division=0)

        # ROC-AUC
        try:
            roc_auc = roc_auc_score(all_targets, all_probs, average='macro')
        except ValueError:
            roc_auc = float('nan')  # nel caso in cui non ci siano esempi positivi

        per_label_thresholds = find_per_label_thresholds(all_targets, all_probs)
        
        print("Threshold per label:", per_label_thresholds)
        with open("outputs/best_thresholds.json", "w") as f:
            json.dump({"per_label_thresholds": per_label_thresholds}, f, indent=2)


        best_thresh, best_f1_macro = find_best_threshold(all_targets, all_probs)
        all_bin = (all_probs > best_thresh).astype(int)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1 micro: {f1_micro:.2%} | F1 macro: {f1_macro:.2%} | ROC-AUC: {roc_auc:.2%} | Best Threshold: {best_thresh:.2f}")

    
        early_stopper(val_loss, model)

    
        if early_stopper.early_stop:
            print("🛑 Stopping early")
            break

    with open('outputs/best_threshold.json', 'w') as f:
        json.dump({'best_threshold': float(best_thresh)}, f)
    print(f"📄 Best common threshold saved in outputs/best_threshold.json: {best_thresh:.3f}")

if __name__ == "__main__":
    train_model(epochs=100)
