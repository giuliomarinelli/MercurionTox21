import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0.0, path='outputs/models/best_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️ EarlyStopping: nessun miglioramento ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print("✅ Modello salvato (early stopping)!")
