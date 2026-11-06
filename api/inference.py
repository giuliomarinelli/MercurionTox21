import numpy as np
from mercurion.labels import tox21_labels
from mercurion.fingerprints import smiles_to_fingerprint
from mercurion.model_spec import TOP4_LABELS
import torch
import json
import math

def load_thresholds():
    # Preserve the current package inputs; coherent export belongs to T21-03.
    with open("outputs/best_threshold.json", "r") as f:
        json.load(f)["best_threshold"]
    with open("outputs/best_thresholds.json", "r") as f:
        return json.load(f)["per_label_thresholds"]

# ✔️ Label e indice: definiti una sola volta, in ordine
ALL_LABELS = tox21_labels
TOP4_INDICES = [ALL_LABELS.index(label) for label in TOP4_LABELS]


# ✔️ Preprocessing SMILES → fingerprint
def smiles_to_fp(smiles):
    fingerprint = smiles_to_fingerprint(smiles)
    if fingerprint is None:
        return None
    return fingerprint.astype(np.float32)

# ✔️ Inference function
def predict(smiles, model, device, *, thresholds=None):
    if thresholds is None:
        thresholds = load_thresholds()
    fp = smiles_to_fp(smiles)
    if fp is None:
        return {'error': 'Invalid SMILES'}

    tensor = torch.tensor(fp, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        if tuple(output.shape) != (1, len(ALL_LABELS)) or not torch.isfinite(output).all():
            raise ValueError("Invalid inference logits")
        probs = torch.sigmoid(output).cpu().numpy().flatten()
        top4_probs = [float(probs[i]) for i in TOP4_INDICES]

    # Crea output con soglia per-label
    results = {}
    for i, label in enumerate(TOP4_LABELS):
        prob = top4_probs[i]
        threshold = thresholds[label]
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid inference threshold")
        is_positive = prob > threshold
        results[label] = {
            "probability": prob,
            "is_positive": is_positive,
            "threshold": threshold 
        }
    return results
