from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import numpy as np
from mercurion.labels import tox21_labels
import torch
import json

with open("outputs/best_threshold.json", "r") as f:
    BEST_THRESHOLD = json.load(f)["best_threshold"]
    
with open("outputs/best_thresholds.json", "r") as f:
    PER_LABEL_THRESHOLDS = json.load(f)["per_label_thresholds"]

# ✔️ Label e indice: definiti una sola volta, in ordine
ALL_LABELS = tox21_labels
TOP4_LABELS = ['SR-ATAD5', 'NR-AhR', 'SR-MMP', 'SR-p53']
TOP4_INDICES = [ALL_LABELS.index(label) for label in TOP4_LABELS]


# ✔️ Preprocessing SMILES → fingerprint
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    fp = generator.GetFingerprint(mol).ToBitString()
    return np.array([int(b) for b in fp], dtype=np.float32)

# ✔️ Inference function
def predict(smiles, model, device):
    fp = smiles_to_fp(smiles)
    if fp is None:
        return {'error': 'Invalid SMILES'}

    tensor = torch.tensor(fp, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output).cpu().numpy().flatten()
        top4_probs = [float(probs[i]) for i in TOP4_INDICES]

    # Crea output con soglia per-label
    results = {}
    for i, label in enumerate(TOP4_LABELS):
        prob = top4_probs[i]
        threshold = PER_LABEL_THRESHOLDS[label]
        is_positive = prob > threshold
        results[label] = {
            "probability": prob,
            "is_positive": is_positive,
            "threshold": threshold 
        }
    return results
