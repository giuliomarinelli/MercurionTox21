"""The existing inference contract, shared with local model export (no training imports)."""
from mercurion.labels import tox21_labels

MODEL_SPEC = {"input_dim": 2048, "hidden_dims": [512, 128], "output_dim": 12, "dropout": 0.3}
FINGERPRINT_SPEC = {"algorithm": "rdkit-morgan", "radius": 2, "bits": 2048}
TOP4_LABELS = ["SR-ATAD5", "NR-AhR", "SR-MMP", "SR-p53"]
LABELS = list(tox21_labels)
