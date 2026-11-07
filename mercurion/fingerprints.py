"""Fingerprint calculation shared by dataset preprocessing and inference."""
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from mercurion.model_spec import FINGERPRINT_SPEC


def smiles_to_fingerprint(smiles, radius=FINGERPRINT_SPEC["radius"], n_bits=FINGERPRINT_SPEC["bits"]):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    fingerprint = GetMorganGenerator(radius=radius, fpSize=n_bits).GetFingerprint(molecule)
    return np.array([int(bit) for bit in fingerprint.ToBitString()], dtype=np.uint8)
