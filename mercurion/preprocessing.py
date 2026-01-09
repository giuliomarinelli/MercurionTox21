import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import os
from sklearn.model_selection import train_test_split
from mercurion.labels import tox21_labels

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    bitstring = fp.ToBitString()
    return np.array([int(bit) for bit in bitstring], dtype=np.uint8)

def preprocess_tox21(input_csv='data/raw/tox21.csv',
                     output_dir='data/processed/',
                     test_size=0.2,
                     val_size=0.1,
                     random_state=42):
    
    df = pd.read_csv(input_csv)

    df['fingerprint'] = df['smiles'].apply(smiles_to_fingerprint)
    df = df[df['fingerprint'].notnull()]

    X = np.stack(df['fingerprint'].values)
    y = df[tox21_labels].values.astype(np.float32)

    # Prima split train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Poi split train vs val
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_relative_size, random_state=random_state)

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    print("✅ Preprocessing completato:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Val:   {X_val.shape}, {y_val.shape}")
    print(f"  Test:  {X_test.shape}, {y_test.shape}")

if __name__ == "__main__":
    preprocess_tox21()
