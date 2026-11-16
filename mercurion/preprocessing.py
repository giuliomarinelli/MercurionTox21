import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from mercurion.labels import tox21_labels
from mercurion.fingerprints import smiles_to_fingerprint

def preprocess_tox21(input_csv='data/raw/tox21.csv',
                     output_dir='data/processed/',
                     test_size=0.2,
                     val_size=0.1,
                     random_state=42):
    
    if (not np.isfinite(test_size) or not np.isfinite(val_size)
            or test_size <= 0 or val_size <= 0 or test_size + val_size >= 1):
        raise ValueError('Split fractions must be positive and sum to less than one')
    df = pd.read_csv(input_csv)
    if not set(['smiles', *tox21_labels]).issubset(df.columns):
        raise ValueError('Missing smiles or ordered Tox21 label columns')

    df['fingerprint'] = df['smiles'].apply(
        lambda value: smiles_to_fingerprint(value) if isinstance(value, str) and value.strip() else None)
    df = df.dropna(subset=['fingerprint'])
    if df.empty:
        raise ValueError('No usable SMILES rows')
    if not df[tox21_labels].isin([0, 1]).all().all():
        raise ValueError('Retained rows require twelve binary labels; missing values are not negatives')

    X = np.stack(df['fingerprint'].to_list())
    y = df[tox21_labels].to_numpy(dtype=np.float32)

    # Prima split train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Poi split train vs val
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = map(
        np.asarray,
        train_test_split(X_temp, y_temp, test_size=val_relative_size, random_state=random_state),
    )
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)

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
