import deepchem as dc
import pandas as pd

def download_tox21(output_path='data/raw/tox21.csv'):
    print("Download dataset Tox21")
    tox21_tasks, datasets, _ = dc.molnet.load_tox21(featurizer='Raw', split='random')
    train_dataset, _, _ = datasets

    df = pd.DataFrame({
        'smiles': train_dataset.ids,
        'labels': list(train_dataset.y)
    })
    
    df = pd.DataFrame(train_dataset.y, columns=tox21_tasks)
    df['smiles'] = train_dataset.ids
    df.to_csv(output_path, index=False)


    df.to_csv(output_path, index=False)
    print(f"Tox21 salvato in: {output_path}")

if __name__ == "__main__":
    download_tox21()
