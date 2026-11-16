from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from mercurion.labels import tox21_labels
from mercurion.preprocessing import preprocess_tox21


# The numerical threshold/preprocessing checks do not need Torch's native DLLs.
with patch.dict('sys.modules', {
    'torch': Mock(),
    'torch.utils.data': SimpleNamespace(DataLoader=Mock(), TensorDataset=Mock()),
    'mercurion.model': SimpleNamespace(MercurionMLP=Mock()),
    'mercurion.early_stopping': SimpleNamespace(EarlyStopping=Mock()),
    'mercurion.focal_loss': SimpleNamespace(FocalLoss=Mock()),
}):
    _training = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / 'mercurion' / 'train.py'),
        run_name='training_pipeline_test',
    )
find_best_threshold = _training['find_best_threshold']
find_per_label_thresholds = _training['find_per_label_thresholds']
train_model = _training['train_model']


class TrainingPipelineTests(unittest.TestCase):
    def test_preprocessing_preserves_array_shapes_and_labels(self):
        rows = 40
        labels = np.asarray([[i % 2] * len(tox21_labels) for i in range(rows)], dtype=np.float32)
        frame = pd.DataFrame(labels, columns=tox21_labels)
        frame['smiles'] = ['CCO', 'CCC'] * (rows // 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / 'input.csv'
            output = root / 'processed'
            frame.to_csv(csv, index=False)
            with redirect_stdout(io.StringIO()):
                preprocess_tox21(str(csv), str(output))
            for split, size in (('train', 28), ('val', 4), ('test', 8)):
                x = np.load(output / f'X_{split}.npy')
                y = np.load(output / f'y_{split}.npy')
                self.assertEqual(x.shape, (size, 2048))
                self.assertEqual(y.shape, (size, len(tox21_labels)))
                self.assertEqual(x.dtype, np.uint8)
                self.assertEqual(y.dtype, np.float32)
                self.assertTrue(np.all(y == y[:, :1]))

    def test_per_label_thresholds_use_the_correct_output_columns(self):
        targets = np.tile(np.asarray([0, 1, 0, 1])[:, None], (1, len(tox21_labels)))
        probs = np.full(targets.shape, 0.95)
        expected = {'SR-ATAD5': 0.2, 'NR-AhR': 0.4, 'SR-MMP': 0.6, 'SR-p53': 0.8}
        for label, negative in expected.items():
            index = tox21_labels.index(label)
            probs[:, index] = [negative, negative + 0.05, negative, negative + 0.05]
        thresholds = find_per_label_thresholds(targets, probs)
        self.assertEqual(set(thresholds), set(expected))
        for label, threshold in expected.items():
            self.assertAlmostEqual(thresholds[label], threshold)

    def test_threshold_search_handles_zero_division(self):
        threshold, score = find_best_threshold(np.zeros(4), np.zeros(4))
        self.assertGreaterEqual(threshold, 0.1)
        self.assertLess(threshold, 0.9)
        self.assertEqual(score, 1.0)

    def test_nonpositive_epochs_fail_before_loading_data_or_writing_outputs(self):
        load_data = Mock()
        with patch.dict(train_model.__globals__, {'load_data': load_data}), patch('builtins.open') as open_file:
            for epochs in (0, -1):
                with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                    train_model(epochs=epochs)
            load_data.assert_not_called()
            open_file.assert_not_called()


if __name__ == '__main__':
    unittest.main()
