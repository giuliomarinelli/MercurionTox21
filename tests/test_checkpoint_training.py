"""Two tiny synthetic epochs; no tracked datasets or output artifacts are used."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from mercurion import train
from mercurion.model import MercurionMLP
from mercurion.export_model import export_model
from mercurion.model_package import checksum, validate_package


class SelectedCheckpointTests(unittest.TestCase):
    def test_final_thresholds_use_best_checkpoint_not_last_epoch(self):
        features = torch.zeros((8, 2048))
        labels = torch.as_tensor(np.tile((np.arange(8) % 2)[:, None], (1, 12)), dtype=torch.float32)
        batches = [(features, labels)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs" / "models").mkdir(parents=True)

            class SelectFirst:
                def __init__(self, **kwargs):
                    self.path = root / "outputs" / "models" / "best_model.pt"
                    self.early_stop = False
                    self.calls = 0

                def __call__(self, loss, model):
                    self.calls += 1
                    if self.calls == 1:
                        torch.save(model.state_dict(), self.path)
                    else:
                        with torch.no_grad():
                            model.model[-1].bias.fill_(20)

            before = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(train, "load_data", return_value=(batches, batches)), \
                     patch.object(train, "EarlyStopping", SelectFirst), \
                     patch.object(train, "find_per_label_thresholds", wraps=train.find_per_label_thresholds) as calibrate, \
                     redirect_stdout(io.StringIO()):
                    train.train_model(epochs=2, device="cpu")
            finally:
                os.chdir(before)
            selected = MercurionMLP()
            selected.load_state_dict(torch.load(root / "outputs/models/best_model.pt", weights_only=True))
            selected.eval()
            with torch.no_grad():
                expected = torch.sigmoid(selected(features)).numpy()
            np.testing.assert_allclose(calibrate.call_args.args[1], expected, atol=1e-7, rtol=1e-6)
            self.assertLess(float(expected.max()), 0.999)
            exported = json.loads((root / "outputs/best_thresholds.json").read_text())
            self.assertEqual(exported["per_label_thresholds"], train.find_per_label_thresholds(labels.numpy(), expected))
            # Complete the selected-epoch -> calibrated, immutable package path.
            validation = root / 'validation'
            validation.mkdir()
            np.save(validation / 'X_val.npy', features.numpy())
            np.save(validation / 'y_val.npy', labels.numpy())
            (validation / 'X_test.npy').write_bytes(b'held-out: must not be read')
            (validation / 'y_test.npy').write_bytes(b'held-out: must not be read')
            package = root / 'package'
            export_model(root / 'outputs/models/best_model.pt', validation, package,
                         model_version='synthetic-selected-epoch', export_revision='a' * 40)
            manifest, thresholds = validate_package(package)
            self.assertEqual(manifest['files']['weights.pt'], checksum(root / 'outputs/models/best_model.pt'))
            self.assertEqual(thresholds['per_label'], exported['per_label_thresholds'])
            common = json.loads((root / 'outputs/best_threshold.json').read_text())
            self.assertEqual(thresholds['global'], common['best_threshold'])


if __name__ == "__main__":
    unittest.main()
