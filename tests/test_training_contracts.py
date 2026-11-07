"""Synthetic correctness checks; no historical data/model or network access."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
import pandas as pd

from mercurion.calibration import find_best_threshold, find_per_label_thresholds
from mercurion.early_stopping import EarlyStopping
from mercurion.labels import tox21_labels
from mercurion.preprocessing import preprocess_tox21
from mercurion.train import load_data


class PreprocessingContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Each row's first twelve feature bits encode its ordered targets.
        self.targets = ((np.arange(40)[:, None] >> np.arange(12)) & 1).astype(np.float32)
        self.frame = pd.DataFrame(self.targets, columns=tox21_labels)
        self.frame['smiles'] = [f'row-{i}' for i in range(40)]

    @staticmethod
    def fingerprint(smiles):
        if not isinstance(smiles, str) or not smiles.startswith('row-'):
            return None
        result = np.zeros(2048, dtype=np.uint8)
        result[:12] = (int(smiles[4:]) >> np.arange(12)) & 1
        return result

    def preprocess(self, frame=None, name='out', **kwargs):
        frame = self.frame if frame is None else frame
        csv = self.root / 'input.csv'
        frame.to_csv(csv, index=False)
        output = self.root / name
        with patch('mercurion.preprocessing.smiles_to_fingerprint', side_effect=self.fingerprint), redirect_stdout(io.StringIO()):
            preprocess_tox21(csv, output, **kwargs)
        return output

    def test_row_splits_are_reproducible_disjoint_complete_and_label_aligned(self):
        # Deliberately scramble CSV columns: semantic label order must prevail.
        frame = self.frame[['smiles', *reversed(tox21_labels)]]
        first = self.preprocess(frame, name='first')
        second = self.preprocess(frame, name='second')
        third = self.preprocess(frame, name='third', random_state=7)
        memberships = []
        changed = False
        for split, size in [('train', 28), ('val', 4), ('test', 8)]:
            x, y = (np.load(first / f'{kind}_{split}.npy') for kind in ('X', 'y'))
            self.assertEqual(x.shape, (size, 2048))
            np.testing.assert_array_equal(x[:, :12], y)
            ids = set((x[:, :12] @ (2 ** np.arange(12))).tolist())
            self.assertEqual(len(ids), size)
            memberships.append(ids)
            for kind in ('X', 'y'):
                np.testing.assert_array_equal(np.load(first / f'{kind}_{split}.npy'), np.load(second / f'{kind}_{split}.npy'))
            changed |= not np.array_equal(x, np.load(third / f'X_{split}.npy'))
        self.assertEqual(set.union(*memberships), set(range(40)))
        for left, right in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(memberships[left] & memberships[right])
        self.assertTrue(changed)

    def test_invalid_smiles_are_omitted_without_losing_alignment(self):
        frame = self.frame.copy()
        frame.loc[0, 'smiles'] = 'invalid'
        frame.loc[1, 'smiles'] = None
        output = self.preprocess(frame)
        ids = set()
        for split in ('train', 'val', 'test'):
            x, y = (np.load(output / f'{kind}_{split}.npy') for kind in ('X', 'y'))
            np.testing.assert_array_equal(x[:, :12], y)
            ids.update((x[:, :12] @ (2 ** np.arange(12))).tolist())
        self.assertEqual(ids, set(range(2, 40)))

    def test_invalid_labels_or_missing_columns_write_nothing(self):
        frames = [self.frame.drop(columns=tox21_labels[0])]
        for invalid in (np.nan, np.inf, -1, 2, 'unknown'):
            frame = self.frame.astype({tox21_labels[0]: object})
            frame.loc[3, tox21_labels[0]] = invalid
            frames.append(frame)
        for index, frame in enumerate(frames):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.preprocess(frame, name=f'invalid-{index}')
            self.assertFalse((self.root / f'invalid-{index}').exists())

    def test_empty_usable_rows_write_nothing(self):
        for frame in (self.frame.iloc[:0], self.frame.assign(smiles='invalid')):
            with self.subTest(rows=len(frame)), self.assertRaises(ValueError):
                self.preprocess(frame)
            self.assertFalse((self.root / 'out').exists())

    def test_invalid_split_fractions_fail_before_writes(self):
        for test, val in [(0, .1), (.2, 0), (1, .1), (.8, .2), (-.1, .1), (np.nan, .1), (.2, np.inf)]:
            with self.subTest(test=test, val=val), self.assertRaises(ValueError):
                self.preprocess(test_size=test, val_size=val)
            self.assertFalse((self.root / 'out').exists())

    def test_missing_or_non_string_smiles_does_not_reach_rdkit(self):
        from mercurion import preprocessing
        frame = self.frame.copy()
        frame.loc[0, 'smiles'] = None
        frame.loc[1, 'smiles'] = 42
        with patch.object(pd, 'read_csv', return_value=frame), \
             patch.object(preprocessing, 'smiles_to_fingerprint', side_effect=self.fingerprint) as convert:
            with redirect_stdout(io.StringIO()):
                preprocessing.preprocess_tox21('unused', self.root / 'filtered')
        self.assertTrue(all(isinstance(call.args[0], str) for call in convert.call_args_list))


class CalibrationContractTests(unittest.TestCase):
    def test_threshold_equality_is_negative_and_first_optimum_wins(self):
        threshold, score = find_best_threshold(np.array([0, 1]), np.array([.1, .15]))
        self.assertAlmostEqual(threshold, .1)
        self.assertEqual(score, 1)

    def test_constant_labels_keep_existing_f1_and_zero_score_fallback(self):
        for targets, probabilities, expected_score, expected_threshold in [
            ([0, 0], [0., 0.], 1., .1),
            ([1, 1], [1., 1.], 1., .1),
            ([1, 1], [0., 0.], 0., .5),
        ]:
            with self.subTest(targets=targets, probabilities=probabilities):
                threshold, score = find_best_threshold(np.array(targets), np.array(probabilities))
                self.assertAlmostEqual(threshold, expected_threshold)
                self.assertEqual(score, expected_score)

    def test_invalid_calibration_arrays_are_rejected(self):
        valid = np.array([0, 1])
        for targets, probabilities in [([], []), (valid, [.1]), (valid, [.1, np.nan]),
                                       (valid, [.1, np.inf]), (valid, [-.1, .5]),
                                       (valid, [.1, 1.1]), ([0, 2], [.1, .5]),
                                       ([0, np.nan], [.1, .5]), ([[0], [1]], [.1, .5])]:
            with self.subTest(targets=targets, probabilities=probabilities), self.assertRaises(ValueError):
                find_best_threshold(np.asarray(targets), np.asarray(probabilities))

    def test_per_label_requires_all_twelve_ordered_columns(self):
        for width in (4, 11, 13):
            with self.subTest(width=width), self.assertRaises(ValueError):
                find_per_label_thresholds(np.zeros((4, width)), np.zeros((4, width)))


class EarlyStoppingContractTests(unittest.TestCase):
    def test_delta_patience_counter_and_saved_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            model = torch.nn.Linear(1, 1, bias=False)
            stopper = EarlyStopping(patience=2, delta=.125, verbose=False, path=path)
            for value, loss, expected_counter in [(1, 1., 0), (2, .875, 1), (3, .75, 0), (4, .8, 1), (5, .9, 2)]:
                with torch.no_grad():
                    model.weight.fill_(value)
                stopper(loss, model)
                self.assertEqual(stopper.counter, expected_counter)
                self.assertEqual(stopper.early_stop, value == 5)
            self.assertEqual(stopper.best_loss, .75)
            self.assertEqual(torch.load(path, weights_only=True)['weight'].item(), 3)

    def test_nonfinite_loss_preserves_good_checkpoint_and_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            stopper = EarlyStopping(verbose=False, path=path)
            model = torch.nn.Linear(1, 1)
            stopper(1., model)
            before = path.read_bytes()
            for invalid in (np.nan, np.inf, -np.inf):
                with self.subTest(loss=invalid), self.assertRaises(ValueError):
                    stopper(invalid, model)
                self.assertEqual(stopper.best_loss, 1.)
                self.assertEqual(stopper.counter, 0)
                self.assertEqual(path.read_bytes(), before)

    def test_nonfinite_first_loss_never_creates_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            stopper = EarlyStopping(verbose=False, path=path)
            with self.assertRaises(ValueError):
                stopper(-np.inf, torch.nn.Linear(1, 1))
            self.assertFalse(path.exists())
            self.assertEqual(stopper.best_loss, np.inf)

    def test_invalid_stopping_configuration(self):
        for kwargs in ({'patience': 0}, {'patience': -1}, {'patience': 1.5}, {'delta': -.1}, {'delta': np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EarlyStopping(**kwargs)


class TrainingInputContractTests(unittest.TestCase):
    def test_invalid_arrays_fail_before_data_loader_construction(self):
        x = np.zeros((4, 2048), dtype=np.uint8)
        y = np.zeros((4, 12), dtype=np.float32)
        for bad_x, bad_y in [(x[:0], y[:0]), (x[:, :10], y), (x, y[:3]),
                             (x, y[:, :4]), (x + 2, y), (x, y + np.nan), (x, y + .5)]:
            for arrays in ([bad_x, bad_y, x, y], [x, y, bad_x, bad_y]):
                with self.subTest(shape=(bad_x.shape, bad_y.shape)), \
                     patch('mercurion.train.np.load', side_effect=arrays), \
                     patch('mercurion.train.DataLoader') as loader:
                    with self.assertRaises(ValueError):
                        load_data()
                    loader.assert_not_called()

    def test_valid_loaders_use_only_training_and_validation_without_pickle(self):
        from torch.utils.data import RandomSampler, SequentialSampler
        x = np.zeros((4, 2048), dtype=np.uint8)
        y = np.tile(np.arange(12) % 2, (4, 1)).astype(np.float32)
        with patch('mercurion.train.np.load', side_effect=[x, y, x, y]) as load:
            training, validation = load_data(batch_size=2)
        self.assertEqual([call.args[0] for call in load.call_args_list],
                         [f'data/processed/{name}.npy' for name in ('X_train', 'y_train', 'X_val', 'y_val')])
        self.assertTrue(all(call.kwargs == {'allow_pickle': False} for call in load.call_args_list))
        self.assertIsInstance(training.sampler, RandomSampler)
        self.assertIsInstance(validation.sampler, SequentialSampler)
        self.assertEqual(len(training), 2)
        np.testing.assert_array_equal(validation.dataset.tensors[1].numpy(), y)
        self.assertEqual(training.dataset.tensors[0].dtype, torch.float32)


if __name__ == '__main__':
    unittest.main()
