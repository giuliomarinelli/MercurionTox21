"""Local validation-set calibration; not imported by the inference service."""
import numpy as np
from sklearn.metrics import f1_score

from mercurion.model_spec import LABELS, TOP4_LABELS


def _validated_arrays(targets, probabilities):
    targets, probabilities = np.asarray(targets), np.asarray(probabilities)
    if (targets.ndim not in (1, 2) or not targets.size
            or probabilities.shape != targets.shape
            or not np.isin(targets, [0, 1]).all()
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)):
        raise ValueError('Expected aligned nonempty binary targets and finite probabilities in [0,1]')
    return targets, probabilities


def find_best_threshold(y_true, y_probs):
    y_true, y_probs = _validated_arrays(y_true, y_probs)
    best_f1 = 0.0
    best_thresh = 0.5
    for threshold in np.arange(0.1, 0.9, 0.05):
        predicted = (y_probs > threshold).astype(int)
        score = f1_score(y_true, predicted, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(threshold)
    return best_thresh, best_f1


def find_per_label_thresholds(targets, probabilities):
    targets, probabilities = _validated_arrays(targets, probabilities)
    if targets.ndim != 2 or targets.shape[1] != len(LABELS):
        raise ValueError('Per-label calibration requires twelve ordered Tox21 columns')
    return {label: find_best_threshold(targets[:, LABELS.index(label)],
                                       probabilities[:, LABELS.index(label)])[0]
            for label in TOP4_LABELS}
