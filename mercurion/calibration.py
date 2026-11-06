"""Local validation-set calibration; not imported by the inference service."""
import numpy as np
from sklearn.metrics import f1_score

from mercurion.model_spec import LABELS, TOP4_LABELS


def find_best_threshold(y_true, y_probs):
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
    return {label: find_best_threshold(targets[:, LABELS.index(label)],
                                       probabilities[:, LABELS.index(label)])[0]
            for label in TOP4_LABELS}
