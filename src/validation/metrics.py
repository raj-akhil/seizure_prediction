"""Window discrimination and calibration with explicit undefined values."""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             log_loss, confusion_matrix, matthews_corrcoef)


def checked_predictions(y, probabilities):
    y, p = np.asarray(y), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Nonempty one-dimensional labels and matching probabilities required')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Binary eligible labels and finite probabilities in [0, 1] required')
    return y.astype(int), p


def ratio(a, b):
    return float(a / b) if b else None


def window_metrics(y, probabilities, threshold=.5, reference_probability=None):
    y, p = checked_predictions(y, probabilities)
    if not 0 <= threshold <= 1:
        raise ValueError('Invalid fixed threshold')
    both = len(np.unique(y)) == 2
    predicted = p >= threshold
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    bs = float(brier_score_loss(y, p))
    result = {'windows':len(y), 'positive_windows':int(y.sum()), 'prevalence':float(y.mean()),
        'average_precision':float(average_precision_score(y, p)) if both else None,
        'pr_chance_baseline_prevalence':float(y.mean()),
        'roc_auc':float(roc_auc_score(y, p)) if both else None,
        'brier_score':bs, 'log_loss':float(log_loss(y, p, labels=[0, 1])),
        'threshold':float(threshold), 'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp,
        'precision':ratio(tp, tp + fp), 'recall':recall, 'specificity':specificity,
        'f1':ratio(2 * tp, 2 * tp + fp + fn),
        'balanced_accuracy':(recall + specificity) / 2 if both else None,
        'mcc':float(matthews_corrcoef(y, predicted)) if both and len(np.unique(predicted)) == 2 else None,
        'brier_reference_probability':reference_probability, 'brier_reference_score':None,
        'brier_skill_score':None}
    if reference_probability is not None:
        if not np.isfinite(reference_probability) or not 0 <= reference_probability <= 1:
            raise ValueError('Invalid training-only reference probability')
        reference = float(np.mean((y - reference_probability) ** 2))
        result.update(brier_reference_score=reference,
                      brier_skill_score=1 - bs / reference if reference else None)
    return result


def calibration_table(y, probabilities, bins=10):
    y, p = checked_predictions(y, probabilities)
    if not isinstance(bins, int) or bins < 2:
        raise ValueError('At least two calibration bins required')
    ids = np.minimum((p * bins).astype(int), bins - 1)
    return [{'bin':i, 'lower':i / bins, 'upper':(i + 1) / bins,
             'count':int((ids == i).sum()),
             'mean_probability':float(p[ids == i].mean()) if (ids == i).any() else None,
             'observed_fraction':float(y[ids == i].mean()) if (ids == i).any() else None}
            for i in range(bins)]
