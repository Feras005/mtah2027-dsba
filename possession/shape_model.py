"""Shape models: P(home has the ball | team shape) and P(dead | team shape), never seeing the ball.

Trained on pseudo-labels (pseudo.py) with gradient-boosted trees, **cross-half**: the model that
predicts the first half is trained only on the second half and vice versa, so every prediction
the pipeline uses is out-of-sample, and the reported accuracies are honest held-out numbers
against the pseudo-labels (not against any ground truth -- none is used).

Gradient boosting rather than a deep model: ~60 tabular features, tens of thousands of rows,
and the result has to be explainable. Permutation importances are printed for the document.
"""

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from . import config as C

SEED = 0
# sklearn 1.5 passes an option newer scipy no longer knows; harmless, and noisy.
warnings.filterwarnings("ignore", message="Unknown solver options")


def _model():
    return HistGradientBoostingClassifier(max_iter=C.SHAPE_TREES, learning_rate=0.05,
                                          max_leaf_nodes=31, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=SEED)


def _balanced_weights(y):
    w = np.ones(len(y))
    for c in np.unique(y):
        w[y == c] = len(y) / (2.0 * (y == c).sum())
    return w


def _subsample(idx, k, rng):
    """A seeded random subset, so huge homogeneous blocks do not dominate a fit."""
    if len(idx) <= k:
        return idx
    return np.sort(rng.choice(idx, size=k, replace=False))


def cross_half(X, y, blocks, names, label, log, importances=False):
    """Out-of-sample P(y = 1) for every frame of each half. y: -1 unlabelled, 0/1 labelled."""
    rng = np.random.default_rng(SEED)
    p = np.full(len(X), np.nan)
    halves = {h: np.arange(s, e + 1) for h, s, e in blocks}
    report = []
    for test_h in halves:
        train_idx = np.concatenate([halves[h] for h in halves if h != test_h])
        train_idx = train_idx[y[train_idx] >= 0]
        train_idx = _subsample(train_idx, C.SHAPE_MAX_TRAIN, rng)
        m = _model()
        m.fit(X[train_idx], y[train_idx], sample_weight=_balanced_weights(y[train_idx]))
        test_idx = halves[test_h]
        p[test_idx] = m.predict_proba(X[test_idx])[:, 1]
        lab = test_idx[y[test_idx] >= 0]
        if len(lab) and len(np.unique(y[lab])) == 2:
            acc = float(((p[lab] > 0.5) == (y[lab] == 1)).mean())
            auc = float(roc_auc_score(y[lab], p[lab]))
            # A linear baseline on the same split, to show what the trees add.
            lin = make_pipeline(SimpleImputer(), StandardScaler(),
                                LogisticRegression(max_iter=500, C=0.1))
            lin.fit(X[train_idx], y[train_idx], logisticregression__sample_weight=
                    _balanced_weights(y[train_idx]))
            lauc = float(roc_auc_score(y[lab], lin.predict_proba(X[lab])[:, 1]))
            report.append((test_h, len(train_idx), len(lab), acc, auc, lauc))
            if importances and test_h == 2:
                sub = _subsample(lab, 4000, rng)
                imp = permutation_importance(m, X[sub], y[sub], n_repeats=3,
                                             random_state=SEED, scoring="roc_auc")
                order = np.argsort(-imp.importances_mean)[:10]
                log("  {} model, top features by permutation importance (AUC drop):".format(label))
                for k in order:
                    log("    {:<28} {:.4f}".format(names[k], imp.importances_mean[k]))
    for h, ntr, nte, acc, auc, lauc in report:
        log("  {} model, test half {}: trained on {} frames, tested on {} pseudo-labelled; "
            "acc {:.3f}  AUC {:.3f}  (logistic AUC {:.3f})".format(label, h, ntr, nte, acc, auc,
                                                                   lauc))
    return p


def logit(p, cap):
    p = np.clip(np.nan_to_num(p, nan=0.5), 1e-6, 1 - 1e-6)
    return np.clip(np.log(p / (1 - p)), -cap, cap)
