"""Evaluation helpers: kNN, reconstruction curve, spectrum stats."""
from __future__ import annotations

import numpy as np

from .pca import mse, reconstruct


def knn_1nn_accuracy(Z_train, y_train, Z_test, y_test):
    Z_train = np.asarray(Z_train, float)
    Z_test = np.asarray(Z_test, float)
    # chunked to bound memory
    preds = np.empty(len(Z_test), dtype=int)
    for i in range(0, len(Z_test), 256):
        d = ((Z_test[i : i + 256, None, :] - Z_train[None, :, :]) ** 2).sum(-1)
        preds[i : i + 256] = y_train[d.argmin(1)]
    return float((preds == np.asarray(y_test)).mean())


def reconstruction_mses(X, mean, components, scores_full, ks):
    out = {}
    for k in ks:
        Xh = reconstruct(scores_full[:, :k], components[:k], mean)
        out[k] = mse(X, Xh)
    return out


def components_for_threshold(ratios, threshold):
    return int(np.searchsorted(np.cumsum(ratios), threshold) + 1)
