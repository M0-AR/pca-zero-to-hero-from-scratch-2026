"""pca_from_scratch: PCA built from zero — power iteration + deflation.

Public API:
    center(X) -> (Xc, mean)
    total_spread(Xc) -> float
    spread_along(Xc, w) -> float
    power_iteration(Xc, seed=0, iters=200, log_every=None)
    pca_power(X, n_components, seed=0, iters=200)
    pca_eigh(X, n_components)  (reference, LAPACK dsyevd via numpy)
    project(Xc, components) / reconstruct(...)
"""
from __future__ import annotations

import numpy as np


def center(X: np.ndarray):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    return X - mean, mean


def total_spread(Xc: np.ndarray) -> float:
    """Mean squared distance from centre = sum of per-feature variances (biased, /n).

    The video's 'all the spread' matches E[||x - mean||^2], i.e. mean (not sum)
    of squared distances. We use 1/n consistently so that
    total_spread == sum(eigvals of (Xc^T Xc)/n) == reconstruction MSE with k=0.
    """
    Xc = np.asarray(Xc, dtype=float)
    return float(np.mean(np.sum(Xc ** 2, axis=1)))


def spread_along(Xc: np.ndarray, w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    w = w / np.linalg.norm(w)
    scores = Xc @ w
    return float(np.mean(scores ** 2))


def power_iteration(
    Xc: np.ndarray,
    seed: int = 0,
    iters: int = 200,
    log_every: tuple | list | None = None,
):
    """Transcript-faithful power iteration ('every picture pulls').

    new_w = sum_i  score_i * x_i   where score_i = x_i . w
          = Xc.T @ (Xc @ w)  ∝  C @ w
    then w /= ||w||. Returns (w, spread_history_dict).
    """
    rng = np.random.default_rng(seed)
    n_features = Xc.shape[1]
    w = rng.standard_normal(n_features)
    w = w / np.linalg.norm(w)
    history = {}
    if log_every is None:
        log_every = []
    log_set = set(log_every)
    # record initial spread as step 0 if requested
    if 0 in log_set:
        history[0] = spread_along(Xc, w)
    for t in range(1, iters + 1):
        scores = Xc @ w
        pull = Xc.T @ scores
        n = np.linalg.norm(pull)
        if n < 1e-300:
            break
        w = pull / n
        if t in log_set:
            history[t] = spread_along(Xc, w)
    history["final"] = spread_along(Xc, w)
    return w, history


def pca_power(X: np.ndarray, n_components: int, seed: int = 0, iters: int = 200):
    """Full PCA via repeated power iteration + deflation (orthogonal)."""
    X = np.asarray(X, dtype=float)
    Xc, mean = center(X)
    R = Xc.copy()
    comps, spreads, scores_all = [], [], []
    for k in range(n_components):
        w, _ = power_iteration(R, seed=seed + k, iters=iters)
        # orthogonalise against previous (kills fp-drift accumulation a bit)
        for prev in comps:
            w = w - float(w @ prev) * prev
        w = w / np.linalg.norm(w)
        s = Xc @ w  # scores always w.r.t. centred original (standard)
        # deflate residual for next direction
        R = R - np.outer(R @ w, w)
        comps.append(w)
        spreads.append(float(np.mean(s ** 2)))
        scores_all.append(s)
    components = np.stack(comps) if comps else np.zeros((0, X.shape[1]))
    return {
        "mean": mean,
        "components": components,  # (k, p), unit rows
        "explained": np.array(spreads),  # variance per comp (biased /n)
        "scores": np.column_stack(scores_all) if scores_all else np.zeros((X.shape[0], 0)),
        "Xc": Xc,
    }


def pca_eigh(X: np.ndarray, n_components: int):
    """Reference PCA via symmetric eigendecomposition (numpy eigh)."""
    X = np.asarray(X, dtype=float)
    Xc, mean = center(X)
    n = Xc.shape[0]
    C = (Xc.T @ Xc) / n
    vals, vecs = np.linalg.eigh(C)  # ascending
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    k = min(n_components, vals.shape[0])
    components = vecs[:, :k].T.copy()
    scores = Xc @ components.T
    return {
        "mean": mean,
        "components": components,
        "explained": vals[:k].copy(),
        "explained_ratio": vals[:k] / vals.sum() if vals.sum() > 0 else vals[:k],
        "all_values": vals.copy(),
        "scores": scores,
        "Xc": Xc,
    }


def reconstruct(scores: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return scores @ components + mean


def mse(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(X_true) - np.asarray(X_pred)) ** 2))
