"""Unit checks that mirror the video's exercise (spread ≈178.9 after 200 pulls)."""
import numpy as np

from pca_from_scratch.data import load_digits
from pca_from_scratch.pca import center, pca_eigh, power_iteration, spread_along, total_spread


def test_digits_shape_and_range():
    X, y, _ = load_digits()
    assert X.shape == (1797, 64)
    assert X.min() >= 0 and X.max() <= 16
    assert set(np.unique(y)) == set(range(10))


def test_power_iteration_matches_eigh_first_component():
    X, _, _ = load_digits()
    Xc, _ = center(X)
    w, _ = power_iteration(Xc, seed=123, iters=200)
    ref = pca_eigh(X, 1)
    spread_ours = spread_along(Xc, w)
    assert abs(spread_ours - float(ref["explained"][0])) / float(ref["explained"][0]) < 0.02
    # video's exercise: spread ≈ 178.9
    assert 170 < spread_ours < 190
    assert abs(float(np.dot(w, ref["components"][0]))) > 0.99


def test_total_spread_matches_eigh_sum():
    X, _, _ = load_digits()
    Xc, _ = center(X)
    ref = pca_eigh(X, 64)
    assert abs(total_spread(Xc) - float(ref["all_values"].sum())) < 1e-6
    assert 1150 < total_spread(Xc) < 1270
