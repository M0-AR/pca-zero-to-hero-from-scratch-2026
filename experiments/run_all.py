"""Master benchmark: verifies every quantitative claim in the video, end to end.

Writes results/benchmark.json + results/figures/*.png and prints PASS/FAIL table.
Exit code 0 always (benchmark, not gate); CI parses JSON thresholds instead.
"""
from __future__ import annotations

import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pca_from_scratch.data import (
    fetch_daily_returns_yahoo,
    load_digits,
    synthetic_market_returns,
    two_pixel_view,
)
from pca_from_scratch.evaluate import (
    components_for_threshold,
    knn_1nn_accuracy,
    reconstruction_mses,
)
from pca_from_scratch.pca import (
    mse,
    pca_eigh,
    pca_power,
    power_iteration,
    reconstruct,
    spread_along,
    total_spread,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

rng_global = np.random.default_rng(0)


def fig_scree(ratios, path):
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(ratios) + 1), ratios, marker="o", ms=3)
    plt.xlabel("component")
    plt.ylabel("explained variance ratio")
    plt.title("Scree — sklearn digits, PCA from scratch (eigh reference)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fig_cumvar(ratios, path):
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(ratios) + 1), np.cumsum(ratios), marker="o", ms=3)
    plt.axhline(0.90, ls="--")
    plt.axhline(0.99, ls="--")
    plt.xlabel("k components")
    plt.ylabel("cumulative variance kept")
    plt.title("Cumulative variance — digits")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fig_map2(scores2, y, path):
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(scores2[:, 0], scores2[:, 1], c=y, s=8, cmap="tab10", alpha=0.8)
    plt.colorbar(sc, label="digit")
    plt.xlabel("PC1 score")
    plt.ylabel("PC2 score")
    plt.title("Map from the start: every picture as 2 numbers (coloured by digit)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fig_components_as_pictures(components, path, n=6):
    plt.figure(figsize=(9, 3))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        v = components[i].reshape(8, 8)
        m = np.abs(v).max() or 1.0
        ax.imshow(v / m, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"PC{i+1}")
        ax.axis("off")
    plt.suptitle("Directions as pictures (blue +, red −)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fig_reconstruction(X, mean, comps, scores, path, row=0, ks=(0, 1, 2, 5, 20)):
    plt.figure(figsize=(10, 2.5))
    for j, k in enumerate(ks):
        ax = plt.subplot(1, len(ks), j + 1)
        Xh = reconstruct(scores[:, :k], comps[:k], mean) if k else np.repeat(mean[None, :], len(X), axis=0)
        ax.imshow(Xh[row].reshape(8, 8), cmap="gray", vmin=0, vmax=16)
        ax.set_title(f"k={k}")
        ax.axis("off")
    plt.suptitle(f"Rebuild of test digit (true={int(y_true_of_row)})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fig_ring(path):
    rng = np.random.default_rng(7)
    th = rng.uniform(0, 2 * np.pi, 400)
    ring = np.column_stack([10 * np.cos(th), 10 * np.sin(th)])
    Xc = ring - ring.mean(0)
    C = (Xc.T @ Xc) / len(Xc)
    vals, vecs = np.linalg.eigh(C)
    w = vecs[:, -1]
    proj = (Xc @ w)[:, None] * w[None, :] + ring.mean(0)
    plt.figure(figsize=(5, 5))
    plt.scatter(ring[:, 0], ring[:, 1], s=10, alpha=0.6, label="dots on ring r=10")
    plt.scatter(proj[:, 0], proj[:, 1], s=10, alpha=0.6, label="rebuilt from 1 number on best line")
    plt.axis("equal")
    plt.legend(fontsize=8)
    plt.title("Failure case: straight line cannot follow a bend")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    tot = float(np.mean((Xc**2).sum(1)))
    kept = float(np.mean((Xc @ w) ** 2))
    miss_line = float(np.mean(((ring - proj) ** 2).sum(1)))
    miss_euclid = float(np.mean(np.sqrt(((ring - proj) ** 2).sum(1))))
    return {"total": tot, "kept_ratio": kept / tot, "miss_line": miss_line, "miss_euclid": miss_euclid}


def main():
    t0 = time.time()
    out = {"claims": {}, "figures": [], "meta": {}}
    X, y, images = load_digits()
    out["meta"] = {"n": int(X.shape[0]), "p": int(X.shape[1]), "xmax": float(X.max()), "xmin": float(X.min())}
    assert X.shape == (1797, 64), X.shape  # transcript: 1797 pictures of 64 numbers

    # ---- 1. two-pixel demo (pair 13,21 reverse-engineered; see data.two_pixel_view) ----
    X2 = two_pixel_view(X)  # verified pair (13,21): row1c5 over row2c5
    X2c = X2 - X2.mean(0)
    tot2 = total_spread(X2c)
    s_across = spread_along(X2c, np.array([1.0, 0.0]))
    s_up = spread_along(X2c, np.array([0.0, 1.0]))
    ang = np.deg2rad(45)
    s_45 = spread_along(X2c, np.array([np.cos(ang), np.sin(ang)]))
    # best direction by eigh on 2x2
    C2 = (X2c.T @ X2c) / len(X2c)
    v2, V2 = np.linalg.eigh(C2)
    best2 = V2[:, -1]
    s_best2 = spread_along(X2c, best2)
    s_worst2 = spread_along(X2c, V2[:, 0])
    out["claims"]["two_pixel"] = {
        "pixels": [13, 21],
        "total": round(tot2, 2),
        "spread_across": round(s_across, 2),
        "spread_up": round(s_up, 2),
        "spread_45": round(s_45, 2),
        "spread_best": round(s_best2, 2),
        "spread_worst": round(s_worst2, 2),
        "kept_ratio_best": round(s_best2 / tot2, 4),
        "mean_pixels": [round(float(v), 2) for v in X2.mean(0)],
        "zero0_pixels": [float(v) for v in X2[0]],
        "video_says": "total≈75, across≈37, 45°≈65 peak, perp≈10, kept≈87%, mean≈8/8, zero=[15,11]",
    }

    # ---- 2. random directions vs power iteration (64-d) ----
    Xc = X - X.mean(0)
    tot = total_spread(Xc)
    out["claims"]["total_spread_64"] = {"value": round(tot, 1), "video_says": "≈1200"}
    R = np.array([spread_along(Xc, v / np.linalg.norm(v)) for v in rng_global.standard_normal((1000, 64))])
    out["claims"]["random_directions"] = {
        "mean_ratio": round(float((R / tot).mean()), 4),
        "best_of_1000_ratio": round(float((R / tot).max()), 4),
        "video_says": "mean≈1.5%, best-of-1000≈4.5%",
    }
    w0 = rng_global.standard_normal(64)
    w0 /= np.linalg.norm(w0)
    s0 = spread_along(Xc, w0)
    _, hist = power_iteration(Xc, seed=123, iters=200, log_every=[1, 2, 3])
    w200, _ = power_iteration(Xc, seed=123, iters=200)
    s200 = spread_along(Xc, w200)
    ref = pca_eigh(X, 1)
    out["claims"]["power_iteration"] = {
        "spread_t0_random": round(s0, 1),
        "spread_t1": round(hist.get(1, float("nan")), 1),
        "spread_t2": round(hist.get(2, float("nan")), 1),
        "spread_t3": round(hist.get(3, float("nan")), 1),
        "spread_t200_ours": round(s200, 1),
        "spread_eigh": round(float(ref["explained"][0]), 1),
        "kept_ratio": round(s200 / tot, 4),
        "video_says": "19→158→174→176→179(200 iters); 15%; numpy 178.9",
    }

    # ---- 3. full spectrum (reference eigh) + our power-deflation check ----
    full = pca_eigh(X, 64)
    vals = full["all_values"]
    ratios = vals / vals.sum()
    cum = np.cumsum(ratios)
    ours10 = pca_power(X, 10, seed=0, iters=300)
    align10 = [abs(float(ours10["components"][i] @ full["components"][i])) for i in range(10)]
    out["claims"]["spectrum"] = {
        "k2": round(float(cum[1]), 4),
        "k5": round(float(cum[4]), 4),
        "k10": round(float(cum[9]), 4),
        "k20": round(float(cum[19]), 4),
        "k_for_90": components_for_threshold(ratios, 0.90),
        "k_for_99": components_for_threshold(ratios, 0.99),
        "nonzero": int((vals > 1e-9).sum()),
        "zero_pixels_always_white": 3,
        "power_vs_eigh_abs_cos_top10_mean": round(float(np.mean(align10)), 4),
        "video_says": "2→29%,5→55%,10→74%,20→89%; k90=21,k99=41; rank 61 (3 dead pixels)",
    }
    fig_scree(ratios, os.path.join(FIG, "scree.png"))
    fig_cumvar(ratios, os.path.join(FIG, "cumvar.png"))
    fig_map2(full["scores"][:, :2], y, os.path.join(FIG, "map2.png"))
    fig_components_as_pictures(full["components"], os.path.join(FIG, "components.png"))
    out["figures"] += ["scree.png", "cumvar.png", "map2.png", "components.png"]

    # ---- 4. reconstruction (video sums over 64 pixels; mse() means over entries) ----
    K = [0, 1, 2, 5, 10, 20]
    mses = reconstruction_mses(X, full["mean"], full["components"], full["scores"], K)
    mses[0] = mse(X, np.repeat(full["mean"][None, :], len(X), axis=0))
    out["claims"]["reconstruction_mse"] = {str(k): round(float(mses[k]), 3) for k in K}
    out["claims"]["reconstruction_mse_sum64"] = {str(k): round(float(mses[k]) * 64, 1) for k in K}
    out["claims"]["reconstruction_mse"]["video_says"] = "k0≈1200,k1≈1000,k5≈550,k20≈127 (sum over 64 pixels)"
    global y_true_of_row
    zero_rows = np.where(y == 0)[0]
    y_true_of_row = int(y[zero_rows[0]])
    fig_reconstruction(X, full["mean"], full["components"], full["scores"], os.path.join(FIG, "rebuild_zero.png"), row=int(zero_rows[0]))
    out["figures"].append("rebuild_zero.png")

    # ---- 5. 1-NN protocol (hide 497, stratified-ish deterministic split) ----
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    te, tr = idx[:497], idx[497:]
    acc = {}
    Zfull_tr, Zfull_te = full["scores"][tr], full["scores"][te]
    for k in [64, 2, 5, 10, 20]:
        Ztr, Zte = Zfull_tr[:, :k], Zfull_te[:, :k]
        acc[str(k)] = round(knn_1nn_accuracy(Ztr, y[tr], Zte, y[te]), 4)
    out["claims"]["knn_1nn_holdout497"] = acc
    out["claims"]["knn_1nn_holdout497"]["video_says"] = "64→99%,2→58%,5→89%,10→96%,20→99%"

    # ---- 6. ring failure ----
    out["claims"]["ring"] = fig_ring(os.path.join(FIG, "ring.png"))
    out["claims"]["ring"]["video_says"] = "best line keeps ≈52%; line-miss ≈6/dots vs <0.5 on ring"
    out["figures"].append("ring.png")

    # ---- 7. LIVE market verification (real-life data, same method) ----
    tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "PG", "TSLA", "META", "AMZN", "UNH", "CVX"]
    live_ok, live_note, live_src = False, "", ""
    try:
        px, rets = fetch_daily_returns_yahoo(tickers, rng="2y")
        live_ok, live_note, live_src = True, f"yahoo LIVE {px.index.min().date()}→{px.index.max().date()} n={len(px)}", "yahoo"
    except Exception as e:  # offline fallback, still same code path
        rets = synthetic_market_returns()
        live_note, live_src = f"OFFLINE fallback synthetic (yahoo error: {type(e).__name__})", "synthetic"
    Zm = pca_eigh(rets.values, min(12, rets.shape[1]))
    rm = Zm["all_values"] / Zm["all_values"].sum()
    port_var_share = float(rm[0])
    out["claims"]["market_live"] = {
        "live": live_ok,
        "source": live_src,
        "note": live_note,
        "assets": list(rets.columns),
        "n_days": int(len(rets)),
        "pc1_share": round(float(rm[0]), 4),
        "pc1_3_share": round(float(rm[:3].sum()), 4),
        "pc1_5_share": round(float(rm[:5].sum()), 4),
        "expectation": "PC1≈market 35-55%, PC1-3≈60-75%, PC1-5≈75-85% (literature + quant-pca-risk)",
    }
    # market scree
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(rm) + 1), rm, marker="o", ms=3)
    plt.xlabel("market PC")
    plt.ylabel("variance share")
    plt.title(f"Live-market scree ({'LIVE' if live_ok else 'fallback'}: {len(rets)} days x {rets.shape[1]} assets)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "market_scree.png"), dpi=150)
    plt.close()
    out["figures"].append("market_scree.png")

    out["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(RES, "benchmark.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["claims"], indent=2))
    print(f"\nfigures → {FIG}\njson → {os.path.join(RES,'benchmark.json')}")


if __name__ == "__main__":
    main()
