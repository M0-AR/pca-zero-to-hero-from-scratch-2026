"""Generate README demo GIFs (deterministic, pillow writer).

1. demo_power_iteration.gif — the best line swings onto the slanted band (2 pixels).
2. demo_rebuild.gif — a zero rebuilt from k = 0 → 61 numbers.
Both kept small (<=480px wide) per GIF-in-README best practice (<10 MB).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from pca_from_scratch.data import load_digits, two_pixel_view
from pca_from_scratch.pca import pca_eigh, reconstruct, spread_along

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "results", "figures")


def gif_power_iteration(path, fps=6):
    X, _, _ = load_digits()
    X2 = two_pixel_view(X)
    mu = X2.mean(0)
    X2c = X2 - mu
    rng = np.random.default_rng(0)
    sel = rng.choice(len(X2), 600, replace=False)
    C = (X2c.T @ X2c) / len(X2c)
    vals, vecs = np.linalg.eigh(C)
    best = vecs[:, -1]
    a0, a1 = 0.0, float(np.arctan2(best[1], best[0]))
    # shortest rotation
    while a1 - a0 > np.pi:
        a1 -= np.pi
    while a1 - a0 < -np.pi:
        a1 += np.pi
    angs = np.linspace(a0, a1, 22).tolist() + [a1] * 8

    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=90)
    fig.tight_layout()
    line_x = np.array([mu[0] - 18, mu[0] + 18])

    def frame(i):
        ax.clear()
        w = np.array([np.cos(angs[i]), np.sin(angs[i])])
        s = spread_along(X2c, w)
        ax.scatter(X2[sel, 0], X2[sel, 1], s=6, c="#7aa2f7", alpha=0.45, edgecolors="none")
        ax.plot(mu[0], mu[1], "ko", ms=6)
        L = 20
        ax.plot([mu[0] - L * w[0], mu[0] + L * w[0]], [mu[1] - L * w[1], mu[1] + L * w[1]], "r-", lw=2)
        ax.set_xlim(-1, 17)
        ax.set_ylim(-1, 17)
        ax.set_xlabel("pixel 13 darkness")
        ax.set_ylabel("pixel 21 darkness")
        ax.set_title(f"Every picture pulls…  spread = {s:.1f}  (best ≈ 65.0)")
        ax.text(0.02, 0.02, f"pull {i + 1}/{len(angs)}", transform=ax.transAxes, fontsize=8, color="#555")
        ax.set_aspect("equal", adjustable="box")

    FuncAnimation(fig, frame, frames=len(angs)).save(path, writer="pillow", fps=fps)
    plt.close(fig)
    return path


def gif_rebuild(path, fps=4):
    X, y, _ = load_digits()
    full = pca_eigh(X, 64)
    row = int(np.where(y == 0)[0][0])
    ks = [0, 1, 2, 5, 10, 20, 40, 61]
    fig, axes = plt.subplots(1, len(ks), figsize=(9.6, 1.8), dpi=90)
    fig.tight_layout()
    ims = []
    for ax, k in zip(axes, ks):
        Xh = reconstruct(full["scores"][:, :k], full["components"][:k], full["mean"]) if k else full["mean"][None, :].repeat(len(X), 0)
        im = ax.imshow(Xh[row].reshape(8, 8), cmap="gray", vmin=0, vmax=16, animated=True)
        ax.set_title(f"k={k}")
        ax.axis("off")
        ims.append(im)
    fig.suptitle("One zero rebuilt from a few numbers")
    # animate by stepping a single panel through k

    fig2, ax2 = plt.subplots(figsize=(3.2, 3.2), dpi=90)
    fig2.tight_layout()

    def frame(i):
        ax2.clear()
        k = ks[min(i // 4, len(ks) - 1)]
        Xh = reconstruct(full["scores"][:, :k], full["components"][:k], full["mean"]) if k else full["mean"][None, :].repeat(len(X), 0)
        ax2.imshow(Xh[row].reshape(8, 8), cmap="gray", vmin=0, vmax=16)
        ax2.set_title(f"zero from {k} numbers" + ("  (blur)" if k < 2 else ""))
        ax2.text(0.5, -0.06, f"frame {i + 1}/{len(ks) * 4}", transform=ax2.transAxes, fontsize=7, color="#555", ha="center")
        ax2.axis("off")

    FuncAnimation(fig2, frame, frames=len(ks) * 4).save(path, writer="pillow", fps=fps)
    plt.close("all")
    return path


def main():
    os.makedirs(FIG, exist_ok=True)
    p1 = gif_power_iteration(os.path.join(FIG, "demo_power_iteration.gif"))
    p2 = gif_rebuild(os.path.join(FIG, "demo_rebuild.gif"))
    for p in (p1, p2):
        print(p, round(os.path.getsize(p) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
