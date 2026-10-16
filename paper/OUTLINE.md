# Doctoral-paper roadmap (from this reproduction)

Target: a publishable reproduction + extension paper. This repo already supplies
Figures 1–7, Table 1 (claim verification), and the live-market study.

1. **Forensic reproduction note** — recovering pixel pair (13, 21) by constraint
   search; method generalises to underspecified experiments.
2. **Per-class rate–distortion** — k=2 error 597 (0) → 1088 (2); formalise
   per-digit k allocation vs sparse PCA; test on MNIST-784.
3. **Split-robust evaluation** — 10-fold CV, fit-on-train-only, kNN + logistic
   baselines with/without PCA; learning curves k=2..64.
4. **Nonlinear follow-up** — kernel PCA (RBF γ sweep) and UMAP/VAE on the ring
   + 5/8 overlap region; close the 6.25 → <0.5 gap quantitatively.
5. **Market regime study** — rolling-window PCA, time-varying loadings,
   ICcPCA clustering, minimum-variance backtest vs equal weight.
6. **Numerics appendix** — power+deflation vs eigh vs randomised SVD on
   ill-conditioned spectra; demonstrate the 1e-16 accumulation case.

All extension experiments reuse `src/pca_from_scratch/` + `results/benchmark.json`
as the baseline; add `experiments/2x_*.py` and register figures here.
