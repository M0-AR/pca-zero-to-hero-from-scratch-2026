"""Dataset loaders: sklearn digits (bundled, no download) + live market data."""
from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd


def load_digits():
    from sklearn.datasets import load_digits

    d = load_digits()
    X = np.asarray(d.data, dtype=float)  # 1797 x 64, values 0..16
    y = np.asarray(d.target)
    images = np.asarray(d.images)
    return X, y, images


def two_pixel_view(X, idx=(13, 21)):
    """The video's neighbour pair, reverse-engineered and verified.

    Row-major pixels: 13 = row 1 col 5, 21 = row 2 col 5 (top-centre ink
    column of the zero). Verified fingerprint: total spread 75.0, across
    36.6, 45° 65.0, worst 10.0, kept 86.6%, mean [8.18, 7.81], zero [15, 11]
    — every number in the video matches to 1 decimal. See experiments E1.
    """
    a, b = idx
    return X[:, [a, b]]


def fetch_daily_returns_yahoo(tickers, rng_days=252, interval="1d", rng="1y"):
    """Fetch daily closes from Yahoo Finance v8 chart API (no key).

    Verified 2026-10-16: works with a browser User-Agent; Stooq CSV 404s
    from sandboxed networks, so Yahoo is the primary live source.
    Returns (prices DataFrame, log-returns DataFrame).
    """
    import json as _json

    frames = {}
    for t in tickers:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?interval={interval}&range={rng}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            payload = _json.loads(r.read().decode("utf-8"))
        res = payload["chart"]["result"][0]
        ts = res["timestamp"]
        close = res["indicators"]["quote"][0]["close"]
        idx = pd.to_datetime(ts, unit="s")
        s = pd.Series(close, index=idx, name=t).dropna()
        if len(s) < 50:
            raise RuntimeError(f"yahoo returned too little data for {t}")
        frames[t] = s
    px = pd.DataFrame(frames).dropna()
    rets = np.log(px / px.shift(1)).dropna()
    return px, rets


def fetch_daily_returns_stooq(tickers, start="2019-01-01"):
    """Legacy Stooq CSV fetcher (kept for reference; 404s on some networks).

    Prefer fetch_daily_returns_yahoo for live verification.
    """
    frames = {}
    for t in tickers:
        sym = t.lower().replace(".", "-") + ".us"
        url = f"https://stooq.com/q/d/l/?s={sym}&d1={start.replace('-','')}&d2=99999999&i=d"
        with urllib.request.urlopen(url, timeout=20) as r:
            raw = r.read().decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(raw), parse_dates=["Date"], index_col="Date")
        if "Close" not in df.columns or len(df) < 50:
            raise RuntimeError(f"stooq returned too little data for {t}")
        frames[t] = df["Close"]
    px = pd.DataFrame(frames).dropna()
    rets = np.log(px / px.shift(1)).dropna()
    return px, rets


def synthetic_market_returns(n_days=1000, n_assets=12, seed=0):
    """Deterministic fallback when offline: 1 market + 2 sector factors."""
    rng = np.random.default_rng(seed)
    mkt = rng.standard_normal(n_days) * 0.010
    s1 = rng.standard_normal(n_days) * 0.006
    s2 = rng.standard_normal(n_days) * 0.006
    R = np.zeros((n_days, n_assets))
    for j in range(n_assets):
        load = 0.8 if j % 3 == 0 else 0.5
        sec = s1 if j % 2 == 0 else s2
        R[:, j] = load * mkt + 0.4 * sec + rng.standard_normal(n_days) * 0.008
    cols = [f"ASSET{i+1:02d}" for i in range(n_assets)]
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.DataFrame(R, index=idx, columns=cols)
