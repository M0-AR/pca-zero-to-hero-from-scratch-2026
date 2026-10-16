"""Build the GitHub Pages site: preview.html -> docs/index.html + vendored figures.

Paths in preview.html use ASSET_<file> placeholders; they are rewritten to
assets/figures/<file> so docs/ is fully self-contained (Pages serves docs/
only). Run:  PYTHONPATH=src python experiments/make_site.py
"""
from __future__ import annotations

import os
import shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_HTML = os.path.join(HERE, "preview.html")
DOCS = os.path.join(HERE, "docs")
ASSETS = os.path.join(DOCS, "assets", "figures")
FIG = os.path.join(HERE, "results", "figures")

NEEDED = [
    "demo_power_iteration.gif",
    "demo_rebuild.gif",
    "map2.png",
    "components.png",
    "cumvar.png",
    "scree.png",
    "ring.png",
    "market_scree.png",
]


def main():
    os.makedirs(ASSETS, exist_ok=True)
    with open(SRC_HTML) as f:
        html = f.read()
    missing = [a for a in NEEDED if f"ASSET_{a}" not in html]
    if missing:
        raise SystemExit(f"preview.html missing placeholders: {missing}")
    html = html.replace("ASSET_", "assets/figures/")
    with open(os.path.join(DOCS, "index.html"), "w") as f:
        f.write(html)
    for a in NEEDED:
        src = os.path.join(FIG, a)
        if not os.path.exists(src):
            raise SystemExit(f"figure missing: {src} (run run_all.py + make_gifs.py first)")
        shutil.copyfile(src, os.path.join(ASSETS, a))
    open(os.path.join(DOCS, ".nojekyll"), "w").close()
    total = sum(os.path.getsize(os.path.join(ASSETS, a)) for a in NEEDED)
    print(f"docs/index.html + {len(NEEDED)} figures ({total/1e6:.2f} MB total)")


if __name__ == "__main__":
    main()
