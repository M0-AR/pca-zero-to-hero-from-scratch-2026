"""Build the GitHub Pages mirrors so the site resolves under EITHER source.

Source file: preview.html with ASSET_<file> placeholders.
Generated mirrors (mechanical path rewrites only):
  docs/index.html    ASSET_ -> assets/figures/      (entry for source /docs)
  docs/preview.html  ASSET_ -> assets/figures/      (canonical page)
  preview.html       ASSET_ -> docs/assets/figures/  (works under source / + file://)
  index.html         redirect -> preview.html + fallback links (entry for source /)
  .nojekyll + docs/.nojekyll (keep Jekyll away from static assets)

Result with source `/` (root):  / -> index.html redirect, /preview.html OK,
/docs/preview.html OK.  With source `/docs`:  / and /preview.html OK.
Run:  PYTHONPATH=src python experiments/make_site.py
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

REDIRECT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="0; url=preview.html">
<link rel="canonical" href="preview.html">
<title>PCA Zero-to-Hero — redirecting…</title>
</head>
<body>
<p>Redirecting to <a href="preview.html">preview.html</a>… mirrors:
<a href="preview.html">/preview.html</a> ·
<a href="docs/preview.html">/docs/preview.html</a> ·
<a href="docs/">/docs/</a></p>
<script>location.replace("preview.html");</script>
</body>
</html>
"""


def render(template: str, prefix: str) -> str:
    """Rewrite placeholders to a prefixed asset path (idempotent input)."""
    html = template.replace("docs/assets/figures/", "ASSET_").replace("assets/figures/", "ASSET_")
    missing = [a for a in NEEDED if f"ASSET_{a}" not in html]
    if missing:
        raise SystemExit(f"preview.html missing placeholders: {missing}")
    out = html.replace("ASSET_", f"{prefix}assets/figures/")
    if "ASSET_" in out:
        raise SystemExit("unrewritten ASSET_ placeholder left in output")
    return out


def main():
    os.makedirs(ASSETS, exist_ok=True)
    with open(SRC_HTML) as f:
        template = f.read()
    pages = {
        os.path.join(DOCS, "index.html"): render(template, ""),
        os.path.join(DOCS, "preview.html"): render(template, ""),
        os.path.join(HERE, "preview.html"): render(template, "docs/"),
    }
    for path, html in pages.items():
        with open(path, "w") as f:
            f.write(html)
    with open(os.path.join(HERE, "index.html"), "w") as f:
        f.write(REDIRECT)
    for a in NEEDED:
        src = os.path.join(FIG, a)
        if not os.path.exists(src):
            raise SystemExit(f"figure missing: {src} (run run_all.py + make_gifs.py first)")
        shutil.copyfile(src, os.path.join(ASSETS, a))
    open(os.path.join(DOCS, ".nojekyll"), "w").close()
    open(os.path.join(HERE, ".nojekyll"), "w").close()
    total = sum(os.path.getsize(os.path.join(ASSETS, a)) for a in NEEDED)
    print(f"3 pages + redirect + {len(NEEDED)} figures ({total/1e6:.2f} MB total)")


if __name__ == "__main__":
    main()
