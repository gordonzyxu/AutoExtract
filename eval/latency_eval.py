"""Small, reproducible extraction-only timing probe (rendering excluded).

Run once against the saved original backend and once against the working tree:
    python eval/latency_eval.py --backend eval/out/baseline/backend
    python eval/latency_eval.py --backend backend
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from synth import BoxSpec, ChartSpec, LineSpec, ViolinSpec, render, render_box, render_line, render_violin


def cases():
    rng = np.random.default_rng(173)
    out = {kind: [] for kind in ("bar", "box", "violin", "line")}
    for i, dpi in enumerate((110, 150, 200, 250, 300)):
        n = 3 + i % 3
        means = rng.uniform(20, 80, n)
        out["bar"].append(render(ChartSpec(means.tolist(), (means * 0.08).tolist(), dpi=dpi)))
        stats = [(m - 18, m - 7, m, m + 8, m + 20) for m in means]
        out["box"].append(render_box(BoxSpec(stats, ylim=(0, 120), dpi=dpi)))
        samples = [rng.normal(float(m), 8, 24) for m in means]
        out["violin"].append(render_violin(ViolinSpec(samples, inner="box" if i % 2 else "lines",
                                                          ylim=(0, 120), dpi=dpi)))
        x = [1, 2, 3, 4, 5]
        ys = [rng.uniform(30, 70, len(x)).tolist(), rng.uniform(35, 75, len(x)).tolist()]
        out["line"].append(render_line(LineSpec(x, ys, errs=[[3] * 5, [4] * 5], dpi=dpi)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="backend")
    args = parser.parse_args()
    images = cases()  # generated before the clock starts
    sys.path.insert(0, os.path.abspath(args.backend))
    from figure_extractor import ExtractOptions, FigureExtractor

    extractor = FigureExtractor()
    for kind, group in images.items():
        opts = ExtractOptions(bar_names=False, titles=False, chart_type=kind)
        extractor.extract(group[0], opts)  # model and library warmup
        times = []
        for _ in range(2):
            for img in group:
                start = time.perf_counter()
                extractor.extract(img, opts)
                times.append(1000 * (time.perf_counter() - start))
        print(f"{kind}: n={len(times)} p50={np.median(times):.0f}ms p95={np.percentile(times, 95):.0f}ms")


if __name__ == "__main__":
    main()
