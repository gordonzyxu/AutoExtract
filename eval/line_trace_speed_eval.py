"""Compare the previous and vectorized duplicate-line check on a fixed chart.

The image is rendered before timing. One warmup pair precedes the timed pairs.
Panel JSON must match in every pair; only the duplicate-trace function changes.

    .venv/Scripts/python eval/line_trace_speed_eval.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "tests")]

import line_plots  # noqa: E402
from figure_extractor import ExtractOptions, FigureExtractor  # noqa: E402
from fuzz import random_line_spec  # noqa: E402
from synth import render_line  # noqa: E402


def previous_same_trace(a, b, line_w):
    xs = [point[0] for point in a.trace]
    differences = [abs(a.y_at(x) - b.y_at(x)) for x in xs if b.y_at(x) is not None]
    return (len(differences) > 0.6 * len(xs)
            and float(np.median(differences)) <= 2 * line_w + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--case", type=int, default=26)
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()
    if args.case < 0 or args.runs < 1:
        parser.error("--case must be nonnegative and --runs must be positive")

    rng = np.random.default_rng(args.seed)
    for _ in range(args.case + 1):
        spec = random_line_spec(rng)
    image = render_line(spec)
    extractor = FigureExtractor()
    options = ExtractOptions(chart_type="line", bar_names=False, titles=False)
    optimized = line_plots._same_trace
    times = {"previous": [], "optimized": []}
    try:
        for pair in range(args.runs + 1):
            order = ("previous", "optimized") if pair % 2 == 0 else ("optimized", "previous")
            outputs = {}
            for label in order:
                line_plots._same_trace = previous_same_trace if label == "previous" else optimized
                start = time.perf_counter()
                outputs[label] = [panel.to_dict() for panel in extractor.extract(image, options)]
                if pair:
                    times[label].append(time.perf_counter() - start)
            if outputs["previous"] != outputs["optimized"]:
                raise AssertionError(f"extraction changed in pair {pair}")
    finally:
        line_plots._same_trace = optimized

    print(f"seed {args.seed} case {args.case}: {args.runs} timed pairs, identical panel JSON")
    for label in ("previous", "optimized"):
        print(f"{label}: median {statistics.median(times[label]):.3f}s; "
              f"runs {[round(value, 3) for value in times[label]]}")


if __name__ == "__main__":
    main()
