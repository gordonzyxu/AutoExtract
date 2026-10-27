"""
Randomised stress test: many synthetic charts with random values and styles.

    .venv/Scripts/python tests/fuzz.py [N] [--seed S] [--show-failures DIR] [--dot | --box | --violin | --line]

Reports detection rate and error distribution; failures can be saved as
images for inspection. The bar/dot chart pass criterion checks means and group
count; cap errors are reported separately because uncertain caps need review.
Exit code 1 if fewer than 95% of charts pass.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("AUTOEXTRACT_BACKEND", os.path.join(HERE, "..", "backend")))
sys.path.insert(0, HERE)

from figure_extractor import ExtractOptions, FigureExtractor  # noqa: E402
from synth import BoxSpec, ChartSpec, LineSpec, ViolinSpec, render, render_box, render_line, render_violin  # noqa: E402

PALETTES = [
    None,
    ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3", "#8c8c8c"],
    ["#222222", "#888888", "#cccccc", "#555555", "#aaaaaa", "#333333", "#999999", "#666666"],
    ["#ffd27f", "#b0c4de", "#d8bfd8", "#98fb98", "#f4a460", "#87ceeb", "#ffb6c1", "#e0e0e0"],
]


def image_variant(img: np.ndarray, jpeg_quality: int | None, resize_factor: float) -> np.ndarray:
    """Simulate a resized screenshot or a lossy chart image."""
    if resize_factor != 1:
        method = cv2.INTER_AREA if resize_factor < 1 else cv2.INTER_CUBIC
        img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, interpolation=method)
    if jpeg_quality is not None:
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return img


def random_spec(rng: np.random.Generator, kind_override: str | None = None) -> ChartSpec:
    n = int(rng.integers(2, 9))
    scale = float(10.0 ** rng.integers(-1, 4))
    kind = kind_override or ("dot" if rng.random() < 0.15 else "bar")
    negative = kind == "bar" and rng.random() < 0.12
    means = rng.uniform(0.15, 1.0, n) * scale
    if negative:
        means *= rng.choice([-1, 1], n)
    errs = means * 0 + rng.uniform(0.02, 0.15, n) * scale
    pal = PALETTES[int(rng.integers(0, len(PALETTES)))]
    colors = None if pal is None else [pal[i % len(pal)] for i in range(n)]
    edge = None if (colors and rng.random() < 0.4) else "black"
    err_color = "black" if (edge or rng.random() < 0.5) else "#555599"
    ylim = None
    if negative:
        lo = float(min(0, (means - errs).min()) * 1.3)
        hi = float(max(0, (means + errs).max()) * 1.3)
        ylim = (lo, hi)
    return ChartSpec(
        means=[float(m) for m in means], errs=[float(e) for e in errs], kind=kind,
        colors=colors, edgecolor=edge, error_color=err_color,
        linewidth=float(rng.choice([1.0, 1.5, 2.0])),
        dots=bool(rng.random() < 0.35), grid=bool(rng.random() < 0.15), frame=bool(rng.random() < 0.15),
        ylim=ylim, figsize=(float(1.6 + 0.45 * n), float(rng.uniform(2.4, 3.6))),
        dpi=int(rng.choice([110, 150, 200, 300])), seed=int(rng.integers(0, 10 ** 6)),
    )


def random_box_spec(rng: np.random.Generator) -> BoxSpec:
    n = int(rng.integers(1, 7))
    scale = float(10.0 ** rng.integers(-1, 4))
    stats = []
    for _ in range(n):
        q = np.sort(rng.uniform(0.05, 0.95, 5)) * scale
        # keep the box and whiskers at least a few pixels apart
        if min(np.diff(q)) < 0.03 * scale:
            q = np.linspace(0.1, 0.9, 5) * scale * rng.uniform(0.6, 1.0)
        stats.append(tuple(float(v) for v in q))
    pal = PALETTES[int(rng.integers(0, len(PALETTES)))]
    colors = None if pal is None else [pal[i % len(pal)] for i in range(n)]
    outliers = None
    if rng.random() < 0.3:
        outliers = [[s[4] + 0.04 * scale] if rng.random() < 0.5 else [] for s in stats]
    return BoxSpec(
        stats=stats, colors=colors, outliers=outliers, ylim=(0, scale),
        linewidth=float(rng.choice([0.8, 1.2, 1.8])), caps=bool(rng.random() < 0.8),
        dashed=bool(rng.random() < 0.15), grid=bool(rng.random() < 0.15), frame=bool(rng.random() < 0.15),
        figsize=(float(1.8 + 0.5 * n), float(rng.uniform(2.4, 3.6))), dpi=int(rng.choice([110, 150, 200, 300])),
    )


def fuzz_boxes(n: int, rng, ex, opts, show, jpeg_quality=None, resize_factor=1) -> int:
    ok, errs, failures = 0, [], []
    scan_ms = []
    t0 = time.time()
    for i in range(n):
        spec = random_box_spec(rng)
        img = image_variant(render_box(spec), jpeg_quality, resize_factor)
        scan_start = time.perf_counter()
        panels = ex.extract(img, opts)
        scan_ms.append(1000 * (time.perf_counter() - scan_start))
        reason = None
        if len(panels) != 1 or panels[0].kind != "box":
            reason = f"{len(panels)} panels" + (f" ({panels[0].kind})" if panels else "")
        elif len(panels[0].boxes) != len(spec.stats):
            reason = f"{len(panels[0].boxes)}/{len(spec.stats)} boxes"
        else:
            span = spec.ylim[1] - spec.ylim[0]
            e = [abs(got - want) / span * 100 if got is not None else 100.0
                 for b, st in zip(panels[0].boxes, spec.stats)
                 for got, want in zip((b.lo, b.q1, b.median, b.q3, b.hi), st)]
            errs += e
            if max(e) >= 1:
                reason = f"stat err {max(e):.2f}%"
        if reason is None:
            ok += 1
        else:
            failures.append((i, spec, reason))
            if show:
                os.makedirs(show, exist_ok=True)
                cv2.imwrite(os.path.join(show, f"box_fail_{i:03d}.png"), img)
    dt = time.time() - t0
    print(f"box plots: {n}  fully correct: {ok} ({100 * ok / n:.1f}%)  time {dt:.1f}s ({1000 * dt / n:.0f} ms/chart)")
    print(f"extraction only: p50 {np.median(scan_ms):.0f} ms  p95 {np.percentile(scan_ms, 95):.0f} ms")
    if errs:
        e = np.array(errs)
        print(f"stat err: median {np.median(e):.3f}%  p95 {np.percentile(e, 95):.3f}%  max {e.max():.2f}%")
    for i, spec, reason in failures[:25]:
        print(f"  #{i}: {reason}  n={len(spec.stats)} colors={'yes' if spec.colors else 'no'} caps={spec.caps} "
              f"dashed={spec.dashed} grid={spec.grid} frame={spec.frame} lw={spec.linewidth} dpi={spec.dpi}")
    return 0 if ok >= 0.95 * n else 1


def random_violin_spec(rng: np.random.Generator) -> ViolinSpec:
    n = int(rng.integers(1, 7))
    scale = float(10.0 ** rng.integers(-1, 4))
    data = []
    for _ in range(n):
        size = int(rng.integers(8, 60))
        kind = rng.choice(["normal", "gamma", "lognormal", "bimodal"])
        if kind == "normal":
            d = rng.normal(0.5, 0.12, size)
        elif kind == "gamma":
            d = rng.gamma(3, 0.07, size) + 0.15
        elif kind == "lognormal":
            d = np.exp(rng.normal(-0.9, 0.35, size))
        else:
            d = np.concatenate([rng.normal(0.35, 0.05, size // 2), rng.normal(0.65, 0.05, size - size // 2)])
        data.append(np.clip(d, 0.03, 0.97) * scale)
    pal = PALETTES[int(rng.integers(0, len(PALETTES)))]
    return ViolinSpec(
        data=data, inner=str(rng.choice(["lines", "lines", "box", "median"])),
        colors=None if pal is None else [pal[i % len(pal)] for i in range(n)],
        linewidth=float(rng.choice([0.8, 1.2, 1.8])), dots=bool(rng.random() < 0.3), ylim=(0, scale),
        figsize=(float(1.8 + 0.55 * n), float(rng.uniform(2.6, 3.6))), dpi=int(rng.choice([110, 150, 200, 300])),
        seed=int(rng.integers(0, 10 ** 6)),
    )


def fuzz_violins(n: int, rng, ex, opts, show, jpeg_quality=None, resize_factor=1) -> int:
    ok, errs, failures = 0, [], []
    scan_ms = []
    t0 = time.time()
    for i in range(n):
        spec = random_violin_spec(rng)
        img = image_variant(render_violin(spec), jpeg_quality, resize_factor)
        scan_start = time.perf_counter()
        panels = ex.extract(img, opts)
        scan_ms.append(1000 * (time.perf_counter() - scan_start))
        reason = None
        if len(panels) != 1 or panels[0].kind != "violin":
            reason = f"{len(panels)} panels"
        elif len(panels[0].boxes) != len(spec.data):
            reason = f"{len(panels[0].boxes)}/{len(spec.data)} violins"
        else:
            span = spec.ylim[1] - spec.ylim[0]
            e = []
            for b, st in zip(panels[0].boxes, spec.stats):
                pairs = [(b.lo, st[0]), (b.median, st[2]), (b.hi, st[4])]
                if spec.inner != "median":
                    pairs += [(b.q1, st[1]), (b.q3, st[3])]
                elif b.quartiles or b.q1 is not None or b.q3 is not None:
                    e.append(100.0)  # claiming undrawn quartiles is a false measurement
                e += [abs(g - w) / span * 100 if g is not None else 100.0 for g, w in pairs]
            errs += e
            if max(e) >= 1:
                reason = f"stat err {max(e):.2f}%"
        if reason is None:
            ok += 1
        else:
            failures.append((i, spec, reason))
            if show:
                os.makedirs(show, exist_ok=True)
                cv2.imwrite(os.path.join(show, f"violin_fail_{i:03d}.png"), img)
    dt = time.time() - t0
    print(f"violin plots: {n}  fully correct: {ok} ({100 * ok / n:.1f}%)  time {dt:.1f}s ({1000 * dt / n:.0f} ms/chart)")
    print(f"extraction only: p50 {np.median(scan_ms):.0f} ms  p95 {np.percentile(scan_ms, 95):.0f} ms")
    if errs:
        e = np.array(errs)
        print(f"stat err: median {np.median(e):.3f}%  p95 {np.percentile(e, 95):.3f}%  max {e.max():.2f}%")
    for i, spec, reason in failures[:25]:
        print(f"  #{i}: {reason}  n={len(spec.data)} inner={spec.inner} colors={'yes' if spec.colors else 'no'} "
              f"dots={spec.dots} lw={spec.linewidth} dpi={spec.dpi}")
    return 0 if ok >= 0.95 * n else 1


LINE_COLOURS = ["black", "tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple", "#555555"]


def random_line_spec(rng: np.random.Generator) -> LineSpec:
    k = int(rng.integers(1, 5))
    npts = int(rng.integers(3, 10))
    step = float(rng.choice([1, 2, 5, 7, 10]))
    x = [float(step * i) for i in range(npts)]
    scale = float(10.0 ** rng.integers(-1, 4))
    ys, errs = [], []
    for _ in range(k):
        y = np.clip(0.5 + np.cumsum(rng.normal(0, 0.08, npts)), 0.12, 0.88) * scale
        ys.append([float(v) for v in y])
        errs.append([float(v) for v in rng.uniform(0.01, 0.05, npts) * scale])
    colours = list(rng.choice(LINE_COLOURS, k, replace=False))
    markers = [str(m) for m in rng.choice(["o", "s", "^", "D"], k)]
    return LineSpec(
        x=x, ys=ys, errs=errs if rng.random() < 0.8 else None, colors=colours, markers=markers,
        hollow=bool(rng.random() < 0.25), linewidth=float(rng.choice([1.0, 1.5, 2.0])),
        legend=bool(rng.random() < 0.4), grid=bool(rng.random() < 0.15), ylim=(0, scale),
        figsize=(float(rng.uniform(3.0, 4.5)), float(rng.uniform(2.6, 3.6))), dpi=int(rng.choice([110, 150, 200, 300])),
    )


def fuzz_lines(n: int, rng, ex, opts, show, jpeg_quality=None, resize_factor=1) -> int:
    """
    Match series one-to-one, requiring exact counts and x/y values within 1%
    of their axis ranges. Measured unflagged caps must be within 1.5%; missing
    unflagged caps are reported as a separate completeness metric.
    """
    ok, complete, missing_caps, errs, failures = 0, 0, 0, [], []
    measured_caps = flagged_caps = flagged_missing_caps = 0
    scan_ms = []
    t0 = time.time()
    for i in range(n):
        spec = random_line_spec(rng)
        img = image_variant(render_line(spec), jpeg_quality, resize_factor)
        scan_start = time.perf_counter()
        panels = ex.extract(img, opts)
        scan_ms.append(1000 * (time.perf_counter() - scan_start))
        reason = None
        missing_here = 0
        if len(panels) != 1 or panels[0].kind != "line":
            reason = f"{len(panels)} panels"
        else:
            p = panels[0]
            span = spec.ylim[1] - spec.ylim[0]
            matches = match_line_series(spec, p.series)
            if matches is None:
                reason = f"series mismatch: expected {len(spec.ys)} x {len(spec.x)} points, got " \
                         f"{[len(s.points) for s in p.series]}"
            for k, s in enumerate(matches or []):
                ys = spec.ys[k]
                if any(pt.mean is None or pt.x is None for pt in s.points):
                    reason = f"series {k + 1}: uncalibrated point"
                    break
                e = [abs(pt.mean - y) / span * 100 for pt, y in zip(s.points, ys)]
                e += [abs(pt.x - x) / (max(spec.x) - min(spec.x)) * 100
                      for pt, x in zip(s.points, spec.x)]
                if spec.errs:
                    missing_here += sum(pt.mean_plus_err is None and not pt.uncertain for pt in s.points)
                    measured_caps += sum(pt.mean_plus_err is not None and not pt.uncertain for pt in s.points)
                    flagged_caps += sum(pt.mean_plus_err is not None and pt.uncertain for pt in s.points)
                    flagged_missing_caps += sum(pt.mean_plus_err is None and pt.uncertain for pt in s.points)
                    # error bars flagged for a check count as caught, as for bars
                    e += [abs((pt.mean_plus_err - pt.mean) - er) / span * 100 / 1.5
                          for pt, er in zip(s.points, spec.errs[k])
                          if pt.mean_plus_err is not None and not pt.uncertain]
                errs += e
                if max(e) >= 1:
                    reason = f"series {k + 1}: err {max(e):.2f}%"
                    break
        if reason is None:
            ok += 1
            if not missing_here:
                complete += 1
        else:
            failures.append((i, spec, reason))
            if show:
                os.makedirs(show, exist_ok=True)
                cv2.imwrite(os.path.join(show, f"line_fail_{i:03d}.png"), img)
        missing_caps += missing_here
    dt = time.time() - t0
    print(f"line plots: {n}  value-correct: {ok} ({100 * ok / n:.1f}%)  "
          f"no unflagged missing caps: {complete} ({100 * complete / n:.1f}%)  "
          f"missing unflagged caps: {missing_caps}  "
          f"time {dt:.1f}s ({1000 * dt / n:.0f} ms/chart, including rendering)")
    print(f"extraction only: p50 {np.median(scan_ms):.0f} ms  p95 {np.percentile(scan_ms, 95):.0f} ms")
    print(f"caps: {measured_caps} confirmed, {flagged_caps} measured but flagged, "
          f"{flagged_missing_caps} missing and flagged, {missing_caps} missing and unflagged")
    if errs:
        e = np.array(errs)
        print(f"point err: median {np.median(e):.3f}%  p95 {np.percentile(e, 95):.3f}%  max {e.max():.2f}%")
    for i, spec, reason in failures[:25]:
        print(f"  #{i}: {reason}  series={len(spec.ys)} pts={len(spec.x)} colours={spec.colors} "
              f"markers={spec.markers} hollow={spec.hollow} legend={spec.legend} dpi={spec.dpi}")
    return 0 if ok >= 0.95 * n else 1


def match_line_series(spec: LineSpec, series):
    """Find a one-to-one match, rejecting missing or extra series/points."""
    if len(series) != len(spec.ys) or any(len(s.points) != len(spec.x) for s in series):
        return None
    xspan = max(spec.x) - min(spec.x)
    yspan = spec.ylim[1] - spec.ylim[0]
    cost = np.array([
        [np.mean([(abs(pt.mean - y) / yspan + abs(pt.x - x) / xspan)
                  if pt.mean is not None and pt.x is not None else 100
                  for pt, x, y in zip(s.points, spec.x, ys)])
         for s in series]
        for ys in spec.ys
    ])
    rows, cols = linear_sum_assignment(cost)
    return [series[cols[k]] for k in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show-failures", default=None)
    ap.add_argument("--box", action="store_true", help="random box plots instead of bar charts")
    ap.add_argument("--violin", action="store_true", help="random violin plots")
    ap.add_argument("--line", action="store_true", help="random line plots")
    ap.add_argument("--dot", action="store_true", help="random dot/mean-and-whisker plots only")
    ap.add_argument("--bar-only", action="store_true", help="random bars only (default mixes bars and dots)")
    ap.add_argument("--jpeg-quality", type=int, default=None, help="JPEG round-trip at quality 1-100")
    ap.add_argument("--resize-factor", type=float, default=1, help="rescale before extraction, e.g. 0.5")
    args = ap.parse_args()
    if args.jpeg_quality is not None and not 1 <= args.jpeg_quality <= 100:
        ap.error("--jpeg-quality must be between 1 and 100")
    if args.resize_factor <= 0:
        ap.error("--resize-factor must be positive")
    if sum((args.box, args.violin, args.line, args.dot, args.bar_only)) > 1:
        ap.error("choose only one chart type")
    rng = np.random.default_rng(args.seed)
    ex = FigureExtractor()
    opts = ExtractOptions(bar_names=False, titles=False)
    if args.box:
        opts.chart_type = "box"
        return fuzz_boxes(args.n, rng, ex, opts, args.show_failures, args.jpeg_quality, args.resize_factor)
    if args.violin:
        opts.chart_type = "violin"
        return fuzz_violins(args.n, rng, ex, opts, args.show_failures, args.jpeg_quality, args.resize_factor)
    if args.line:
        opts.chart_type = "line"
        return fuzz_lines(args.n, rng, ex, opts, args.show_failures, args.jpeg_quality, args.resize_factor)
    if args.dot:
        opts.chart_type = "dot"
    ok = 0
    mean_errs, cap_errs, flagged, failures = [], [], 0, []
    scan_ms = []
    t0 = time.time()
    for i in range(args.n):
        spec = random_spec(rng, kind_override="dot" if args.dot else "bar" if args.bar_only else None)
        img = image_variant(render(spec), args.jpeg_quality, args.resize_factor)
        scan_start = time.perf_counter()
        panels = ex.extract(img, opts)
        scan_ms.append(1000 * (time.perf_counter() - scan_start))
        reason = None
        if len(panels) != 1:
            reason = f"{len(panels)} panels"
        elif (args.dot or args.bar_only) and panels[0].kind != ("dot" if args.dot else "bar"):
            reason = f"detected as {panels[0].kind}, expected {'dot' if args.dot else 'bar'}"
        elif len(panels[0].bars) != len(spec.means):
            reason = f"{len(panels[0].bars)}/{len(spec.means)} bars"
        else:
            p = panels[0]
            rngv = abs(p.value_at(p.plot_top) - p.value_at(p.axis_y))
            em = [abs(b.mean - m) / rngv * 100 for b, m in zip(p.bars, spec.means)]
            mean_errs += em
            for b, e in zip(p.bars, spec.errs):
                if b.uncertain:
                    flagged += 1
                else:
                    cap_errs.append(abs((b.mean_plus_err - b.mean) - e) / rngv * 100)
            if max(em) >= 1:
                reason = f"mean err {max(em):.2f}%"
        if reason is None:
            ok += 1
        else:
            failures.append((i, spec, reason))
            if args.show_failures:
                os.makedirs(args.show_failures, exist_ok=True)
                cv2.imwrite(os.path.join(args.show_failures, f"fail_{i:03d}.png"), img)
    dt = time.time() - t0
    label = "dot plots" if args.dot else "bar charts" if args.bar_only else "bar/dot charts"
    print(f"{label}: {args.n}  mean-correct charts: {ok} ({100 * ok / args.n:.1f}%)  "
          f"time {dt:.1f}s ({1000 * dt / args.n:.0f} ms/chart, including rendering)")
    print(f"extraction only: p50 {np.median(scan_ms):.0f} ms  p95 {np.percentile(scan_ms, 95):.0f} ms")
    if mean_errs:
        me, ce = np.array(mean_errs), np.array(cap_errs) if cap_errs else np.zeros(1)
        print(f"mean err: median {np.median(me):.3f}%  p95 {np.percentile(me, 95):.3f}%  max {me.max():.2f}%")
        print(f"cap err (unflagged): median {np.median(ce):.3f}%  p95 {np.percentile(ce, 95):.3f}%  max {ce.max():.2f}%  "
              f"| flagged {flagged} of {flagged + len(cap_errs)}")
    for i, spec, reason in failures[:25]:
        print(f"  #{i}: {reason}  kind={spec.kind} n={len(spec.means)} edge={spec.edgecolor} colors={'yes' if spec.colors else 'no'} "
              f"dots={spec.dots} grid={spec.grid} frame={spec.frame} dpi={spec.dpi} neg={spec.ylim is not None}")
    return 0 if ok >= 0.95 * args.n else 1


if __name__ == "__main__":
    sys.exit(main())
