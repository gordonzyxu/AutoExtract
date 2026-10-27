"""
Line plots: synthetic charts with exactly known points and error bars.

    python -m pytest tests/test_lines.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
sys.path.insert(0, HERE)

from figure_extractor import ExtractOptions, FigureExtractor  # noqa: E402
from line_plots import _error_bar  # noqa: E402
from fuzz import random_line_spec  # noqa: E402
from synth import BoxSpec, ChartSpec, LineSpec, render, render_box, render_line  # noqa: E402

LINE = ExtractOptions(bar_names=False, titles=False, chart_type="line")


@pytest.fixture(scope="module")
def extractor():
    return FigureExtractor()


CASES = {
    "two_series_errors": LineSpec([0, 7, 14, 21, 28], [[20, 22, 25, 27, 30], [16, 17, 17.5, 18, 19]],
                                  errs=[[1, 1.2, 1.1, 1.5, 1.4], [0.8, 1, 0.9, 1.1, 1.2]], legend=True),
    "three_colors": LineSpec([1, 2, 3, 4, 5, 6], [[5, 8, 6, 9, 11, 10], [3, 4, 4.5, 5, 6, 7], [9, 7, 6, 4, 3, 2]],
                             colors=["tab:red", "tab:blue", "tab:green"], markers=["o", "s", "D"],
                             errs=[[0.5] * 6, [0.4] * 6, [0.6] * 6]),
    "single_series": LineSpec([0, 1, 2, 3, 4, 5, 6, 7], [[2, 3, 5, 4, 6, 8, 7, 9]], errs=[[0.4] * 8]),
    "gridlines": LineSpec([0, 10, 20, 30], [[50, 60, 75, 70], [40, 42, 47, 52]], grid=True,
                          colors=["tab:purple", "tab:orange"], errs=[[3] * 4, [2] * 4]),
    "small_image": LineSpec([0, 1, 2, 3, 4], [[1, 2, 2.5, 3, 4]], errs=[[0.2] * 5], dpi=120),
}


def _span(p):
    return abs(p.value_at(p.plot_top) - p.value_at(p.axis_y))


def _match(series, truth_ys):
    """Extracted series whose point means are closest to a truth series."""
    best = None
    for s in series:
        means = [pt.mean for pt in s.points]
        if len(means) != len(truth_ys):
            continue
        d = float(np.mean(np.abs(np.array(means) - np.array(truth_ys))))
        if best is None or d < best[0]:
            best = (d, s)
    return best[1] if best else None


@pytest.mark.parametrize("name", sorted(CASES))
def test_line_values(extractor, name):
    spec = CASES[name]
    panels = extractor.extract(render_line(spec), LINE)
    assert len(panels) == 1 and panels[0].kind == "line"
    p = panels[0]
    span = _span(p)
    assert p.x_slope is not None, [t[2] for t in p.x_ticks]
    for k, ys in enumerate(spec.ys):
        s = _match(p.series, ys)
        assert s is not None, [[round(pt.mean, 2) for pt in sr.points] for sr in p.series]
        for pt, x, y, e in zip(s.points, spec.x, ys, spec.errs[k]):
            assert abs(pt.mean - y) / span < 0.01, (pt.mean, y)
            assert abs(pt.x - x) < 0.02 * (max(spec.x) - min(spec.x)) + 1e-9, (pt.x, x)
            if pt.uncertain and pt.mean_plus_err is None:
                continue                 # a marker hidden under another one: flagged for a check
            assert pt.mean_plus_err is not None and abs((pt.mean_plus_err - pt.mean) - e) / span < 0.015


def test_line_without_markers_sampled_at_ticks(extractor):
    spec = LineSpec([0, 10, 20, 30, 40, 50], [[1, 3, 2, 5, 4, 6]], markers=[None], colors=["tab:red"])
    p = extractor.extract(render_line(spec), LINE)[0]
    s = p.series[0]
    assert not s.markers and len(s.points) == 6
    for pt, y in zip(s.points, spec.ys[0]):
        assert abs(pt.mean - y) / _span(p) < 0.01


def test_line_page_ignores_bars_and_boxes(extractor):
    for img in (render(ChartSpec([12.0, 31.5, 22.0], [1.5, 3.0, 2.2])),
                render_box(BoxSpec([(2, 5, 7, 9, 13), (4, 8, 11, 15, 19)], ylim=(0, 20)))):
        assert extractor.extract(img, LINE) == []


def test_short_capped_whisker_is_not_lost_at_marker_edge():
    mask = np.zeros((40, 40), np.uint8)
    import cv2
    cv2.circle(mask, (20, 20), 5, 1, -1)
    mask[12:17, 20] = 1
    mask[12, 17:24] = 1
    assert _error_bar(mask.astype(bool), 20, 20, 6, -1, 1) == pytest.approx(12)
    mask[12:17, :] = 0
    assert _error_bar(mask.astype(bool), 20, 20, 6, -1, 1) is None


def test_dark_series_does_not_borrow_neighbour_cap(extractor):
    rng = np.random.default_rng(7)
    for _ in range(14):
        spec = random_line_spec(rng)
    assert spec.colors[0] == "black" and spec.colors[2] == "#555555"
    p = extractor.extract(render_line(spec), LINE)[0]
    black = _match(p.series, spec.ys[0])
    assert black is not None
    assert black.points[1].mean_plus_err is None or black.points[1].uncertain


def test_aligned_markers_use_numeric_tick_grid(extractor):
    rng = np.random.default_rng(7)
    spec = random_line_spec(rng)
    p = extractor.extract(render_line(spec), LINE)[0]
    purple = _match(p.series, spec.ys[1])
    assert purple is not None and p.x_slope is not None
    assert abs(purple.points[0].x - spec.x[0]) < 0.01 * (spec.x[-1] - spec.x[0])


def test_shared_tick_grid_does_not_copy_cap_fragments_across_series(extractor):
    rng = np.random.default_rng(7)
    for _ in range(27):
        spec = random_line_spec(rng)
    assert len(spec.x) == 6 and len(spec.ys) == 4 and spec.hollow
    p = extractor.extract(render_line(spec), LINE)[0]
    assert len(p.series) == 4
    assert p.x_ticks[0][1:] == (0.0, "0")
    for colour in ((0, 0, 0), (180, 119, 31), (14, 127, 255)):
        series = next(s for s in p.series if s.color == colour)
        assert len(series.points) == len(spec.x)
        assert max(abs(point.x - x) for point, x in zip(series.points, spec.x)) < 0.01 * spec.x[-1]


def test_shared_tick_grid_preserves_a_regularly_staggered_series(extractor):
    import cv2
    import matplotlib.pyplot as plt

    ticks = np.array([0, 2, 4, 6, 8, 10], float)
    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    ax.plot(ticks, [20, 23, 26, 25, 29, 31], "-o", color="tab:red")
    ax.plot(ticks, [45, 48, 51, 50, 53, 55], "-s", color="tab:blue")
    ax.plot(ticks + 0.8, [67, 65, 69, 66, 71, 68], "-D", color="tab:green")
    ax.set_xticks(ticks)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.canvas.draw()
    image = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    plt.close(fig)

    p = extractor.extract(image, LINE)[0]
    staggered = _match(p.series, [67, 65, 69, 66, 71, 68])
    assert staggered is not None and len(staggered.points) == len(ticks)
    assert max(abs(pt.x - expected) for pt, expected in zip(staggered.points, ticks + 0.8)) < 0.03


@pytest.mark.parametrize("case,colour", [(17, "tab:purple"), (25, "tab:orange")])
def test_crossing_series_cap_is_flagged_for_review(extractor, case, colour):
    rng = np.random.default_rng(7)
    for _ in range(case + 1):
        spec = random_line_spec(rng)
    p = extractor.extract(render_line(spec), LINE)[0]
    index = spec.colors.index(colour)
    series = _match(p.series, spec.ys[index])
    assert series is not None
    assert series.points[0].uncertain


def test_unmeasured_error_bars_are_flagged_when_panel_has_capped_points(extractor):
    rng = np.random.default_rng(7)
    for _ in range(14):
        spec = random_line_spec(rng)
    panel = extractor.extract(render_line(spec), LINE)[0]
    points = [pt for series in panel.series for pt in series.points]
    assert any(pt.cap_hi_px is not None for pt in points)
    missing = [pt for pt in points if pt.cap_hi_px is None and pt.cap_lo_px is None]
    assert missing and all(pt.uncertain for pt in missing)
    assert any("error bars need review" in warning for warning in panel.warnings)


def _seeded_line(seed, index):
    rng = np.random.default_rng(seed)
    for _ in range(index + 1):
        spec = random_line_spec(rng)
    return spec


def test_series_crossing_a_flat_segment_is_not_a_dashed_reference(extractor):
    # The red series' last segment is flat; the black series crosses its row.
    # That row once read as a dashed reference line and was erased, and the
    # red line's pale anti-aliased fringe was traced in place of its core.
    spec = _seeded_line(2026, 28)
    p = extractor.extract(render_line(spec), LINE)[0]
    red = _match(p.series, spec.ys[1])
    assert red is not None and len(red.points) == len(spec.x)
    span = spec.ylim[1] - spec.ylim[0]
    assert max(abs(pt.mean - y) for pt, y in zip(red.points, spec.ys[1])) < 0.01 * span


def test_one_stray_cap_does_not_flag_a_chart_without_error_bars(extractor):
    spec = _seeded_line(2026, 26)
    assert spec.errs is None
    p = extractor.extract(render_line(spec), LINE)[0]
    points = [pt for s in p.series for pt in s.points]
    assert sum(pt.uncertain for pt in points) <= 2
    assert not any("error bars need review" in w for w in p.warnings)


def test_numeric_x_labels_are_read_without_text_detection(extractor, monkeypatch):
    # Compressed small chart whose tick strokes are too faint: the labels are
    # split by their ink and recognised directly (detection takes seconds).
    from fuzz import image_variant
    spec = _seeded_line(2026, 35)
    img = image_variant(render_line(spec), 65, 0.75)
    calls = []
    original = type(extractor.ocr).boxes
    monkeypatch.setattr(type(extractor.ocr), "boxes",
                        lambda self, image: calls.append(image.shape) or original(self, image))
    p = extractor.extract(img, LINE)[0]
    assert p.x_slope is not None and not calls
