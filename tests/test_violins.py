"""
Violin plots: synthetic violins of real samples with exactly known
min/Q1/median/Q3/max, in the styles Prism, ggplot2 and matplotlib draw.

    python -m pytest tests/test_violins.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
sys.path.insert(0, HERE)

from estimators import estimate  # noqa: E402
from figure_extractor import ExtractOptions, FigureExtractor  # noqa: E402
from synth import BoxSpec, ChartSpec, ViolinSpec, render, render_box, render_violin  # noqa: E402
from fuzz import image_variant, random_violin_spec  # noqa: E402

VIOLIN = ExtractOptions(bar_names=False, titles=False, chart_type="violin")


@pytest.fixture(scope="module")
def extractor():
    return FigureExtractor()


def _data(seed=1):
    rng = np.random.default_rng(seed)
    return [rng.normal(10, 2, 14), rng.normal(14, 3, 14), rng.gamma(4, 2, 14)]


CASES = {
    "prism_lines_white": ViolinSpec(_data(), inner="lines"),
    "prism_lines_colored": ViolinSpec(_data(), inner="lines", colors=["#888888", "#d62728", "#1f77b4"]),
    "lines_with_dots": ViolinSpec(_data(), inner="lines", colors=["#888888", "#d62728", "#1f77b4"], dots=True),
    "ggplot_inner_box": ViolinSpec(_data(), inner="box", colors=["#ffd27f"] * 3),
    "dark_fill": ViolinSpec(_data(2), inner="lines", colors=["#222244", "#333333", "#552222"]),
    "small_image": ViolinSpec(_data(3), inner="lines", dpi=120),
}


def _span(p):
    return abs(p.value_at(p.plot_top) - p.value_at(p.axis_y))


def test_median_only_panel_reselects_median_after_false_quartiles(extractor):
    rng = np.random.default_rng(7)
    for _ in range(10):
        spec = random_violin_spec(rng)
    assert spec.inner == "median" and len(spec.data) == 5
    p = extractor.extract(render_violin(spec), VIOLIN)[0]
    assert len(p.boxes) == 5
    assert all(not b.quartiles for b in p.boxes)
    assert abs(p.boxes[2].median - spec.stats[2][2]) / _span(p) < 0.01


def test_tip_adjacent_artifacts_do_not_create_quartiles(extractor):
    rng = np.random.default_rng(7)
    for _ in range(12):
        spec = random_violin_spec(rng)
    assert spec.inner == "median" and spec.dots and len(spec.data) == 3
    panels = extractor.extract(render_violin(spec), VIOLIN)
    assert len(panels) == 1 and len(panels[0].boxes) == 3
    for box, stat in zip(panels[0].boxes, spec.stats):
        assert not box.quartiles and box.q1 is None and box.q3 is None
        assert abs(box.median - stat[2]) / _span(panels[0]) < 0.01


def test_scatter_covered_inner_box_uses_three_drawn_bands(extractor):
    rng = np.random.default_rng(7)
    for _ in range(42):
        spec = random_violin_spec(rng)
    assert spec.inner == "box" and spec.dots and len(spec.data) == 2
    p = extractor.extract(render_violin(spec), VIOLIN)[0]
    first = p.boxes[0]
    assert first.quartiles
    for got, want in zip((first.q1, first.median, first.q3), spec.stats[0][1:4]):
        assert abs(got - want) / _span(p) < 0.01


def test_vertical_dot_stack_does_not_invent_an_inner_box(extractor):
    rng = np.random.default_rng(7)
    for _ in range(41):
        spec = random_violin_spec(rng)
    assert spec.inner == "median" and spec.dots and len(spec.data) == 3
    p = extractor.extract(render_violin(spec), VIOLIN)[0]
    assert all(not b.quartiles and b.q1 is None and b.q3 is None for b in p.boxes)
    assert abs(p.boxes[1].median - spec.stats[1][2]) / _span(p) < 0.01


@pytest.mark.parametrize("name", sorted(CASES))
def test_violin_values(extractor, name):
    spec = CASES[name]
    panels = extractor.extract(render_violin(spec), VIOLIN)
    assert len(panels) == 1 and panels[0].kind == "violin"
    p = panels[0]
    assert p.whisker_type == "unknown"  # the density may extend past the data
    assert len(p.boxes) == len(spec.data)
    span = _span(p)
    # ends move by a scatter dot's radius when a dot sits on the tip
    end_tol = 0.015 if spec.dots else 0.01
    for b, (lo, q1, med, q3, hi) in zip(p.boxes, spec.stats):
        assert abs(b.median - med) / span < 0.01, (b.median, med)
        assert abs(b.q1 - q1) / span < 0.01 and abs(b.q3 - q3) / span < 0.01, (b.q1, q1, b.q3, q3)
        assert abs(b.lo - lo) / span < end_tol and abs(b.hi - hi) / span < end_tol, (b.lo, lo, b.hi, hi)


def test_median_only_violin(extractor):
    spec = ViolinSpec(_data(), inner="median", colors=["#bbbbbb"] * 3)
    p = extractor.extract(render_violin(spec), VIOLIN)[0]
    for b, (lo, q1, med, q3, hi) in zip(p.boxes, spec.stats):
        assert abs(b.median - med) / _span(p) < 0.01
        assert b.q1 is None and b.q3 is None and not b.quartiles
    # S1 estimate only after confirming that the ends show sample min/max.
    assert estimate(None, p.boxes[0].median, None, 14, p.boxes[0].lo, p.boxes[0].hi).mean is None
    e = estimate(None, p.boxes[0].median, None, 14, p.boxes[0].lo, p.boxes[0].hi, "minmax")
    assert e.scenario == "S1" and e.mean is not None and e.sd > 0


@pytest.mark.parametrize("case_index", [10, 20])
def test_compressed_scatter_dots_do_not_invent_inner_box(extractor, case_index):
    rng = np.random.default_rng(123)
    spec = None
    for _ in range(case_index + 1):
        spec = random_violin_spec(rng)
    assert spec.inner == "median" and spec.dots
    image = image_variant(render_violin(spec), jpeg_quality=65, resize_factor=0.75)
    panels = extractor.extract(image, VIOLIN)
    assert len(panels) == 1 and len(panels[0].boxes) == len(spec.data)
    assert all(not b.quartiles and b.q1 is None and b.q3 is None for b in panels[0].boxes)


@pytest.mark.parametrize("seed,case_index,compressed", [(7, 0, False), (7, 18, False),
                                                       (123, 0, True), (123, 2, True),
                                                       (123, 6, True), (123, 18, True)])
def test_nearby_quartile_bands_and_thin_tips(extractor, seed, case_index, compressed):
    """Close drawn bands and narrow connected tips survive rasterization."""
    rng = np.random.default_rng(seed)
    for _ in range(case_index + 1):
        spec = random_violin_spec(rng)
    image = render_violin(spec)
    if compressed:
        image = image_variant(image, jpeg_quality=65, resize_factor=0.75)
    panels = extractor.extract(image, VIOLIN)
    assert len(panels) == 1 and len(panels[0].boxes) == len(spec.stats)
    span = spec.ylim[1] - spec.ylim[0]
    for box, expected in zip(panels[0].boxes, spec.stats):
        assert box.quartiles
        for got, want in zip((box.lo, box.q1, box.median, box.q3, box.hi), expected):
            assert got is not None and abs(got - want) / span < 0.01


def test_violin_page_ignores_other_charts(extractor):
    for img in (render(ChartSpec([12.0, 31.5, 22.0], [1.5, 3.0, 2.2])),
                render(ChartSpec([80, 40, 75, 55], [4, 3, 5, 4], dots=True, seed=3)),
                render_box(BoxSpec([(2, 5, 7, 9, 13), (4, 8, 11, 15, 19)], ylim=(0, 20)))):
        assert extractor.extract(img, VIOLIN) == []


def test_other_pages_ignore_violins(extractor):
    img = render_violin(CASES["prism_lines_colored"])
    for chart_type in ("bar", "box"):
        opts = ExtractOptions(bar_names=False, titles=False, chart_type=chart_type)
        assert extractor.extract(img, opts) == []


def test_s1_estimators_recover_normal_samples():
    rng = np.random.default_rng(0)
    for n in (10, 50):
        ms, sds = [], []
        for _ in range(400):
            x = rng.normal(50, 10, n)
            e = estimate(None, float(np.median(x)), None, n, x.min(), x.max(), "minmax")
            ms.append(e.mean)
            sds.append(e.sd)
        assert abs(np.mean(ms) - 50) < 0.6
        assert abs(np.mean(sds) / 10 - 1) < 0.08


def _seeded_violin(seed, index):
    rng = np.random.default_rng(seed)
    for _ in range(index + 1):
        spec = random_violin_spec(rng)
    return spec


def test_thick_median_on_dark_fill_is_kept(extractor):
    # A dark fill hides the outline, so the outline width (and the thin-line
    # limit derived from it) is underestimated; the thick white median was
    # erased and the violin dropped, or Q1 read as the median.
    spec = _seeded_violin(2026, 21)
    p = extractor.extract(render_violin(spec), VIOLIN)
    assert len(p) == 1 and len(p[0].boxes) == len(spec.data)
    spec = _seeded_violin(2026, 23)
    box = extractor.extract(render_violin(spec), VIOLIN)[0].boxes[0]
    q1, med, q3 = np.percentile(spec.data[0], [25, 50, 75])
    span = spec.ylim[1] - spec.ylim[0]
    assert box.quartiles
    assert max(abs(box.q1 - q1), abs(box.median - med), abs(box.q3 - q3)) < 0.01 * span


def test_bands_at_both_tips_of_a_lone_violin_are_not_quartiles(extractor):
    spec = _seeded_violin(2026, 6)
    assert spec.inner == "median" and len(spec.data) == 1
    box = extractor.extract(render_violin(spec), VIOLIN)[0].boxes[0]
    assert not box.quartiles
