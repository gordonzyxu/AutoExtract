"""
Accuracy tests on synthetic charts with known values.

    .venv/Scripts/python -m pytest tests -q

Tolerance is 1% of the axis range for means and 1.5% for error-bar caps
(the cap is a thin line, so a pixel matters more).
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
sys.path.insert(0, HERE)

from figure_extractor import OCR, ExtractOptions, FigureExtractor, Panel, parse_number  # noqa: E402
from synth import BoxSpec, ChartSpec, LineSpec, ViolinSpec, render, render_box, render_line, render_violin  # noqa: E402

FAST = ExtractOptions(bar_names=False, titles=False)


def test_identical_ocr_crops_reuse_recognition_without_reusing_changed_pixels():
    from collections import OrderedDict

    class Engine:
        calls = 0

        def __call__(self, _image, **_kwargs):
            self.calls += 1
            return [("8", 0.9)], None

    ocr = OCR.__new__(OCR)
    ocr.engine = Engine()
    ocr._line_cache = OrderedDict()
    ocr.line_cache_hits = 0
    crop = np.full((32, 20, 3), 255, np.uint8)
    assert ocr.line(crop) == ocr.line(crop.copy())
    assert ocr.engine.calls == 1 and ocr.line_cache_hits == 1
    changed = crop.copy()
    changed[10, 10] = 0
    ocr.line(changed)
    assert ocr.engine.calls == 2


@pytest.fixture(scope="module")
def extractor():
    return FigureExtractor()


CASES = {
    "outlined_white": ChartSpec([12.0, 31.5, 22.0, 8.4], [1.5, 3.0, 2.2, 0.9]),
    "filled_colors": ChartSpec([45, 72, 60], [5, 6, 4], colors=["#4c72b0", "#dd8452", "#55a868"]),
    "dark_fill": ChartSpec([2.4, 1.1, 3.3], [0.3, 0.2, 0.4], colors=["#222244", "#333333", "#552222"]),
    "no_outline": ChartSpec([150, 210, 95, 180], [12, 20, 9, 15], colors=["#e377c2", "#17becf", "#bcbd22", "#7f7f7f"], edgecolor=None),
    "colored_error": ChartSpec([30, 52, 41], [3, 5, 4], colors=["#ffd27f", "#b0c4de", "#d8bfd8"],
                               edgecolor=None, error_color="#7a5cb8"),
    "with_dots": ChartSpec([80, 40, 75, 55], [4, 3, 5, 4], dots=True, seed=3),
    "gridlines": ChartSpec([0.62, 0.81, 0.44], [0.05, 0.06, 0.04], grid=True, colors=["#999999"] * 3),
    "frame": ChartSpec([3200, 1800, 2600], [180, 150, 210], frame=True),
    "many_bars": ChartSpec([5, 9, 7, 12, 3, 8, 10, 6], [0.5, 0.8, 0.6, 1.0, 0.4, 0.7, 0.9, 0.5],
                           figsize=(5.0, 3.0)),
    "small_image": ChartSpec([18, 27, 22], [2, 3, 2], dpi=110),
    "decimal_ticks": ChartSpec([0.35, 1.25, 0.8], [0.1, 0.15, 0.12], ylim=(0, 1.5), yticks=[0, 0.5, 1.0, 1.5]),
    "negative": ChartSpec([2.5, -4.5, -5.5, 1.0], [0.8, 1.2, 1.5, 0.5], ylim=(-10, 5),
                          colors=["#666666", "#d62728", "#1f77b4", "#ff7f0e"]),
    "nonzero_floor": ChartSpec([62, 85, 74], [3, 4, 3], ylim=(50, 100), yticks=[50, 60, 70, 80, 90, 100]),
    "dot_plot": ChartSpec([110, 170, 126, 116], [44, 52, 35, 45], kind="dot", dots=True,
                          colors=["#1f77b4", "#d62728", "#2ca02c", "#9467bd"], ylim=(0, 300)),
}


def _axis_range(panel):
    return abs(panel.value_at(panel.plot_top) - panel.value_at(panel.axis_y))


@pytest.mark.parametrize("name", sorted(CASES))
def test_values(extractor, name):
    spec = CASES[name]
    panels = extractor.extract(render(spec), FAST)
    assert len(panels) == 1, f"expected one panel, found {len(panels)}"
    p = panels[0]
    assert len(p.bars) == len(spec.means), [round(b.mean, 3) for b in p.bars]
    rng = _axis_range(p)
    for bar, m, e in zip(p.bars, spec.means, spec.errs):
        assert abs(bar.mean - m) / rng < 0.01, f"mean {bar.mean:.4g} vs {m}"
        got_err = bar.mean_plus_err - bar.mean
        if not bar.uncertain:
            assert abs(got_err - abs(e)) / rng < 0.015, f"error {got_err:.4g} vs {e}"


def test_kind_dot(extractor):
    p = extractor.extract(render(CASES["dot_plot"]), FAST)[0]
    assert p.kind == "dot"


def test_dedicated_dot_mode_accepts_dots_and_rejects_bars(extractor):
    opts = ExtractOptions(chart_type="dot", bar_names=False, titles=False)
    panels = extractor.extract(render(CASES["dot_plot"]), opts)
    assert len(panels) == 1 and panels[0].kind == "dot"
    assert len(panels[0].bars) == len(CASES["dot_plot"].means)
    assert extractor.extract(render(CASES["filled_colors"]), opts) == []
    assert extractor.extract(render_box(BoxSpec([(2, 5, 7, 9, 13), (4, 8, 11, 15, 19)],
                                                ylim=(0, 20))), opts) == []
    assert extractor.extract(render_violin(ViolinSpec([np.random.default_rng(1).normal(10, 2, 20),
                                                       np.random.default_rng(2).normal(14, 2, 20)],
                                                     inner="lines")), opts) == []
    assert extractor.extract(render_line(LineSpec([0, 7, 14, 21], [[20, 22, 25, 27], [16, 17, 18, 19]],
                                                  errs=[[1] * 4, [0.8] * 4])), opts) == []


def test_low_resolution_dot_with_split_mean_stroke(extractor):
    from fuzz import random_spec

    rng = np.random.default_rng(7)
    for _ in range(47):
        spec = random_spec(rng, kind_override="dot")
    assert spec.dpi == 110 and len(spec.means) == 4
    panels = extractor.extract(render(spec), ExtractOptions(chart_type="dot", bar_names=False, titles=False))
    assert len(panels) == 1 and len(panels[0].bars) == 4
    axis_range = _axis_range(panels[0])
    assert max(abs(b.mean - expected) / axis_range for b, expected in zip(panels[0].bars, spec.means)) < 0.01


def test_dot_means_are_recovered_when_they_cross_light_gridlines(extractor):
    from fuzz import image_variant, random_spec

    rng = np.random.default_rng(7)
    for _ in range(23):
        spec = random_spec(rng, kind_override="dot")
    assert spec.grid and len(spec.means) == 6
    opts = ExtractOptions(chart_type="dot", bar_names=False, titles=False)
    for image in (render(spec), image_variant(render(spec), 65, 0.75)):
        panels = extractor.extract(image, opts)
        assert len(panels) == 1 and len(panels[0].bars) == 6
        axis_range = _axis_range(panels[0])
        assert max(abs(b.mean - expected) / axis_range
                   for b, expected in zip(panels[0].bars, spec.means)) < 0.01


def test_gridline_fragment_at_frame_does_not_hide_dot_mean(extractor):
    from fuzz import random_spec

    rng = np.random.default_rng(7)
    for _ in range(14):
        spec = random_spec(rng, kind_override="dot")
    assert spec.grid and len(spec.means) == 4
    panels = extractor.extract(render(spec), ExtractOptions(chart_type="dot", bar_names=False, titles=False))
    assert len(panels) == 1 and len(panels[0].bars) == 4
    axis_range = _axis_range(panels[0])
    assert max(abs(b.mean - expected) / axis_range
               for b, expected in zip(panels[0].bars, spec.means)) < 0.01


def test_resized_decimal_axis_does_not_become_ten_times_too_large(extractor):
    from fuzz import image_variant, random_spec

    rng = np.random.default_rng(7)
    for _ in range(15):
        spec = random_spec(rng, kind_override="dot")
    image = image_variant(render(spec), 65, 0.75)
    panels = extractor.extract(image, ExtractOptions(chart_type="dot", bar_names=False, titles=False))
    assert len(panels) == 1 and len(panels[0].bars) == len(spec.means)
    p = panels[0]
    assert p.calibration_inliers >= 4
    assert max(abs(b.mean - expected) / _axis_range(p)
               for b, expected in zip(p.bars, spec.means)) < 0.01


def test_two_digit_decimal_ticks_survive_jpeg_ocr(extractor):
    from fuzz import image_variant, random_spec

    rng = np.random.default_rng(123)
    for _ in range(4):
        spec = random_spec(rng, kind_override="dot")
    image = image_variant(render(spec), 65, 0.75)
    panels = extractor.extract(image, ExtractOptions(chart_type="dot", bar_names=False, titles=False))
    assert len(panels) == 1 and len(panels[0].bars) == len(spec.means)
    panel = panels[0]
    assert max(abs(b.mean - expected) / _axis_range(panel)
               for b, expected in zip(panel.bars, spec.means)) < 0.01


@pytest.mark.parametrize("case_index", [28, 30])
def test_jpeg_dot_strokes_rejoin_across_overlaid_markers(extractor, case_index):
    from fuzz import image_variant, random_spec

    rng = np.random.default_rng(7)
    for _ in range(case_index + 1):
        spec = random_spec(rng, kind_override="dot")
    image = image_variant(render(spec), 65, 0.75)
    panels = extractor.extract(image, ExtractOptions(chart_type="dot", bar_names=False, titles=False))
    assert len(panels) == 1 and len(panels[0].bars) == len(spec.means)
    panel = panels[0]
    assert max(abs(b.mean - expected) / _axis_range(panel)
               for b, expected in zip(panel.bars, spec.means)) < 0.01


def test_rotated_ocr_does_not_override_a_coherent_decimal_axis(extractor):
    from fuzz import image_variant, random_spec

    rng = np.random.default_rng(7)
    for _ in range(31):
        spec = random_spec(rng, kind_override="bar")
    image = image_variant(render(spec), 65, 0.75)
    panels = extractor.extract(image, ExtractOptions(chart_type="bar", bar_names=False, titles=False))
    assert len(panels) == 1 and len(panels[0].bars) == len(spec.means)
    p = panels[0]
    assert p.calibration_inliers >= 4
    assert max(abs(b.mean - expected) / _axis_range(p)
               for b, expected in zip(p.bars, spec.means)) < 0.01


def test_colored_cap_is_separated_from_thick_whisker_stem(extractor):
    ink = np.zeros((100, 80), np.uint8)
    ink[54:58, 29:39] = 1  # cap, ten pixels wide
    ink[58:80, 32:36] = 1  # antialiased coloured stem, four pixels wide
    ink[80:83, 20:48] = 1  # bar outline
    info = extractor._cap_candidates(ink, 80, 20, 47, 3, 10)
    assert info["whisker_end"] == 54
    assert any(54 <= y <= 58 and width >= 9 for y, width, _ in info["bands"])


def test_short_real_bars_are_not_filtered_as_scatter_markers(extractor):
    spec = ChartSpec([0.12, 0.18], [0.015, 0.02], ylim=(0, 1.0))
    panels = extractor.extract(render(spec), FAST)
    assert len(panels) == 1 and panels[0].kind == "bar"
    assert len(panels[0].bars) == 2


def test_x_axis_without_tick_strokes_uses_aligned_labels():
    class LabelOCR:
        def boxes(self, image):
            return [((17, 4, 29, 16), "10", 0.99),
                    ((117, 4, 129, 16), "20", 0.99),
                    ((217, 4, 229, 16), "30", 0.99),
                    ((125, 23, 210, 34), "Time", 0.99)]

    extractor = FigureExtractor(ocr=LabelOCR())
    image = np.full((200, 350, 3), 255, dtype=np.uint8)
    panel = Panel(x1=20, y1=10, x2=300, y2=150, axis_x=20, axis_y=150,
                  plot_top=10, plot_right=300)
    extractor._calibrate_x(panel, image, np.zeros(image.shape[:2], np.uint8), 20, 300, 150, 1)
    assert len(panel.x_ticks) == 3
    assert panel.x_slope == pytest.approx(0.1)
    assert panel.x_value_at(143) == pytest.approx(20)


def test_thick_tick_marks_do_not_hide_line_axis():
    dark = np.zeros((200, 300), np.uint8)
    dark[20:151, 20:24] = 1
    for y in (20, 60, 100):
        dark[y:y + 3, 20:42] = 1
    dark[148:151, 20:250] = 1
    axes = FigureExtractor()._find_axes(dark, dark, max_thickness=25)
    assert any(abs(a["axis_left"] - 20) <= 3 and abs(a["axis_y"] - 149) <= 3 for a in axes)


def test_tall_numeric_x_labels_retry_after_tick_ocr_failure():
    class TallLabelOCR:
        calls = 0

        def line(self, image):
            self.calls += 1
            return ("2" if self.calls == 3 else "T"), 0.9

        def boxes(self, image):
            return [((17, 8, 27, 36), "0", 0.99),
                    ((117, 8, 127, 36), "1", 0.99),
                    ((217, 8, 227, 36), "2", 0.99)]

    extractor = FigureExtractor(ocr=TallLabelOCR())
    image = np.full((200, 250, 3), 255, np.uint8)
    dark = np.zeros(image.shape[:2], np.uint8)
    for x in (22, 122, 222):
        dark[102:110, x - 1:x + 2] = 1
        image[120:126, x - 2:x + 3] = 0
    panel = Panel(x1=20, y1=10, x2=240, y2=100, axis_x=20, axis_y=100,
                  plot_top=10, plot_right=240)
    extractor._calibrate_x(panel, image, dark, 20, 240, 100, 2)
    assert len(panel.x_ticks) == 3
    assert panel.x_value_at(142) == pytest.approx(1.0)


def test_inward_y_ticks_can_calibrate_from_numeric_label_boxes():
    class LabelOCR:
        def boxes(self, image):
            return [((18, 7, 30, 17), "20", 0.99),
                    ((18, 87, 30, 97), "10", 0.99),
                    ((18, 167, 30, 177), "0", 0.99),
                    ((0, 25, 17, 145), "Response", 0.99)]

    extractor = FigureExtractor(ocr=LabelOCR())
    image = np.full((220, 180, 3), 255, np.uint8)
    dark = np.zeros(image.shape[:2], np.uint8)
    dark[15:188, 35] = 1
    panel = Panel(x1=35, y1=15, x2=175, y2=187, axis_x=35, axis_y=187,
                  plot_top=15, plot_right=175)
    extractor._calibrate(panel, image, dark, 35, 15, 187, 1)
    assert panel.slope == pytest.approx(-0.125)
    assert len(panel.ticks) == 3


def test_numeric_label_rows_ignore_rotated_axis_title():
    image = np.full((420, 80, 3), 255, np.uint8)
    for y, label in zip((20, 96, 172, 248, 324, 400), ("50", "40", "30", "20", "10", "0")):
        cv2.putText(image, label, (19, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                    (0, 0, 0), 1, cv2.LINE_AA)
    for y in range(145, 300, 14):
        image[y:y + 7, 12:18] = 0  # letters of a vertical title to the left
    rows = FigureExtractor._label_rows_as_ticks(image, axis_left=35, top=15,
                                                 axis_row=405, line_w=1)
    assert len(rows) == 6


def test_dot_cap_at_image_edge_does_not_index_past_image(extractor):
    ink = np.zeros((40, 20), dtype=np.uint8)
    ink[10:21, 19] = 1
    ink[20, 15:20] = 1
    extractor._dot_cap(ink, 19, {"x1": 15, "x2": 19, "y0": 20, "yc": 20}, 1, 0)


def test_bar_names(extractor):
    spec = ChartSpec([10, 20, 15], [1, 2, 1], labels=["Control", "CUMS", "CUMS+EA"])
    p = extractor.extract(render(spec), ExtractOptions(bar_names=True, titles=True))[0]
    assert [b.label for b in p.bars] == ["Control", "CUMS", "CUMS+EA"]
    assert "Value" in p.y_title


def test_blank_and_photo_images(extractor):
    assert extractor.extract(np.full((400, 600, 3), 255, np.uint8)) == []
    noise = np.random.default_rng(1).integers(0, 255, (500, 700, 3), dtype=np.uint8)
    assert extractor.extract(noise) == []
    assert extractor.extract(np.zeros((0, 0, 3), np.uint8)) == []


def test_grayscale_and_alpha_inputs(extractor):
    img = render(CASES["outlined_white"])
    gray = img[:, :, 1].copy()
    bgra = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    assert len(extractor.extract(gray, FAST)) == 1
    assert len(extractor.extract(bgra, FAST)) == 1


@pytest.mark.parametrize("text,value", [
    ("100", 100), ("0.5", 0.5), ("-10", -10), ("−5", -5), ("2.5.", 2.5), ("50°", 50),
    ("1,000", 1000), ("。", 0), ("O", 0), ("Control", None), ("1 000", 1000.0), ("1 00", None), ("***", None),
    (".08", 0.08), ("-.04", -0.04),
])
def test_parse_number(text, value):
    assert parse_number(text) == value
