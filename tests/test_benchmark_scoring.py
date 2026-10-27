"""Guard the benchmark against counting a reused or incomplete line as success."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "eval"))

from chartinfo_eval import match_by_x  # noqa: E402
from fuzz import match_line_series  # noqa: E402
from line_plots import LinePoint, Series  # noqa: E402
from synth import LineSpec  # noqa: E402


def _series(values):
    s = Series(color=(0, 0, 0))
    s.points = [LinePoint(x_px=i, y_px=0, x=float(i), mean=float(y))
                for i, y in enumerate(values)]
    return s


def test_line_benchmark_matches_each_series_once():
    spec = LineSpec([0, 1], [[2, 3], [7, 8]], ylim=(0, 10))
    low, high = _series([2, 3]), _series([7, 8])
    assert match_line_series(spec, [high, low]) == [low, high]
    assert match_line_series(spec, [low]) is None
    assert match_line_series(spec, [low, high, high]) is None
    assert match_line_series(spec, [low, _series([7])]) is None


def test_real_chart_benchmark_cannot_reuse_one_detected_element():
    one_bar = SimpleNamespace(x1=10, x2=30)
    assert match_by_x([18, 22], [one_bar]) == [(0, 0)]
    assert match_by_x([18, 22], [one_bar, SimpleNamespace(x1=24, x2=45)]) == [(0, 0), (1, 1)]
    # The nearest-first greedy assignment would lose a valid second match.
    assert match_by_x([20, 12], [SimpleNamespace(x1=10, x2=25),
                                 SimpleNamespace(x1=20, x2=35)]) == [(0, 1), (1, 0)]
