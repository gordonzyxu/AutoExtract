"""Paired extraction-only timing of identical-crop OCR caching on a mapped PDF figure.

The PDF is decoded before timing. Cache entries are cleared before each figure;
the uncached condition clears them before every OCR call. Both use the same
extractor code and must return identical panel JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from figure_extractor import ExtractOptions, FigureExtractor  # noqa: E402
from pdf_utils import figures  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=pathlib.Path,
                        default=os.environ.get("AUTOEXTRACT_PAPER_ROOT"))
    parser.add_argument("--paper-number", type=int, default=1)
    parser.add_argument("--runs", type=int, default=7, help="timed pairs after one warmup pair")
    args = parser.parse_args()
    if not args.paper_root or args.runs < 1:
        parser.error("--paper-root and a positive --runs are required")

    cases = json.loads((ROOT / "eval" / "paper_gold.json").read_text(encoding="utf-8"))["cases"]
    case = next(c for c in cases if c["paper_folder"].split()[0] == f"(paper{args.paper_number})")
    pdf = next((args.paper_root / case["paper_folder"]).glob("*.pdf"))
    image = next(f["image"] for f in figures(str(pdf)) if f["figure"] == case["figure"])
    opts = ExtractOptions(chart_type=case["chart_type"], bar_names=False, titles=False)

    cached, uncached = FigureExtractor(), FigureExtractor()
    original_line = uncached.ocr.line

    def no_cache(crop):
        uncached.ocr._line_cache.clear()
        return original_line(crop)

    uncached.ocr.line = no_cache
    times = {"cached": [], "uncached": []}
    hits = []
    for repeat in range(args.runs + 1):
        order = (("uncached", uncached), ("cached", cached)) if repeat % 2 == 0 else (
            ("cached", cached), ("uncached", uncached))
        outputs = {}
        for name, extractor in order:
            extractor.ocr._line_cache.clear()
            before = extractor.ocr.line_cache_hits
            start = time.perf_counter()
            panels = extractor.extract(image, opts)
            elapsed = time.perf_counter() - start
            outputs[name] = [p.to_dict() for p in panels]
            if repeat:
                times[name].append(elapsed)
                if name == "cached":
                    hits.append(extractor.ocr.line_cache_hits - before)
        if outputs["cached"] != outputs["uncached"]:
            raise AssertionError(f"OCR cache changed extraction output on pair {repeat}")

    print(f"{case['paper_folder']} Figure {case['figure']}: {len(outputs['cached'])} panels; "
          f"{args.runs} timed paired runs, identical outputs")
    for name in ("uncached", "cached"):
        print(f"{name}: median {statistics.median(times[name]):.3f}s; "
              f"runs {[round(v, 3) for v in times[name]]}")
    print(f"cached OCR hits per figure: {hits}")


if __name__ == "__main__":
    main()
