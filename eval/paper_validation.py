"""Compare explicitly mapped PDF groups with the author's WebPlotDigitizer tables.

The mappings in paper_gold.json are fixed before extraction. Unmapped PDF
charts must not be counted as either successes or failures in these metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import openpyxl


ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER_ROOT = os.environ.get("AUTOEXTRACT_PAPER_ROOT")


def workbook_rows(path: pathlib.Path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(wb.active.values)
    finally:
        wb.close()
    return [dict(zip(rows[0], row)) for row in rows[1:]]


def run(backend: pathlib.Path, paper_root: pathlib.Path, cases: list[dict]):
    sys.path.insert(0, str(backend.resolve()))
    from figure_extractor import ExtractOptions, FigureExtractor
    from pdf_utils import figures

    extractor = FigureExtractor()
    results = []
    for case in cases:
        folder = paper_root / case["paper_folder"]
        pdfs = list(folder.glob("*.pdf"))
        if len(pdfs) != 1:
            raise ValueError(f"Expected one PDF in {folder}, got {len(pdfs)}")
        matches = [f for f in figures(str(pdfs[0]), pages=[case["page"]] if "page" in case else None)
                   if f["figure"] == case["figure"]
                   and ("page" not in case or f["page"] == case["page"])]
        if len(matches) != 1:
            raise ValueError(f"Expected one Figure {case['figure']} in {pdfs[0]}, got {len(matches)}")
        fig = matches[0]
        start = time.perf_counter()
        panels = extractor.extract(fig["image"], ExtractOptions(
            chart_type=case["chart_type"], bar_names=False, titles=False))
        duration = time.perf_counter() - start
        for mapping in case["maps"]:
            rows = workbook_rows(folder / mapping["workbook"])
            pi = mapping["panel"]
            if "axis_anchor" in mapping:
                ax, ay = mapping["axis_anchor"]
                nearby = [(i, p) for i, p in enumerate(panels)
                          if abs(p.axis_x - ax) < 0.06 * fig["image"].shape[1]
                          and abs(p.axis_y - ay) < 0.06 * fig["image"].shape[0]]
                panel = min(nearby, key=lambda ip: abs(ip[1].axis_x - ax) + abs(ip[1].axis_y - ay))[1] if nearby else None
            else:
                panel = panels[pi] if pi < len(panels) else None
            axis_range = abs(panel.value_at(panel.plot_top) - panel.value_at(panel.axis_y)) if panel else None
            indices = mapping.get("bar_indices", mapping.get("series_indices"))
            if len(indices) != len(mapping["excel_rows"]):
                raise ValueError(f"Mapping length differs in {folder}: {mapping['caption_group']}")
            for excel_row, item_index in zip(mapping["excel_rows"], indices):
                ref = rows[excel_row - 2]
                item = None
                if case["chart_type"] == "bar" and panel and item_index is not None and item_index < len(panel.bars):
                    item = panel.bars[item_index]
                elif case["chart_type"] == "line" and panel and item_index is not None and item_index < len(panel.series):
                    points = [p for p in panel.series[item_index].points
                              if p.x is not None and abs(p.x - mapping["point_x"]) < 0.15]
                    item = min(points, key=lambda p: abs(p.x - mapping["point_x"])) if points else None
                mean = float(ref["mean"])
                error_column = mapping.get("error_column", "sem")
                hand_error = float(ref[error_column]) if ref.get(error_column) is not None else None
                pred_mean = float(item.mean) if item and item.mean is not None else None
                pred_error = (abs(float(item.mean_plus_err - item.mean)) if item and item.mean_plus_err is not None
                              and item.mean is not None else None)
                results.append({
                    "paper": case["paper_folder"], "figure": case["figure"],
                    "chart_type": case["chart_type"],
                    "panel": pi, "caption_group": mapping["caption_group"],
                    "workbook": mapping["workbook"], "excel_row": excel_row,
                    "item_index": item_index, "axis_range": axis_range,
                    "hand_mean": mean, "pred_mean": pred_mean,
                    "hand_error": hand_error, "pred_error": pred_error, "error_column": error_column,
                    "uncertain_cap": item.uncertain if item else None,
                    "mean_error_pct_axis": abs(pred_mean - mean) / axis_range * 100 if pred_mean is not None else None,
                    "error_pct_axis": abs(pred_error - hand_error) / axis_range * 100
                    if pred_error is not None and hand_error is not None else None,
                })
        print(f"{case['paper_folder']} Figure {case['figure']}: "
              f"{len(panels)} panels in {duration:.1f}s (extraction only)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="backend")
    ap.add_argument("--paper-root", type=pathlib.Path, default=pathlib.Path(PAPER_ROOT) if PAPER_ROOT else None,
                    required=PAPER_ROOT is None, help="Path to the meta-analysis mice studies folder")
    ap.add_argument("--papers", help="Comma-separated paper numbers, for example 2,3")
    ap.add_argument("--output", type=pathlib.Path)
    args = ap.parse_args()
    cases = json.loads((ROOT / "eval" / "paper_gold.json").read_text(encoding="utf-8"))["cases"]
    if args.papers:
        wanted = {f"(paper{int(n)})" for n in args.papers.split(",")}
        cases = [c for c in cases if c["paper_folder"].split()[0] in wanted]
    rows = run(ROOT / args.backend, args.paper_root, cases)
    for kind in sorted({r["chart_type"] for r in rows}):
        group = [r for r in rows if r["chart_type"] == kind]
        mean_errors = [r["mean_error_pct_axis"] for r in group if r["mean_error_pct_axis"] is not None]
        error_errors = [r["error_pct_axis"] for r in group if r["error_pct_axis"] is not None
                        and not r["uncertain_cap"]]
        print(f"{kind}: means {sum(e < 1 for e in mean_errors)}/{len(group)} within 1% axis; "
              f"matched {len(mean_errors)}/{len(group)}; "
              f"median matched error {np.median(mean_errors):.3f}% axis" if mean_errors
              else f"{kind}: no matched means / {len(group)} expected")
        print(f"{kind}: unflagged error magnitudes {sum(e < 1 for e in error_errors)}/{len(group)} "
              f"within 1% axis; matched {len(error_errors)}/{len(group)}" if error_errors
              else f"{kind}: no unflagged error caps / {len(group)} expected")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
