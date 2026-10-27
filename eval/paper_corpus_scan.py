"""Resumable operational scan of the meta-analysis PDFs (not an accuracy score).

Each captioned raster figure is extracted through the regular PDF path and
measured at a bounded pixel size for screening. Use paper_validation.py and
manually mapped groups for accuracy. The source directory is read-only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

import cv2


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from figure_extractor import ExtractOptions, FigureExtractor  # noqa: E402
from pdf_utils import figures  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-root", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, default=ROOT / "eval/out/paper_corpus_bar_scan.jsonl")
    ap.add_argument("--type", choices=("bar", "dot", "box", "violin", "line"), default="bar")
    ap.add_argument("--max-dimension", type=int, default=1600)
    ap.add_argument("--one-figure-per-paper", action="store_true",
                    help="Screen one captioned behavioral figure from each PDF")
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["paper"], rec["page"], rec["figure"], rec["candidate"], rec["chart_type"]))
    papers = sorted((p for p in args.paper_root.iterdir()
                     if p.is_dir() and p.name.startswith("(paper") and "unused" not in p.parts),
                    key=lambda p: int(p.name.split(")")[0][6:]))
    extractor = FigureExtractor()
    options = ExtractOptions(chart_type=args.type, bar_names=False, titles=False)
    totals = dict(figures=0, panels=0, papers=0)
    for paper in papers:
        pdfs = list(paper.glob("*.pdf"))
        if len(pdfs) != 1:
            print(f"SKIP {paper.name}: {len(pdfs)} PDFs", flush=True)
            continue
        count, panels_found = 0, 0
        candidates = [(i, f) for i, f in enumerate(figures(str(pdfs[0]))) if f["figure"]]
        if args.one_figure_per_paper and candidates:
            terms = ("sucrose", "immobil", "distance", "zone transition", "struggl",
                     "open.field", "forced.swim", "tail.suspension", "depress",
                     "behav", "locomotor", "oft", "fst", "spt", "tst")
            def rank(entry):
                i, f = entry
                caption = f["caption"].lower().replace("\n", " ")
                score = sum(bool(re.search(term, caption)) for term in terms)
                score -= 3 * bool(re.search(r"experimental (?:schedule|design)|schematic diagram", caption))
                return (score, -i)
            candidates = [max(candidates, key=rank)]
        for candidate, fig in candidates:
            if not fig["figure"]:
                continue
            key = (paper.name, fig["page"], fig["figure"], candidate, args.type)
            if key in done:
                continue
            source_shape = fig["image"].shape[:2]
            scale = min(1.0, args.max_dimension / max(source_shape))
            image = cv2.resize(fig["image"], None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA) if scale < 1 else fig["image"]
            t0 = time.perf_counter()
            try:
                panels = extractor.extract(image, options)
                error = ""
            except Exception as exc:
                panels, error = [], f"{type(exc).__name__}: {exc}"
            rec = dict(paper=paper.name, source=pdfs[0].name, page=fig["page"],
                       figure=fig["figure"], candidate=candidate, chart_type=args.type,
                       source_shape=source_shape, analyzed_shape=image.shape[:2],
                       seconds=round(time.perf_counter() - t0, 3), error=error,
                       panels=[dict(kind=p.kind, items=len(p.items), axis_x=p.axis_x,
                                    axis_y=p.axis_y, warnings=p.warnings) for p in panels])
            with args.output.open("a", encoding="utf-8") as out:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
            panels_found += len(panels)
        totals["figures"] += count
        totals["panels"] += panels_found
        totals["papers"] += 1
        print(f"{paper.name}: {count} figures, {panels_found} {args.type} panels", flush=True)
    print(totals, flush=True)


if __name__ == "__main__":
    main()
