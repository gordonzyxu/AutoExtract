"""
PDF helpers: pull figure images out of papers.

Journal PDFs usually embed each figure as one raster image, so we extract the
embedded image at its native resolution (sharper than rendering the page).
Pages whose figures are vector graphics are rendered at high DPI instead.
Each image is paired with its own caption, so a page holding two figures
reports two figure numbers.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


MIN_FIGURE_PIXELS = 600 * 400
MIN_SMALL_FIGURE_PIXELS = 400 * 300
CAPTION_RE = re.compile(r"\s*(?:Fig\.?|Figure)\s*(\d+)", re.IGNORECASE)


def _to_bgr(pix) -> np.ndarray:
    import pymupdf
    if pix.alpha or pix.n - pix.alpha != 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _captions(page) -> List[Tuple[float, str, str, float, float]]:
    """[(top y, figure number, caption text, left x, right x)] on the page."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        lines = block.get("lines", [])
        entries = [("".join(span["text"] for span in line["spans"]), line["bbox"])
                   for line in lines]
        starts = [(j, CAPTION_RE.match(entry[0])) for j, entry in enumerate(entries)]
        starts = [(j, m) for j, m in starts if m]
        for k, (j, match) in enumerate(starts):
            stop = starts[k + 1][0] if k + 1 < len(starts) else len(entries)
            part = entries[j:stop]
            text = "\n".join(row[0] for row in part)
            out.append((part[0][1][1], match.group(1), text,
                        min(row[1][0] for row in part), max(row[1][2] for row in part)))
    return sorted(out, key=lambda caption: (caption[0], caption[3]))


def _caption_for(rect, captions, prefer_above=False) -> Tuple[str, str]:
    """Match a caption to an image in a page's above- or below-caption layout."""
    if not captions:
        return "", ""
    if rect is not None:
        # Two-column pages can place captions at the same y. Prefer a caption
        # horizontally under/above this image before applying vertical order.
        aligned = [c for c in captions if c[4] >= rect.x0 - 5 and c[3] <= rect.x1 + 5]
        if aligned:
            captions = aligned
        if prefer_above:
            above = [c for c in captions if c[0] <= rect.y0 + 5]
            if above:
                return above[-1][1], above[-1][2]
        below = [c for c in captions if c[0] >= rect.y1 - 5]
        if below:
            return below[0][1], below[0][2]
        above = [c for c in captions if c[0] <= rect.y0 + 5]
        if above:
            return above[-1][1], above[-1][2]
        return "", ""
    return captions[0][1], captions[0][2]


def figure_number(pdf_path: str, page_index: int) -> str:
    """The first figure number captioned on the page ("Fig. 7." -> "7"), or ''."""
    import pymupdf
    with pymupdf.open(pdf_path) as doc:
        caps = _captions(doc[page_index])
        return caps[0][1] if caps else ""


def error_type_from_caption(caption: str) -> str:
    """'sd' or 'sem' if the caption says which the error bars show, else ''."""
    t = caption.replace("\n", " ")
    sem = re.search(r"\bS\.?\s?E\.?\s?M\.?\b|standard error", t, re.IGNORECASE)
    sd = re.search(r"(?:±|\+/-|\bmean\s*\(?)\s*S\.?D\.?\b|standard deviation", t, re.IGNORECASE)
    if sem and not sd:
        return "sem"
    if sd and not sem:
        return "sd"
    return ""


def _has_sparse_vector_axes(drawings) -> bool:
    """Recognize long perpendicular strokes in a simple vector chart."""
    horizontal = vertical = False
    for drawing in drawings:
        r = drawing["rect"]
        horizontal |= r.width >= 100 and r.height <= 4
        vertical |= r.height >= 100 and r.width <= 4
        for item in drawing.get("items", ()):
            if item[0] != "l":
                continue
            a, b = item[1:3]
            horizontal |= abs(a.x - b.x) >= 100 and abs(a.y - b.y) <= 4
            vertical |= abs(a.y - b.y) >= 100 and abs(a.x - b.x) <= 4
        if horizontal and vertical:
            return True
    return False


def figures(pdf_path: str, pages: Optional[Iterable[int]] = None,
            render_dpi: int = 300) -> List[dict]:
    """
    Every figure-sized image in the PDF:
    [{page, figure, caption, error_type, image}] with image as BGR.
    """
    import pymupdf
    out = []
    with pymupdf.open(pdf_path) as doc:
        for i in (pages if pages is not None else range(len(doc))):
            page = doc[i]
            caps = _captions(page)
            found = False
            placements = []
            seen_xrefs = set()
            for info in page.get_images(full=True):
                xref, w, h = info[0], info[2], info[3]
                if xref in seen_xrefs or w * h < MIN_SMALL_FIGURE_PIXELS:
                    continue
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []
                # A smaller chart can still occupy a full figure in a paper.
                # Require a substantial page footprint before accepting one,
                # so small logos and thumbnails do not become figures.
                if w * h < MIN_FIGURE_PIXELS:
                    rects = [r for r in rects if r.width >= 150 and r.height >= 100]
                    if not rects:
                        continue
                seen_xrefs.add(xref)
                placements.append((xref, rects))
            rects_all = [r for _, rects in placements for r in rects]
            prefer_above = bool(caps and rects_all and caps[0][0] < min(r.y0 for r in rects_all))
            for xref, rects in placements:
                try:
                    img = _to_bgr(pymupdf.Pixmap(doc, xref))
                except Exception:
                    continue
                for rect in rects or [None]:
                    fig, caption = _caption_for(rect, caps, prefer_above)
                    out.append(dict(page=i, figure=fig, caption=caption,
                                    error_type=error_type_from_caption(caption), image=img))
                    found = True
            # A simple vector bar plot can have only a few dozen drawing paths,
            # but so do title pages and ruled text. Rendering a page at 300 dpi
            # and scanning it is costly, so a page with fewer paths than a
            # typical vector figure must also show a long axis-like pair.
            drawings = page.get_drawings() if not found else []
            if not found and (len(drawings) > 200 or ((len(drawings) >= 20 or caps)
                                                      and _has_sparse_vector_axes(drawings))):
                fig, caption = _caption_for(None, caps)
                out.append(dict(page=i, figure=fig, caption=caption,
                                error_type=error_type_from_caption(caption),
                                image=_to_bgr(page.get_pixmap(dpi=render_dpi))))
    return out


def page_images(pdf_path: str, pages: Optional[Iterable[int]] = None,
                render_dpi: int = 300) -> List[Tuple[int, np.ndarray]]:
    """[(page_index, BGR image)] for every figure-sized image in the PDF."""
    return [(f["page"], f["image"]) for f in figures(pdf_path, pages, render_dpi)]
