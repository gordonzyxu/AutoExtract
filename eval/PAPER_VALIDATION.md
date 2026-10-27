# Validation against the user's meta-analysis papers

The source directory contains 21 active paper folders, 21 local PDFs (281
pages), and 98 `Long_format_*.xlsx` workbooks. The `unused` folder is excluded.
No PDF or workbook is copied into this repository. These files are reference
material only; the benchmark does not edit them.

## Paired numerical benchmark

`paper_gold.json` records panel and group mappings checked against the PDFs
*before* looking at extraction output. The figure, x position, panel axis,
group order, and workbook row identify each comparison. `paper_validation.py`
reads the hand-extracted values directly from the Excel sheets and the figure
image from its local PDF. This avoids choosing whichever prediction is nearest
to the reference value. It is still a small, selected development set: papers
1–7 informed debugging; paper 8 was mapped before the revised code was run.
The author's WebPlotDigitizer values are comparison measurements, not the
unobservable underlying experiment data.

The score uses absolute error below 1% of the chart's vertical axis span.
An unmatched group fails. Error-bar magnitudes are scored only when the
detector marks the cap as confirmed; flagged caps remain in the output for
manual review and count as unverified. Paper 4 displays SD and uses the `sd`
column; the mapped bar figures display SEM, confirmed by captions or methods.
The duplicated `treatment_group` labels in paper 8 workbook rows 6–7 do not
match the source figure's EA1/EA2 labels, so the numeric rows are mapped by
the figure's left-to-right order. This source-data labeling issue is not
corrected here.

| Metric | Last commit (`2263ca6`) | Current (October 26) |
| --- | ---: | ---: |
| Mapped bar means within 1% | 91/92 | 91/92 |
| Confirmed bar error magnitudes within 1% | 81/92 | 85/92 |
| Mapped week-2 line means within 1% (paper 4) | 0/4 | 2/4 |
| Line error bars confirmed without review (paper 4) | 0/4 | 0/4 |
| False bar panels in paper 5 Fig. 4 and paper 7 Fig. 7 | 2 | 0 |

Remaining misses:

- **One bar mean** (Lin 2023, sucrose preference) is 1.07% of the axis span
  off, just over the threshold.
- **Seven bar error bars are flagged for review.** Six were measured within
  1% anyway (four within 0.4%); only one, in Zhang 2023 Fig. 1, is genuinely
  off (2.4%). In Jung 2021 Fig. 2 the three flags come from significance marks
  (`*`, `#`, `%`) printed directly above the cap, which trip the check that
  rejects scatter-dot rings. None of the 85 confirmed error bars exceeds 1%.
- **Paper 4's line figure** has two same-colour series that overlap; they are
  not separated, so two of the four means are missed and all four error bars
  are flagged.

How the earlier rounds got here:

1. **Coloured error caps:** separate a cap from a thick anti-aliased whisker
   using the local stem width (recovered a Jung 2021 cap); a suspicious
   near-zero cap in Zhang 2023 is flagged instead of accepted.
2. **Line axes:** retry wider vertical-axis components only when the ordinary
   detector finds no line panels, and read numeric x labels when tick marks
   are missing. Kawanokuchi 2021 Fig. 4 went from zero panels to two.
3. **False panels:** reject two tiny pseudo-bars at the base of a dot plot and
   overlapping bar footprints inside a movement-trace image.
4. **PDF intake:** a reused embedded image gets one figure per placement,
   captions above or beside figures are matched, wrapped captions keep their
   SD/SEM statement, and paper 7 pins its Figure 7 to PDF page index 8 (the
   caption matches two images).

## All-paper operational screen

`paper_corpus_scan.py` visited all 21 PDFs and screened **one caption-selected
figure per paper** in bar mode, downscaled to at most 1000 pixels. It returned
70 candidate panels across those 21 images, with four selected figures giving
no panels. The selection can choose a non-chart or the wrong outcome; the
candidate panels are **not** verified true positives or a recall score. The
screen took 161.4 seconds of extraction time in total (median 4.07 seconds
per analyzed image, maximum 42 seconds), excluding PDF decoding. Native
mapped-figure timings and these resized screening timings must not be mixed.

The saved operational records are in ignored `eval/out/paper_corpus_one_bar_v2.jsonl`.
They are deliberately separate from the source-linked paired benchmark.

## Reproduce

From the AutoExtract repository in PowerShell, set `$paperRoot` to the user's
`mice studies` folder and run:

```powershell
.venv\Scripts\python.exe eval/paper_validation.py --paper-root $paperRoot --backend backend --output eval/out/paper_gold_current.json
.venv\Scripts\python.exe eval/paper_validation.py --paper-root $paperRoot --backend eval/out/baseline/backend --output eval/out/paper_gold_original.json
.venv\Scripts\python.exe eval/paper_corpus_scan.py --paper-root $paperRoot --one-figure-per-paper --max-dimension 1000 --output eval/out/paper_corpus_one_bar_v2.jsonl
```

The original-backend path is a locally saved snapshot and is not committed.
Reconstruct it from the same Git baseline before claiming paired historical
results on another machine. Violin and box accuracy cannot be inferred from
this selected comparison set; it contains mapped bars and one line figure.

## Rechecks

The eight mapped figures were rerun after every round of changes (October 24,
25 and 26). The scores in the table did not change after the first round: the
later rounds improved synthetic dot, box, violin and line charts and PDF
intake, but did not move these bar and line values. The October 26 round's
PDF page filter skips 34 figure-less text pages across the 21 PDFs; none of
them produced a panel in any chart mode.
