# Chart extraction benchmark protocol

This project has three distinct accuracy checks. Do not combine their percentages:

1. `tests/fuzz.py` renders Matplotlib charts with known values. It is a useful regression test but shares styles and fonts with the extractor's development fixtures.
2. `eval/chartinfo_eval.py` uses the independent ICPR 2022 CHART-Info UB-Unitec PMC test set of real paper figures. Download it from https://chartinfo.github.io/toolsanddata.html and keep it outside the repository. The archive's license is CC BY-NC-SA 4.0. Its annotated test split contains 280 vertical bars, 88 vertical boxes, and 312 lines; it has no violin or mean-and-whisker dot category.
3. API tests pass PNG and raster-embedded PDF figures through the upload route. A separate test checks a simple vector PDF. These test the input pipeline on a few known charts, not population-level accuracy.

## What counts as success

- For synthetic charts, a value is correct when its absolute error is below 1% of the known vertical axis range. Line error-bar lengths use 1.5%. Box and violin chart passes require every applicable statistic and the expected number of groups. The bar/dot chart pass checks group count and means; the script separately reports unflagged cap errors and how many caps were flagged for review.
- Synthetic line scoring requires every expected series and point, no extra series or points, a one-to-one series match, and both x and y values within 1% of their axis ranges. The script also counts charts for which every unflagged error cap is present. Flagged caps require human review; their absence is not proof of correctness.
- CHART-Info matching is one-to-one in horizontal pixel position for bars and boxes and one-to-one by traced series for lines. Report **detection**, **matched ground-truth elements**, **accuracy conditional on a match**, and **accuracy across all annotated elements**. An unmatched element fails the last measure. Stacked bars are excluded because the extractor does not support them.
- CHART-Info line annotations are sampled positions along the drawn curves. The line score checks trace recovery and y calibration; it does not directly score the extractor's discrete markers or error bars. Keep it separate from the synthetic line-point score.
- Benchmark timings from `tests/fuzz.py` include chart rendering and can be distorted by other processes. Measure extraction-only latency separately, warmed up, with repeated runs and p50/p95 before claiming a speed change.

## Verified results, October 23-24, 2026

Fixed seed 7, 50 synthetic charts per type, current worktree:

| Type | Chart passes | Scope |
| --- | ---: | --- |
| Bar only | 50/50 | Group count and means; cap quality separately reported |
| Dot only | 44/50 | Group count, type, and means; cap quality separately reported |
| Box | 49/50 | All five box statistics |
| Violin | 34/50 | Visible ends, median, and displayed quartiles; invented quartiles fail |
| Line | 31/50 | Complete one-to-one series, x/y values, and accuracy of caps that are present and unflagged; cap completeness is separate |

The original checkout scored dot 43/50, box 46/50, violin **32/50 under the current false-quartile rule**, and line 24/50 using the same synthetic cases and stricter line scorer. The earlier mixed bar/dot score of 50/50 should not be presented as a dot-specific score. An earlier line score of 32/50 allowed reuse of one extracted series and did not check x positions.

CHART-Info full test set, current worktree:

| Type | Chart detection | Matched elements | Within 1% among matched | Within 1% of all ground truth |
| --- | ---: | ---: | ---: | ---: |
| Non-stacked vertical bar | 193/262 | 1807/3834 | 86% values | 41% values |
| Vertical box | 71/88 | 410/629 | Q1 92%, median 88%, Q3 92% | Q1 60%, median 57%, Q3 60% |
| Line | 170/312 | 11602/29688 sampled trace positions | 64% y values | 25% y values |

Before the axis-recovery refinement below, the box comparison to the original checkout matched the same 376/629 elements and found Q1 91%, median 88%, and Q3 90% within 1% among matched boxes. On the first 50 real line charts, the original checkout detected 22/50 charts and matched 1190/4151 points; 44% of those matched points were within 1%. The previous refinement detected 27/50 and matched 1762/4151, with 63% of matched points within 1%.

Downsampling to 75% of each dimension and then re-encoding at JPEG quality 65 gives a first robustness check. Bar and dot use 50 charts with seed 7; box, violin, and line use 30 charts with seed 123. These are synthetic transformations, not a representative sample of actual screenshots.

| Type | Native render | Resized JPEG | Pass criterion |
| --- | ---: | ---: | --- |
| Bar | 50/50 | 46/50 | Group count and means |
| Dot | 44/50 | 43/50 | Type, group count, and means |
| Box | 30/30 | 28/30 | Five box statistics |
| Violin | 27/30 | 23/30 | Applicable displayed statistics; false quartiles fail |
| Line | 22/30 | 10/30 | One-to-one series, x/y values, and accuracy of measured unflagged caps |

For line charts in this 30-chart sample, only 4/30 native and 2/30 resized JPEG charts had all unflagged error caps present as well as correct values. Missing unflagged caps and catastrophic axis-calibration errors are concrete risks exposed by these tests.
On the separate native seed-7 set, only 7/50 line charts had correct values
and every unflagged cap present; 148 unflagged caps remained missing. Cap
coverage is therefore a larger unresolved problem than the line value score.

Run the synthetic checks:

```powershell
.venv\Scripts\python.exe tests/fuzz.py 50 --seed 7 --bar-only
.venv\Scripts\python.exe tests/fuzz.py 50 --seed 7 --dot
.venv\Scripts\python.exe tests/fuzz.py 50 --seed 7 --box
.venv\Scripts\python.exe tests/fuzz.py 50 --seed 7 --violin
.venv\Scripts\python.exe tests/fuzz.py 50 --seed 7 --line
```

Use `--jpeg-quality 65 --resize-factor 0.75` to test a downsampled, compressed image of the same generated chart. The PDF tests run with `python -m pytest tests/test_api.py`.

Run the real check, replacing `$chartInfoRoot` with the unzipped test-set directory:

```powershell
.venv\Scripts\python.exe eval/chartinfo_eval.py --data $chartInfoRoot --type box
.venv\Scripts\python.exe eval/chartinfo_eval.py --data $chartInfoRoot --type bar
.venv\Scripts\python.exe eval/chartinfo_eval.py --data $chartInfoRoot --type line
```

## Next benchmark additions

1. **Real dot and violin gold set:** sample licensed paper figures, record the visible mean/median/quartiles and endpoint semantics, have two people digitize each independently, adjudicate disagreement, and freeze a held-out split. CHART-Info has no suitable category for these two types.
2. **Cross-renderer synthetic set:** make the same numerical cases with Matplotlib, Seaborn, ggplot2, and GraphPad Prism. Include Tukey versus min/max box whiskers and Seaborn violins with `cut=0` and `cut>0`; the density tails and raw sample endpoints differ.
3. **Format and degradation grid:** score native PNG, JPEG quality levels, resized screenshots, raster PDFs, and vector PDFs separately at matched visual resolution. Record calibration failures as their own category.
4. **Failure strata:** report low DPI, dark fill, white fill, overlaid individual dots, multiple series, intersections, log axes, grids, and mixed-panel figures separately. Keep samples and seeds fixed while revising code.
5. **Operational metrics:** report false panels on other chart types, cap/whisker coverage, export completeness, extraction-only p50/p95 latency, and memory use. A conditional value error alone hides missed charts.

Other candidate datasets: [ChartQA](https://github.com/vis-nlp/ChartQA) provides bar/line image annotations and tables, [PlotQA](https://iitmnlp.github.io/PlotQA/) provides scientific-style synthetic plots with underlying data, and [ExChart-Bench](https://exchart.github.io/) combines real and synthetic bar/line/scatter examples. Their chart types and scoring tasks differ from this project's box/violin/meta-analysis output, so they need adapters and should be reported separately.

## Second three-cycle refinement (October 24, 2026)

The same fixed images and scorers were used for each paired comparison. The
original checkout was kept under `eval/out/baseline/backend` for repeatable
tests. The historical 97% figure in the README is from a different 200-chart
synthetic bar run, so it is not a before score for these cases.

1. **Axis OCR and compressed images.** Parsing `.08` as `0.8` caused tenfold
   errors; preserve the leading decimal point. At seed 7, resized to 75% and
   JPEG quality 65, bar-only increased 43/50 to 46/50 and dot 34/50 to 42/50.
   At seed 123, degraded box increased 26/30 to 28/30. Native bar-only was
   50/50 in the previous round; original-checkout bar-only was not measured.
2. **Real line plots and error caps.** When a framed x-axis has printed labels
   but no tick strokes, detect aligned numeric labels in a narrow strip. A
   marker-width cap-search adjustment reduced missing unflagged caps on the
   30 native seed-123 charts from 111 to 100, but whole-chart line accuracy
   stayed 22/30 native and 10/30 degraded. On the paired first 50 real
   CHART-Info line charts, chart detection rose from 22 to 27, matched trace
   positions from 1190/4151 to 1762/4151, and positions within 1% of the
   axis range from 13% to 27% of all annotated positions. The matched-only
   rate rose from 44% to 63%. This trace metric does not score error caps.
   Across the **full** 312-chart real line set, detections rose from 148 to
   163, matched trace positions from 8864/29688 to 10941/29688, and the
   within-1% rate across all annotated positions from approximately 17% to
   24% (58% to 64% among matches).
3. **Violin false quartiles and box regression.** Require continuous original
   side strokes for a violin's narrow inner box and withhold a lone quartile
   result inconsistent with three or more median-only violins. The violin
   scorer now fails charts that invent quartiles absent from median-only
   plots. Under this stricter scorer, original checkout versus current is
   32/50 versus 34/50 native seed 7. The resized JPEG seed-123 score is
   23/30; 22/30 under the earlier, looser scorer is not directly comparable.
   For ordinary boxes, a wide edge scan had reduced real Q1 and median
   accuracy. Preferring the local edge and widening only when it is missing
   retained 49/50 synthetic seed-7 boxes and restored full real matched-box
   Q1 and median rates to 91% and 88%; Q3 is 91% versus 90% in the original
   checkout. Chart detection and matching stayed 65/88 and 376/629.

The earlier full real non-stacked bar evaluation detected 185/262 charts and
matched 1714/3834 bars; 87% of matched values and 39% of all annotated values
fall within 1% of the axis range. This is still far from reliable unattended
scientific extraction. Box quartiles and medians are accurate *conditional on
matching*, while 219 of 629 ground-truth boxes remain unmatched after the
axis-recovery refinement below.

An extraction-only timing probe (`eval/latency_eval.py`) rendered five fixed
charts per type before timing, warmed the extractor, then timed two passes.
Four runs of the original and three of the revision gave the following ranges
of per-run medians (10 timed images each):

| Type | Original p50 range | Revised p50 range |
| --- | ---: | ---: |
| Bar | 187-383 ms | 183-427 ms |
| Box | 111-233 ms | 97-178 ms |
| Violin | 127-249 ms | 153-228 ms |
| Line | 393-849 ms | 421-761 ms |

Host load and cache state varied enough that these short runs do **not**
establish a reliable speed gain or regression. This is a measurement limit,
not a speed claim. The label OCR fallback only runs when the cheap tick-stroke
detection finds fewer than two x ticks; its cost on real no-tick charts needs
a dedicated timing study.

The unresolved errors include compressed violin axes whose decimal points
disappear entirely, overlaid dots hiding inner lines, low-resolution inner
boxes, intersecting same-colour lines, and missing error caps. A future real
violin/dot gold set should record whether tails are data extrema or density
extensions. Do not infer min/max from violin ends without that semantic check.

## Dedicated dot-plot refinement (October 24, 2026)

The Dot Plot Analyzer now has its own page, API `chart_type="dot"`, and CLI
`--type dot`. It returns only dot panels; the previous Bar Analyzer behavior
is retained for existing workflows. The same fixed seed-7 dot images scored
43/50 through the old bar mode and **44/50** through the new dot mode. The
low-DPI gain came from treating two adjacent fragments of a mean stroke as
one line; distinct box edges still require meaningful vertical separation.
Among reported groups, mean error was 0.072% median and 0.279% at p95 of
the axis range. The compressed 75%-size JPEG score is 43/50, versus 42/50
before this refinement. Its worst detected mean error remains very large
(90% of axis range), so these plots need review rather than unattended use.
Bar-only native charts remain 50/50. A relaxed width-threshold experiment
gave no dot gain (43/50) and was reverted. API tests exercise PDF dot
extraction, and a regression test reproduces the split low-DPI mean line.
## Axis-recovery refinement (October 24, 2026)

A failure audit of the real box charts found two recurring, readable layouts
that the geometric axis detector had missed: figures with a vertical y-axis
but no horizontal x-axis stroke, and figures with inward-pointing tick marks
plus a rotated y-axis title. The [CHART-Info task design](https://chartinfo.github.io/tasks_2022.html)
separates text recognition, axis analysis, and data extraction; this suggests
using the numeric label column as a guarded fallback when the stroke geometry
is incomplete. We added that fallback without changing the ordinary path.

On the full 88-chart real box set, detection improved from 65 to 71 charts and
one-to-one matched boxes from 376 to 410. Q1, median, and Q3 within 1% across
*all* ground-truth boxes rose from 55%, 52%, 55% to 60%, 57%, 60%.
Matched-only Q1/median/Q3 accuracy is 92%, 88%, 92%. The fixed 50-chart
synthetic box score remains 49/50.
The shared narrow-row label detector also raised full real non-stacked bar
coverage from 185/262 to 193/262 charts and 1714/3834 to 1807/3834 matched
bars. Accuracy within 1% across all ground-truth bars rose from 39% to 41%,
while accuracy among matched bars slipped from 87% to 86%. The tradeoff is
reported separately rather than hidden in the overall gain.

Full detector OCR on a narrow figure strip cost about 13 seconds across six
calls in the 88-chart audit. A narrow text-row projection now handles
right-aligned labels without that call. If it cannot calibrate, detector OCR
is attempted only when the proposed axis is long and left-side and box or
violin geometry is already present. The full real box evaluation took 27
seconds after this gate, versus about 14 seconds before the new recovery path.
These wall times are machine-state dependent; the 10-chart real-bar cross check
fell from 45.6 to 4.6 seconds after the gate.
On the complete 280-chart real-bar cross check, the box page produced panels
on 4 charts, the same four as before the axis-recovery change. This guards
against reporting extra bar figures as box plots, though those four cases
still need review.

Research directions still requiring separate evidence:

- [LineEX](https://openaccess.thecvf.com/content/WACV2023/papers/P._LineEX_Data_Extraction_From_Scientific_Line_Charts_WACV_2023_paper.pdf)
  groups detected keypoints and maps them through axes and legends, while
  [LineFormer](https://github.com/TheJaeLal/LineFormer) treats each curve as an
  instance. A trained model may help same-colour crossings and marker
  occlusion, but its weights, inference cost, and real-paper benchmark need
  evaluation before adding it to this fast local pipeline.
- [WebPlotDigitizer](https://github.com/automeris-io/WebPlotDigitizer) provides
  colour-based tracing and interactive correction. The next local experiments
  should compare explicit grid removal and per-series colour masks against
  the current dynamic-programming trace on fixed crossing, dashed-line, and
  low-contrast subsets.
- Build a real violin and mean-and-whisker dot gold set, measure false panels
  across chart types, and separately score axis calibration, group matching,
  displayed statistics, uncertainty flags, and extraction-only latency.
