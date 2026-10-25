# Completed conditional fallback comparison

Jeff Barrientos tested all 76,544 images and 128 synthetic validation controls locally with the supplied YOLO checkpoint on 25 October 2026. The cascade preserved every accepted BASE answer and BASE multiple-letter rejection, and the independent decision replay and metric checks passed with zero technical errors. Fallback recovered 4,573 additional correct answers and also accepted 7,724 additional wrong answers, so recovered predictions are marked `needs_review=yes`. The browser scanner and UI are unchanged, and the order was selected from earlier results on these same datasets, so this is an implementation retest rather than independent validation.

| Dataset | BASE accuracy | Fallback accuracy | BASE macro F1 | Fallback macro F1 |
|---|---:|---:|---:|---:|
| Original print | 69.53% | 71.89% | 80.47% | 78.67% |
| Modified print | 47.98% | 63.03% | 61.58% | 69.41% |
| Original handwriting | 2.62% | 4.83% | 4.68% | 7.80% |
| Modified handwriting | 4.62% | 13.28% | 8.24% | 18.97% |

## Files

- [Reports.docx](Reports.docx) contains the interpretation and all four completed dataset reports.
- [All_Results.zip](All_Results.zip) contains the complete frozen local result package: per-image CSVs and decision traces, per-letter and modification metrics, the same Word report, validation metadata, and the tested standalone runner snapshot. No model weights or source images are included.
- [comparison.csv](comparison.csv), [negative_controls.csv](negative_controls.csv), and [paired_comparison.csv](paired_comparison.csv) provide the summary evidence without extracting the ZIP.
- [run_info.json](run_info.json) and [validation.json](validation.json) record the environment, checkpoint and source hashes, scope, and verification.

Use the current runner in the parent TESTING folder for new runs. The ZIP is the frozen package from the completed local test, and its README's pre-push status describes that snapshot. All 14 checks passed: ten fallback checks and four existing API checks. The extra eight mirrored-image checks remain deferred in [Future Work](../FUTURE_WORK.md).
