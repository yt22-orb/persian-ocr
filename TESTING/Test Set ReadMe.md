# Persian Letter Recognition Test Set ReadMe

Prepared by Jeff Barrientos.

## Conditional fallback in the TESTING runner

The `test-preprocessing-branch` defaults to conditional fallback in this batch runner. The browser scanner, its backend, and all UI files are unchanged. The earlier committed `Results` and `Reports.docx` remain the original BASE evidence; new runs go to their separate output folders.

The runner tries BASE first. An accepted BASE answer is final, and a BASE multiple-letter rejection is final. Only no-detection or low-confidence BASE inputs are retried, in this order: contrast, contrast with eight rotations, then eight rotations without contrast. It stops at the first accepted stage. At rotation stages it chooses the highest-confidence accepted single-letter view, with the lower angle breaking equal-confidence ties. The 80% threshold is unchanged. If all stages fail, it retains the original BASE rejection.

Contrast converts to grayscale, normalizes the 90th-percentile paper level toward white, then applies CLAHE with clip limit 2.0 and a 4 by 4 tile grid. Rotations are 0, 45, 90, 135, 180, 225, 270 and 315 degrees counterclockwise. Right angles use lossless transposes; diagonals use bicubic interpolation and an expanded white canvas. Zero-degree results are reused, so at most 16 detector views are needed per image. Input files are never overwritten.

The new `selected_method`, `selected_angle_degrees`, `recovered`, `needs_review`, `attempted_methods`, and `trace_json` columns explain each fallback decision. The BASE prediction and correctness are also retained in every row for paired comparisons. Accepted fallback predictions have `needs_review=yes`; a high model score does not guarantee a correct answer. The new Word report compares BASE and fallback on the same images and reports both additional correct and additional wrong accepted answers.

Existing run commands below now run fallback by default. Add `--recognition base` to reproduce the original detector-only workflow. Add `--no-docx` to omit the optional Word report. The full repository also supplies 64 blank and 64 two-letter validation controls; if only TESTING was copied, these controls are skipped with a message. Keep `preprocessing.py`, `fallback_runner.py`, and `build_fallback_report.py` beside `run_test.py`. Reinstall `requirements.txt` to ensure OpenCV is available.

The completed 25 October 2026 local comparison is in [Conditional Fallback Results](Conditional_Fallback_Results/ReadMe.md), including the merged Word report, summary CSVs, validation record, and full downloadable evidence ZIP. The original four datasets were tested in full, with zero technical errors and all 128 synthetic controls rejected. Run `python -m unittest discover -s TESTING/tests -v` from the repository root for the ten fallback checks; the four existing API checks remain under `tests`. Mirrored-image checks are recorded in [Future Work](FUTURE_WORK.md) and remain deferred.

This folder tests the separately supplied Persian letter recognizer on two source datasets, each with an unmodified and a modified version. There are four input folders and 76,544 image evaluations in total. The script loads the supplied `yolo11n.pt`, checks its checksum and its 32 class names against `labels.json`, and produces one CSV row per image plus a report for each folder.

## The two source datasets

| Source | Original GitHub repository | Local unmodified folder | Images | Organization |
|---|---|---|---:|---|
| Printed Persian letters | [mehr32 Image dataset of Persian letters](https://github.com/mehr32/Image-dataset-of-Persian-letters/tree/main) | `DS1_Persian` | 6,272 | 196 font folders, with one image for each of 32 letters per font; 512 by 512 pixels |
| Handwritten Persian letters | [myousefnezhad persianhandwriting](https://github.com/myousefnezhad/persianhandwriting) | `DS2_Persian_Hand` | 32,000 | 32 letter folders, each containing the original Random1 through Random10 subsets; 77 by 95 pixels |

The printed images come from the source repository's `output` folder. Caption files were removed, and filename letter prefixes were changed to the canonical transliterations in the supplied labels.json; the source image pixels were preserved. The handwritten images were obtained from the repository's four-part raw 7z archive, which contains a UDF disk image and BMP files. Those BMPs were converted losslessly to PNG, preserving their pixels and dimensions; source class ordering was retained and the letter folders were named using labels.json. The repository's `PersianHWR1.mat` contains extracted features, not the original image pixels.

Here, “unmodified” means no synthetic rotation, flip, blur, or tone adjustment. It includes the naming and format preparation described above. The handwritten collection contains 31,998 distinct pixel images and two duplicate copies; all 32,000 supplied samples were retained. The source disk image has header warnings, but all 32,000 listed BMP files extracted at their expected sizes and decoded successfully.

## The four folders to upload or select

| Folder | Version | Image count | What to keep together |
|---|---|---:|---|
| `DS1_Persian` | Unmodified printed | 6,272 | All font folders and PNGs |
| `DS1_Persian_Smoke` | Modified printed | 6,272 | Nested `DS1_Persian` image folder plus `modifications.csv`, `summary.csv`, `run_settings.json`, and `labels.json` |
| `DS2_Persian_Hand` | Unmodified handwritten | 32,000 | All letter folders and Random subset folders |
| `DS2_Persian_Hand_Smoke` | Modified handwritten | 32,000 | Nested `DS2_Persian_Hand` image folder plus the four metadata files listed above |

The `_Smoke` folders are the full modified datasets; the name does not indicate a reduced sample. Each original has exactly one modified counterpart. Existing ZIP files inside the processed folders contain duplicate archive copies of the image folders and are ignored by the test script. They can be omitted when transferring the four folders, but retain the modification CSVs for category scores and paired comparisons.

## Modifications performed

| Modification category | Parameter range and order | Printed images | Handwritten images |
|---|---|---:|---:|
| Rotation | A non-zero angle from 5 to 40 degrees clockwise or counterclockwise | 5,024 | 28,800 |
| Flip | Random horizontal or vertical | 320 | 1,600 |
| Rotation and flip | Flip first, then rotate using the same angle range | 320 | 1,600 |
| Gaussian blur | Radius from 0.6 to 1.3 pixels | 320 | 0 |
| Brightness and contrast | Each factor independently selected from 0.75–0.90 or 1.10–1.25; brightness applied before contrast | 288 | 0 |

The requested printed allocation was 80% rotation and 5% each for the other four categories. Integer allocation per letter gives 157 rotation, 10 flip, 10 rotation-and-flip, 10 blur, and 9 brightness-and-contrast images for each set of 196 printed images. The actual overall percentages are 80.10%, 5.10%, 5.10%, 5.10%, and 4.59%, respectively. The handwritten allocation is exactly 90% rotation, 5% flip, and 5% rotation-and-flip, with 900, 50, and 50 images per letter.

Rotated images use an expanded white canvas, fitted back into the original dimensions without cropping the expanded result. Every output was checked to differ from its original. The seed was 20261024; the local processor also derives parameter choices from the relative image filename and assigned modification. Filenames have `rotation_`, `flip_`, `rotationFlip_`, `blur_`, or `brightnessContrast_` prefixes. Each modification CSV contains the exact selected parameters, original and output paths, canonical labels, seed, dimensions, and checksums. A positive logged rotation is counterclockwise; a negative angle is clockwise.

## Model and recognition rules

The model is not included in TESTING. Supply the trained YOLO detector used for the original evaluation; the runner does not download a replacement. The current repository UI can use a separate crop classifier, while these recorded results are for the supplied YOLO checkpoint. Its SHA256 is `1b82317638d84627ee1ca455bad970ce70041d12194b0fb9c7762c1ba51540ab`. Its class indices exactly match the characters and zero-based IDs in the supplied `labels.json`; for example, ID 0 is alef. The labels file SHA256 is `ba8483c682099bb42257ca8ab415b08c6f5cd4e0589d2cb201e133fcec38e0e1`.

This standalone runner evaluates the supplied YOLO detector using the saved recognition policy: square 640-pixel model input, candidate confidence floor 0.03, class-agnostic NMS IoU 0.40, and a maximum of 32 detections. Recognition is accepted only when exactly one candidate remains and its confidence is at least 0.80. No detection, multiple detections, or low confidence produces a rejected recognition. The supplied run uses FP32 inference on the NVIDIA RTX 4050 Laptop GPU; run_info.json records the model, library versions, device, and settings. No training, threshold tuning, refitting, or further image modification is performed by run_test.py.

## Contents and dataset download

Everything added on the testing branch is inside TESTING:

| Item | Purpose |
|---|---|
| DatasetZip.zip | The exact supplied archive containing the four full image folders and processed metadata |
| run_test.py and Run Test.cmd | Batch evaluation with native folder selection or explicit paths |
| labels.json and decision.json | Fixed class mapping and recognition policy |
| requirements.txt | Python dependencies |
| build_report.py and error_analysis.py | Rebuild one merged report and its error analysis from completed test results |
| Reports.docx | One document containing the interpretation, all four completed reports, and the error analysis of transformations, fonts, letters, and confidence thresholds |
| Results | Recorded per-image CSVs, per-letter/category metrics, and run metadata |

DatasetZip.zip is stored using [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage). Install Git LFS before cloning, or run these commands in your existing clone:

```powershell
git switch testing
git lfs install
git lfs pull --include="TESTING/DatasetZip.zip"
```

The ZIP should be 493,149,827 bytes, not a small text pointer. Its SHA256 is `cec1160dde331fe2f6f7731597cc4788b4fe0a80dd23b5018f708916d6e57f1d`.
Extract it once into a new data folder. The result must contain DS1_Persian, DS1_Persian_Smoke, DS2_Persian_Hand, and DS2_Persian_Hand_Smoke at the same level. On Windows:

```powershell
Expand-Archive -LiteralPath '.\TESTING\DatasetZip.zip' -DestinationPath '.\test_data'
```

Keep the Smoke wrapper folders and their metadata intact. Do not extract their internal duplicate ZIPs for evaluation.

## Run on Windows

Use Python 3.12 or a Python version supported by the pinned libraries. Run commands from TESTING. If the python command is unavailable, use the full path to your installed python.exe; the Windows py launcher is not required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_test.py --model "C:\models\yolo11n.pt"
```

Choose the four extracted folders when prompted, in the displayed order. Replace the example checkpoint path with your trained model file. On the original device that file was in Downloads as yolo11n.pt. The required SHA256 is shown above; the script rejects a different checkpoint or a mismatched class mapping. Putting the model in the repository root as yolo11n.pt also works, and the runner finds it automatically. Keep model files out of commits.

Run Test.cmd tries TESTING/.venv, then the prepared local GPU environment at ../../work/ocr_eval_env when present, then the repository .venv, then python on PATH. If no model is found in the repository root, the interactive runner opens a model file picker. Dependencies must already be installed in the selected environment.

For explicit folder paths, run:

```powershell
.\.venv\Scripts\python.exe run_test.py "..\test_data\DS1_Persian" "..\test_data\DS1_Persian_Smoke" "..\test_data\DS2_Persian_Hand" "..\test_data\DS2_Persian_Hand_Smoke" --model "C:\models\yolo11n.pt" --output ".\Runs\my_test" --device auto --batch 32
```

The runner automatically uses CUDA when available, otherwise CPU. For NVIDIA acceleration, install the appropriate CUDA build of PyTorch using the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install requirements.txt. Use --batch 16 if GPU memory is limited. The recorded run used an RTX 4050 Laptop GPU and FP32; CPU runs can take longer and results from different library or device versions can vary slightly.

A completed execution automatically writes one merged Reports.docx in its output directory. The default output is a new TESTING/Runs/timestamp folder, keeping the recorded Results separate. To resume an interrupted run, use the same explicit output path and settings; existing CSV rows and cached predictions are checked before reuse. Dataset inputs are not changed.

To rebuild the committed merged document from the recorded Results without loading a model:

```powershell
.\.venv\Scripts\python.exe build_report.py Results --output Reports.docx
```

For a hosted runtime, extract DatasetZip.zip, upload the separate model, and pass explicit paths; Tkinter is not required in command-line mode:

```python
!pip install -r requirements.txt
!python run_test.py /content/test_data/DS1_Persian /content/test_data/DS1_Persian_Smoke /content/test_data/DS2_Persian_Hand /content/test_data/DS2_Persian_Hand_Smoke --model /content/yolo11n.pt --output /content/test_results --device auto
```

## Output files and metric meaning

| Output | Contents |
|---|---|
| `predictions.csv` in each dataset report folder | One row per image: expected and accepted predicted letters, confidence, correct yes/no, status, candidate details, source image identity, modification category, and parameters |
| `metrics.csv` | Overall four metrics and separate metrics for each modification category |
| `per_letter.csv` | Precision, recall, F1, support, and correct counts for each of the 32 letters |
| `confusion_matrix.csv` | Expected letters by predicted letters, with a separate rejection or error column |
| `report.json` and `Report.md` | Machine-readable and human-readable dataset reports |
| `comparison.csv` | The four overall dataset results |
| `paired_comparison.csv` | Each modification group compared with the unmodified versions of those exact source images; identities and hashes must match |
| `run_info.json` | Checkpoint and labels checksums, inference policy, library versions, device, precision, batch size, and timestamp |
| `prediction_cache.jsonl` in new runs | Saved actual model outcomes for identical visible image inputs, enabling reuse and recovery; the original cache is omitted from the committed evidence |
| `Reports.docx` | One merged document containing the interpretation and all four dataset reports |

| Metric | Interpretation |
|---|---|
| Accuracy | Accepted correct predictions divided by all images, shown as a percentage |
| Macro precision | Precision calculated separately for each letter, then averaged equally across 32 letters |
| Macro recall | Recall calculated separately for each letter, then averaged equally across 32 letters |
| Macro F1 | The per-letter harmonic mean of precision and recall, then averaged equally across 32 letters |

Use macro F1 and accuracy as the main comparisons, with precision and recall explaining the errors. Rejected inputs count as misses for their expected letter; they are not silently removed from the metric denominator. Precision uses accepted predictions, while recall and F1 reflect missed recognition. Undefined per-letter scores are set to zero. Accuracy and macro recall can be equal because each letter has the same number of images in these test conditions. All score tables use percentages; differences between scores are reported in percentage points.

The `correct` column is yes only for an accepted prediction that matches the expected label. Rejected rows have no accepted predicted label; the highest candidate is retained separately for diagnosis. Read/checksum failures produce error rows and mark the dataset report incomplete. These are image-level recognition metrics, not bounding-box detection mAP.

## Scope of the interpretation

The runner reads expected labels from canonical printed filenames or handwritten letter folders, and uses the modification manifests for processed images. These labels were not manually re-annotated image by image. The numeric handwritten class ordering comes from the source Class.mat; readable aliases were matched to the supplied labels.json and sample images were visually checked during conversion.

Independence from the checkpoint's earlier training data has not been established. Results therefore describe recognition on these specific folders and the change under the saved modifications; they do not establish unseen-font or unseen-writer generalization. Source duplicate handwritten samples are retained and counted as supplied. No model setting is selected or tuned using these test scores.

Metric implementation follows the [scikit-learn definitions for precision, recall, F1, and macro averaging](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html). Inference uses [Ultralytics model prediction](https://docs.ultralytics.com/modes/predict/).

## Recorded result summary

| Dataset | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| DS1_Persian | 69.53% | 96.33% | 69.53% | 80.47% |
| DS1_Persian_Smoke | 47.98% | 91.52% | 47.98% | 61.58% |
| DS2_Persian_Hand | 2.62% | 67.74% | 2.62% | 4.68% |
| DS2_Persian_Hand_Smoke | 4.62% | 82.52% | 4.62% | 8.24% |

All 76,544 images were evaluated with zero technical errors. The low handwritten accuracy largely reflects rejected recognitions: 96.47% of original handwritten images were rejected. Among the 1,131 accepted original handwritten predictions, 74.09% were correct; that diagnostic excludes rejected inputs and is not the overall accuracy. The source numeric-to-letter mapping and model class IDs were checked, with visual sample checks across all 32 letters. This does not replace manual annotation of every image.

The preparation and test workflow is attributed to Jeff Barrientos. The dataset repositories remain credited to their original creators through the source links above.
