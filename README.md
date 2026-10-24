# Persian isolated-letter recognition

A 32-class Persian character recognition project. The Colab workflow uses fine-tuned YOLO11n for letter localization, then a crop-cleaning step and a trained PyTorch crop CNN for recognition. The [Colab notebook](https://colab.research.google.com/github/yt22-orb/persian-ocr/blob/main/Persian_Isolated_Letter_Recognizer.ipynb) runs this two-stage pipeline on images and exports predictions to CSV. The earlier full-image CNN remains available as a comparison. The trained YOLO checkpoint and comparison CNN are attached to the [latest release](https://github.com/yt22-orb/persian-ocr/releases/latest).

## What was built

The target is one complete, isolated Persian letter per image, including its dots. The detector has 32 classes, from alef through yeh. The project also includes a three-block CNN to compare detection against single-label classification, validation-only decision calibration, a local web upload demo, synthetic stress evaluation, and reports for provenance and visual review.

The local camera/upload interface runs the same detector and crop classifier as the Colab notebook. See [UI_README.md](UI_README.md) for setup and model download instructions.

The experiment uses rendered typography, not natural handwriting. The results below describe this benchmark only; they are not estimates of real-world OCR accuracy.

## Dataset creation and splitting

The generated benchmark dataset used for these results is published in [`data/prepared/`](data/prepared/README.md) (about 168 MB). It includes the synthetic train/validation/test images, validation/test negatives, stress views, YOLO labels, and checksum-bearing manifests. It is separate from [`dataset/`](dataset/README.md), which remains the Colab notebook's image upload folder.

The model was trained on synthetic isolated-letter images rendered from licensed fonts. Pillow’s RAQM text shaping and complete glyph alpha bounds were used so detached dots and strokes were included in the boxes.

| Split | Font families | Positive images | Images per class |
| --- | --- | ---: | ---: |
| Train | Lateef, Markazi Text, Vazirmatn, Gulzar, Noto Naskh Arabic, Noto Sans Arabic | 1,536 | 48 |
| Validation | Amiri | 256 | 8 |
| Test | Scheherazade New | 256 | 8 |

Each split covers all 32 classes. The split is by font family: no font family is shared between train, validation, and test. This tests transfer to two unseen fonts, although one font in each held-out split makes the benchmark small.

There are also 64 blank and 64 two-letter negative images in each of validation and test. These negatives are used to measure rejection behavior; they are not additional letter classes. The test set therefore has 256 positive examples plus 128 negatives.

The source Persian OCR archives were inspected and mapped to the project’s Unicode labels, but their connected-text crops were excluded from training and evaluation. Contextual letter forms and dot completeness were not individually verified. Synthetic renders were used instead. Exact image hashes and parent/font identities were checked for split separation. The audit found no near-duplicate glyph candidates, but its coarse fingerprint check cannot prove that every possible near duplicate is absent.

Generated manifests are frozen for the run. Any future real samples should include writer/session grouping metadata and be separated by those groups before training or evaluation.

## Training

Both models were trained on an AMD Radeon RX 7800 XT using ROCm 7.2 and PyTorch 2.11.0. The common run settings were seed 42, batch size 16, a maximum of 100 epochs, and early stopping patience 20. YOLO training completed 70 epochs; CNN training completed 86. No batch-size adjustment was made for hardware.

### YOLO11n detector

YOLO11n was initialized from Ultralytics’ pretrained `yolo11n.pt` weights and fine-tuned at 640-pixel input size. The run was deterministic and used full precision (`amp=false`). Horizontal/vertical flips, mosaic, mixup, and copy-paste were disabled. Mild geometric changes were enabled (rotation up to 3°, translation 0.01, scale 0.025, perspective 0.00005); value-channel brightness augmentation and training-only mild shadow, blur, and compression variation were also used. The best checkpoint was selected using the library’s validation detection fitness.

### CNN comparison

The comparison model has three convolution–ReLU–max-pooling blocks with 32, 64, and 128 channels, followed by global average pooling and a 32-logit output layer. It uses AdamW (learning rate 0.003, weight decay 0.0001) and cross-entropy loss.

CNN preprocessing converts to grayscale, crops around all pixels darker than 180 with a four-pixel margin, pads to a white square, resizes to 64×64 with LANCZOS, and maps inverted grayscale to [-1, 1]. The crop comes from image pixels and does not use a ground-truth box. CNN training-only augmentation includes small rotations, mild brightness/shadow changes, blur, and JPEG compression.

## Calibration and evaluation

Decision thresholds were selected using validation data only, then frozen before test evaluation. For YOLO, the candidate confidence floor was 0.03, the acceptance threshold was 0.80, and class-agnostic NMS IoU was 0.40. YOLO rejects an image if no candidate is found, multiple candidates remain, or the single candidate is below the acceptance threshold. The CNN uses a confidence threshold of 0.80; because it emits one class, it cannot explicitly count letters.

The calibration objective equally weighted valid-letter exact accuracy and negative rejection accuracy. On validation, YOLO achieved an objective of 0.9980 (99.61% valid accuracy and 100% negative rejection); the CNN objective was 0.8906 (92.19% valid accuracy and 85.94% negative rejection).

### Held-out test results

Rejected positive examples count as incorrect. Both models used the same 256 positive test identities.

| Model | Correct / valid | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Positive rejection rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11n | 256 / 256 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| Reduced CNN | 207 / 256 | 80.86% | 80.65% | 80.86% | 79.70% | 79.70% | 12.89% |

Precision, recall, and F1 are macro averaged across the 32 letter classes; weighted F1 uses class support. A rejected valid image counts as a false negative and an incorrect prediction. Rejected inputs are not counted as predictions of a letter. Per-class metrics and confusion matrices are available in [RESULTS.md](RESULTS.md) and the [published evaluation files](results/).

Negative rejection was measured separately (64 images per category):

| Model | Blank rejection | Two-letter rejection |
| --- | ---: | ---: |
| YOLO11n | 64 / 64 (100%) | 64 / 64 (100%) |
| Reduced CNN | 64 / 64 (100%) | 42 / 64 (65.62%) |

The two-letter negative set is deliberately small and synthetic. A second glyph the detector misses can still lead to a false acceptance.

### Synthetic stress views

A frozen stress set generated blurred and shadow-gradient versions of the held-out test images. These are synthetic transformations, not camera captures.

| Model | Blur exact accuracy | Shadow exact accuracy |
| --- | ---: | ---: |
| YOLO11n | 100.00% | 100.00% |
| Reduced CNN | 80.47% | 15.23% |

The test font, validation font, and all scores are small synthetic samples. No camera photos, natural handwriting, connected words, or pages were evaluated. The results do not support claims for those inputs. Dark photographic backgrounds can also interfere with the CNN’s foreground crop.

Detailed measurements, confusion matrices, per-class metrics, subgroup results, latency, and local verification reports are summarized in [RESULTS.md](RESULTS.md). Data and model source credits and license context are in [ATTRIBUTION.md](ATTRIBUTION.md).

## Bounding-box crop classifier

The crop classifier was trained to consume a tightly cropped letter, rather than a full image. YOLO11n supplies the predicted box. Preprocessing expands it by 6%, estimates and flattens near-background texture from the border, retains detached dots, and centers the grayscale result on a 96×96 white canvas. The model checkpoint, preprocessing/inference code, calibrated policy, split-preserving derived crop dataset, and machine-readable evaluation files are included in this repository.

Training used 1,536 train crops and 256 validation crops from font-family-separated synthetic splits. On an AMD Radeon RX 7800 XT with ROCm 7.2, three AdamW configurations were trained for up to 80 epochs; all reached 100% validation crop accuracy. The selected run (`lr=0.001`, `weight_decay=0.0001`, `dropout=0.25`) achieved 100% at epoch 76 with validation loss 0.0003685. See [`artifacts/crop_cnn/config.json`](artifacts/crop_cnn/config.json) for run metadata and [`CROP_PIPELINE_RESULTS.md`](CROP_PIPELINE_RESULTS.md) for metrics and limitations.

On the held-out Scheherazade New test font, crop CNN top-1 accuracy on YOLO boxes was 256/256 (100%). The frozen confidence-gated two-stage pipeline accepted 248/256 (96.88%), rejecting eight correctly labeled `vav` crops under the conservative 0.99 confidence threshold. Blank and two-letter negatives were rejected 128/128 with no false accepts. Standalone YOLO scored 256/256 (100%) on the same benchmark, so the crop cascade improves on the earlier full-image CNN (224/256 raw top-1) but does not outperform YOLO alone. These are small synthetic-font results, not handwriting or camera performance.

The Colab notebook now loads `artifacts/crop_cnn/best.pt` from this repository automatically. Keep `yolo11n.pt` from the latest release in Google Drive; that detector checkpoint is the one used to calibrate the crop policy.

## Run inference in Colab

1. Open the [Colab notebook](https://colab.research.google.com/github/yt22-orb/persian-ocr/blob/main/Persian_Isolated_Letter_Recognizer.ipynb).
2. Download `yolo11n.pt` from the [latest release](https://github.com/yt22-orb/persian-ocr/releases/latest) into Google Drive. The notebook loads labels, crop weights, preprocessing code, and calibrated thresholds from this repository.
3. Upload images or a ZIP into this repository’s [`dataset/` folder](https://github.com/yt22-orb/persian-ocr/tree/main/dataset). The notebook clones the public repository and scans that folder recursively. Or choose `colab_upload` in the notebook to upload a ZIP directly.
4. Run the notebook to download a CSV containing each image’s prediction, confidence, and review status. Class-named subfolders are included as reference labels in the CSV; the notebook does not calculate accuracy automatically.

The `dataset/` directory is an input location for user-provided inference images. It was not used to train or evaluate the released models. Do not commit private or sensitive images.

## Reproducibility and package scope

This repository is the inference package: it contains the notebook, labels, calibrated policies, model metadata, results summary, and the prepared benchmark under `data/prepared/`. The full training pipeline and raw source archives remain in the development workspace and are not included here. Reported settings and metrics document the completed run; exact retraining requires the training code and pinned environment recorded in that workspace.

The release ZIP includes the notebook, inference code, detector and full-image CNN checkpoints, labels, policies, results, and attribution files. The crop CNN checkpoint and derived crops are versioned in the GitHub repository. See [release v1.0.2](https://github.com/yt22-orb/persian-ocr/releases/tag/v1.0.2).
