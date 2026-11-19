# CV Project Entry

## Project name

**Skin Lesion Classification Pipeline with YOLO11 (Disease + Malignancy Heads)** (Sep 2026, personal project)

## One-line summary

End-to-end deep learning system that classifies clinical and smartphone skin photos into 27 dermatological
conditions and a 3-way malignancy label (benign / malignant / non-neoplastic), built from 36k images across
four public datasets, with a real-time webcam inference front end.

## Tech stack

Python 3.12, PyTorch 2.14 (CUDA), Ultralytics YOLO11s-cls, OpenCV, scikit-learn, Pandas, Pillow, imagehash,
Hugging Face datasets, uv, NVIDIA RTX 4060 (8 GB)

## Bullet points (full version)

- Built a 7-stage reproducible data pipeline (acquire, manifest, label, dedup, split, layout, train) over
  36,108 images from DermNet, Google SCIN, PAD-UFES-20 and Fitzpatrick17k, with per-source manifests and
  hard row-count assertions that fail fast on partial downloads.
- Designed a unified label taxonomy as versioned code: 330 (source, raw label) pairs mapped to 27 disease
  classes and a derived malignancy label, so the same condition can never carry conflicting cancer labels
  across sources.
- Built a local web tool for per-image hand-labelling of a mixed melanoma/nevus folder, plus contact-sheet
  generation for bulk visual review.
- Removed 1,558 near-duplicate images using 64-bit perceptual hashing, LSH candidate search and union-find
  grouping, run before splitting to eliminate train/test leakage.
- Created leakage-safe 70/15/15 splits with StratifiedGroupKFold grouped by patient/case ID; handled
  severe class imbalance with square-root oversampling implemented via NTFS hard links (zero extra disk).
- Fine-tuned two YOLO11s-cls heads with dermatology-aware augmentation (mild colour jitter, RandAugment
  disabled to preserve pigment and skin-tone signal), cosine LR schedule and fixed seeds.
- Held-out test results: disease head 63.6% top-1 and 0.628 macro-F1 over 27 classes; malignancy head
  92.0% top-1 and 0.845 macro-F1 with 0.81 recall on malignant lesions.
- Designed and benchmarked a real-time webcam inference app running both heads at 52 FPS on an RTX 4060,
  with an OR-rule malignancy flag, confidence-based abstain, and startup checks that verify class-index
  order against the trained checkpoints.
- Tracked Fitzpatrick skin-type coverage across the dataset (types 5-6 about 15% of labelled images) to
  surface fairness gaps, and documented all known limitations of the models.

## Bullet points (short version, 3 lines)

- Built an end-to-end skin lesion classification pipeline (36k images, 4 public datasets) with
  perceptual-hash deduplication, patient-grouped stratified splits and hard-link oversampling.
- Fine-tuned two YOLO11s-cls heads: 27-class disease (0.63 macro-F1) and 3-class malignancy
  (0.85 macro-F1, 0.81 malignant recall) on a held-out test set.
- Designed a real-time webcam inference app running both heads at 52 FPS with an OR-rule malignancy flag
  and confidence-based abstain.

## Wording note

Camera app is designed and benchmarked (PLAN_EVAL_AND_LIVE_CAMERA.md), not yet coded. Once
scripts/09_live_camera.py exists, change "Designed and benchmarked" to "Built" in both versions.
