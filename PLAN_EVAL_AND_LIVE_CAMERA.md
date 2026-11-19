# Plan: Evaluation + Live Camera Inference App

> **Status: written 2026-09-20, not started.**
>
> Execution is deliberately held. The decision to run this plan depends on the Stage 6
> training results: if Model A and Model B come out good enough to be worth building on,
> this plan runs as written. If they do not, the right response is to improve the models
> first, not to wrap weak ones in an app.
>
> Nothing here has been implemented. `scripts/07_eval.py` and `scripts/09_live_camera.py`
> do not exist. The measured benchmark numbers in Stage 2 are real and were taken on this
> machine while the disease training run was in flight.


## Context

The project has two trained YOLO-cls models and no way to use or judge them.

`scripts/06_train.py` writes only top-1 and top-5 accuracy to `test_metrics.json`. For
this problem those numbers are close to meaningless. `PIPELINE_PLAN.md` Stage 7 names
macro-F1 as the headline metric for Model A and malignant recall as the metric Model B
lives or dies by. Neither is computed anywhere. `scripts/07_eval.py` is referenced in the
plan and in `PROJECT_OVERVIEW.md` but has never been written.

Separately, the repo contains no inference code at all. The pipeline stops at training.
The user wants a live webcam app: point a camera at a skin lesion, see the condition and
whether it looks cancerous.

These two are coupled. The app's abstain behavior needs a confidence threshold, and that
threshold must be chosen on val data and frozen before test is touched. Inventing a
number would make the app's most safety-relevant behavior arbitrary. So evaluation comes
first and hands the app a frozen threshold.

Outcome: a reproducible metrics report that reflects what these models actually do, and a
desktop app that runs them on a live camera without mis-mapping a class or overstating its
own confidence.

---

## Stage 1 — `scripts/07_eval.py`

Follows existing script conventions exactly: module docstring stating rationale,
`ROOT = Path(__file__).resolve().parents[1]`, module-level constants, argparse,
`main() -> int`, `sys.exit(main())`, deferred `ultralytics` import.

### Inputs

| Path | Use |
|---|---|
| `runs/disease/weights/best.pt` | Model A |
| `runs/malignancy/weights/best.pt` | Model B |
| `data/yolo/dataset_disease/{val,test}` | Model A evaluation splits |
| `data/yolo/dataset_malignancy/{val,test}` | Model B evaluation splits |
| `data/manifest_final.csv` | per-image `fitz_scale`, `source`, `group_id` |
| `data/label_map.csv` | disease to malignancy lookup |

Evaluate on the unbalanced trees, never the `_bal/` ones. The balanced trees oversample
the train split only, but pointing at them invites the mistake.

### The join back to metadata

The fairness breakdown needs `fitz_scale` per predicted image. `scripts/05_layout.py:38`
defines `flat_name`, which is deterministic:

```
data/processed/dermnet/train/Melanoma Skin Cancer Nevi and Moles/006618.jpg
  -> dermnet__train__Melanoma_Skin_Cancer_Nevi_and_Moles__006618.jpg
```

Build `{flat_name(row["path"]): row}` over `manifest_final.csv` and look up each
prediction by its file basename. Reuse `flat_name` by copying it verbatim with a comment
naming its source; the scripts are not importable, filenames start with digits.

Two assertions, both mandatory:

- **Collision check.** `flat_name` applies `.replace(" ", "_")`, which is lossy in
  principle. Assert the dict has as many keys as the manifest has rows.
- **Coverage check.** Every file found in the test tree must resolve to a manifest row.
  A miss means the tree and the manifest have drifted.

**Trap:** the `train` inside those paths is DermNet's own folder name, not this project's
split. The sample row above has `split == "test"` while its path says `train`. Never infer
split from the filename. Use the manifest's `split` column.

### Metrics to compute

**Model A (disease), on test:**
- top-1, top-5
- **macro-F1** — the headline, printed first
- per-class precision, recall, F1, support, sorted by support ascending so the rare
  classes are the ones you read first
- confusion matrix as CSV alongside the PNG, so the `melanoma` to `melanocytic_nevus`
  cell can be read numerically rather than squinted at

**Model B (malignancy), on test:**
- per-class precision, recall, F1
- **recall on `malignant`** as its own printed line
- full ROC for `malignant` versus the rest, with AUC
- sensitivity at fixed 90% specificity, with the operating point that produces it

**Melanoma, its own section:** recall on the `melanoma` disease class, and what Model B
predicted for those same images. Melanoma has 214 images total and 32 in the test split.
State the count next to the number. A recall computed on 32 images has wide error bars
and the report must not present it as precise.

**Fairness, `fitz_scale` buckets 1-2, 3-4, 5-6, plus unknown:** every Model B metric
repeated per bucket. Report bucket sizes next to every figure. Of 29,640 images, 18,777
have no Fitzpatrick annotation at all, so the known-type subset is 10,863 and types 5-6
are 1,633 of those. Any per-bucket number here is directional, not precise.

### Threshold selection

Run on **val**, apply to **test**, never the reverse. Selection criterion: the lowest
confidence at which `malignant` specificity reaches 90%. Write the result to
`runs/thresholds.json`:

```json
{
  "malignancy_abstain": 0.0,
  "disease_abstain": 0.0,
  "malignancy_operating_point": 0.0,
  "chosen_on": "val",
  "criterion": "sensitivity @ 90% specificity",
  "n_val": 0,
  "date": "YYYY-MM-DD",
  "model_a_run": "disease",
  "model_b_run": "malignancy"
}
```

Two thresholds, never one shared value. A 0.45 softmax peak over 3 classes and a 0.45 peak
over 27 classes represent very different amounts of evidence.

### Outputs

```
runs/eval/metrics.json          every number, machine readable
runs/eval/report.md             the same, human readable, macro-F1 first
runs/eval/per_class_disease.csv
runs/eval/per_class_malignancy.csv
runs/eval/fairness_fitzscale.csv
runs/eval/confusion_disease.csv
runs/eval/roc_malignant.png
runs/thresholds.json            consumed by Stage 2
```

---

## Stage 2 — `scripts/09_live_camera.py`

One self-contained script plus `scripts/run_live_camera.cmd`, a launcher mirroring
`scripts/run_full_training_skin.cmd` so the `skin\Scripts\python.exe` path is not
something to remember.

Numbered `09`, not `07` or `08`: `07_eval.py` is Stage 1 of this plan and Stage 8 is
reserved for ONNX export in `PIPELINE_PLAN.md:509`.

Single file, not a package. Every script here re-derives its own constants, there is no
`__init__.py`, and filenames start with digits so they cannot be imported. Splitting this
into a package would invent the repo's first cross-module import for one feature.
`scripts/02c_build_label_map.py` is 16.8 KB, so a ~450-line app is within the norm.

### Modes, one shared code path

| Mode | Purpose |
|---|---|
| default | live camera |
| `--selftest` | run every startup assertion, print the index-to-name-to-malignancy table, exit non-zero on mismatch. No camera, no GPU |
| `--image PATH` / `--image-dir DIR [--expect CLASS]` | same pipeline on files |
| `--bench N` | N frames, ms split by capture / model A / model B / draw |

A separate verification script would duplicate the mapping logic and drift from the app.
Modes share one path.

### Model loading and the class-index assertion

Use `YOLO(path)` from ultralytics. **Never hand-roll `torch.load`.** The in-flight
`runs/disease/weights/best.pt` stores `model=None` with real weights under `ema`;
`YOLO()` resolves that, a naive loader returns `None` and crashes.

`verify_model_classes(model, classes_txt, expected_n)`, called per model, hard-failing
with exit code 2:

```
file_names  = [ln.strip() for ln in txt.read_text("utf-8").splitlines() if ln.strip()]
model_names = [model.names[i] for i in range(len(model.names))]
len mismatch        -> fail "stale classes file / wrong checkpoint"
set mismatch        -> fail, print symmetric difference
order mismatch      -> fail, print FIRST differing index and both names
```

Four details that matter:

1. `.splitlines()` and strip, never `.split("\n")`. Both class files end with a trailing
   newline, which `.split` turns into a phantom empty class.
2. `[model.names[i] for i in range(...)]`, never `list(.values())`. The invariant is
   "index *i* maps to name *x*" and only integer indexing expresses it.
3. Distinguish reordered from different. A set mismatch means the wrong checkpoint. A set
   match with an order mismatch is worse: every prediction is wrong by a permutation and
   accuracy merely looks mediocre instead of broken.
4. No fallback. A mismatch means one artifact is stale and the app cannot know which.

Model B additionally asserts its names equal `{benign, malignant, non-neoplastic}` against
module constants. **`non-neoplastic` uses a hyphen, not an underscore.** The literal
appears exactly once in the file, so a typo fires this assertion at startup rather than
making the OR rule silently never match.

Display names come from `model.names` at runtime, but only after the file has agreed.

Print provenance at startup: absolute path, mtime, checkpoint epoch. **If `smoke` appears
in a resolved weights path, render a permanent red banner `SMOKE CHECKPOINT — PREDICTIONS
MEANINGLESS`.**

### Disease to malignancy lookup

Built from `data/label_map.csv` with `csv.DictReader`. Skip the `__exclude__` and
`__per_image__` sentinels. Assert 1:1, failing with the offending disease and its
conflicting values. Result is 33 entries.

Assert every one of Model A's 27 names is a key. The reverse is expected and fine: 6
diseases in the CSV are dropped-as-rare and unreachable by the disease head
(`cutaneous_lymphoma`, `kaposi_sarcoma`, `malignant_lesion_nos`, `keratosis_pilaris`,
`miliaria`, `photodermatosis`). Keep them, do not prune, print one line noting them.

**Tripwire, hardcoded on purpose:**

```
MALIGNANT_DISEASES = {actinic_keratosis, basal_cell_carcinoma, melanoma, squamous_cell_carcinoma}
BENIGN_DISEASES    = {melanocytic_nevus, seborrheic_keratosis_benign_tumor, vascular_tumor}
```

Assert the lookup reproduces these exact sets. This hardcodes a derived fact and will need
editing if the taxonomy legitimately changes. That is the point: if someone re-runs
`scripts/02c_build_label_map.py` with an edited rule and melanoma silently becomes benign,
the app refuses to start rather than painting a green banner over a melanoma.

### Decision rule and abstain

Base rule from `PROJECT_OVERVIEW.md:49-53`:

```
malignant_flag = (Model B says malignant) OR (lookup[Model A top-1] == malignant)
```

The prose rule assumes both heads always answer. With abstain, resolve all four cases:

| Model B >= t_B | Model A >= t_A | Verdict |
|---|---|---|
| yes | yes | B's verdict, upgraded to MALIGNANT if `lookup[A.top1] == malignant` |
| yes | no | B's verdict as-is |
| no | yes | MALIGNANT if `lookup[A.top1] == malignant`, else ABSTAIN |
| no | no | ABSTAIN, "unsure — move closer / better light" |

Row 3 is asymmetric deliberately. A confident disease head may raise a malignant flag when
Model B abstains, but may never assert benign on its own. That is the recall bias applied
consistently. Tag the verdict with its source (`Model B` / `disease-derived` / `both`) in
small text so the user can see which branch fired.

Threshold loading: `--conf-malignancy` / `--conf-disease` override; else
`runs/thresholds.json` from Stage 1, rendered as `THRESHOLD: 0.62 (val-frozen <date>)`;
else provisional mode with a permanent amber `THRESHOLD: UNSET (provisional)` tag. Keys
`[` and `]` nudge the malignancy threshold by 0.05 for eyeballing. The amber tag stays
regardless, because nudging is not calibration.

### Display

Per the locked decision: Model B's verdict large and primary, Model A's top-3 with
confidences smaller beneath, abstain message in place of a guess below threshold.

### Frame pipeline

```
VideoCapture.read() -> BGR uint8
  -> quality gate (mean luma, Laplacian variance)
  -> model_a.predict(frame, imgsz=320, device=dev, verbose=False)
  -> model_b.predict(same frame, same args)
  -> combine() -> verdict, source, top3, abstain
  -> draw_overlay() on a copy
  -> imshow + waitKey(1)
```

**Capture:** `cv2.VideoCapture(index, cv2.CAP_DSHOW)`. DirectShow is compiled into the
OpenCV 5.0.0 already in `skin/` and opens far faster than MSMF on Windows. Set
`CAP_PROP_BUFFERSIZE=1` so `read()` returns the newest frame, which matters specifically
because there is no temporal smoothing. DSHOW may ignore the property; do not assert on
it. Flags: `--camera N`, `--backend {dshow,msmf,any}`, `--res WxH` default 1280x720,
`--mirror` default on.

**Preprocessing: pass the raw BGR numpy frame straight to `predict()`.** Ultralytics then
applies the exact validation transform, `Resize(short edge to 320) -> CenterCrop(320) ->
ToTensor -> Normalize(ImageNet)`. Hand-rolling `cv2.resize(frame, (320,320))` would
stretch instead of center-crop, diverging from training silently with no error. It also
invites a BGR/RGB swap, which also fails silently. Always pass `imgsz=320` explicitly;
`predict`'s own default is 640.

**Draw the crop square.** Center crop means the model never sees the frame edges. On a
1280x720 capture it sees only the middle 720x720. Render that square and dim the outside
~40% with `addWeighted`. Without it "move closer" is meaningless, because the user cannot
see what is being discarded.

**Two sequential calls, not batched.** These are two different networks. Batching only
helps for one model over many frames.

**Device:** `--device {auto,cuda,cpu,0,1}`, default auto = cuda when available and
`mem_get_info()` shows more than 1.5 GB free, else cpu with a printed reason.

**Warmup:** 2-3 `predict` calls on a synthetic zero frame before opening the window. The
first CUDA call costs several hundred ms and the window otherwise appears frozen.

### Measured performance

Benchmarked on this machine with the disease training run concurrently in flight, both
models, imgsz=320, fp32, full `predict` API:

| Device | ms/frame, both models | FPS |
|---|---|---|
| CUDA, RTX 4060 Laptop, training concurrent | 19.2 | 52 |
| CPU, 14 threads, training contending | 89.2 | 11.2 |

VRAM cost: ~61 MB of tensors plus CUDA context, against 7.36 GB free. The app does not
meaningfully disturb training, so `auto` correctly picks CUDA even now.

**52 FPS exceeds any webcam's 30 FPS, so per-frame independent inference costs nothing on
GPU.** Build the straightforward single-threaded loop. No capture/inference threading in
v1: it is unnecessary on the real configuration, it is the main source of concurrency bugs
in apps like this, and a held label describing a frame no longer on screen is worse than a
slower honest preview.

Print a warning when `--device cpu` resolves: expect 10-15 FPS.

### The known cost of per-frame raw prediction

The user chose pure per-frame inference with no smoothing, and was warned. Designing for
it anyway, the real cost is legibility, not speed:

- At 30 FPS a 27-way head near its decision boundary re-orders the top-3 several times per
  second, which is effectively unreadable as scrolling text.
- The OR rule makes the large verdict bistable. One frame flipping Model A's top-1 into
  any of the 4 malignant classes flips the whole banner red. Expect strobing.

One escape hatch that is **not** smoothing: `--panel-hz 5` redraws the text panel at 5 Hz
using the most recent single frame's raw result. It decimates the display rate and
combines zero frames. **Default 0**, meaning redraw every frame exactly as specified.

Worth stating: the strobing is informative. Visible instability is honest evidence that
the model is not confident.

### Error cases

| Case | Behavior |
|---|---|
| No camera | Probe indices 0-3, print which opened and the backend tried, exit 1. Never let an empty frame reach cv2 |
| Camera held by another app | On Windows DSHOW usually opens but `read()` fails. Name the likely cause, retry once with the other backend |
| Weights missing | `Path.exists()` before loading. Print the expected path and where to check progress. Two explicit escapes, never automatic: `--model-b runs\malignancy_smoke\weights\best.pt`, or `--no-model-b` for Model A only with a large `DISEASE-DERIVED ONLY` tag |
| Torn checkpoint read | `best.pt` is rewritten every epoch during training. Catch `UnpicklingError`/`EOFError`/`RuntimeError`, retry 3x at 5 s, then fail naming mid-write as the likely cause |
| Class index out of range | `names.get(idx)` returning None renders `IDX n UNKNOWN` and the loop keeps running. Do not kill a live session over a display glitch |
| Frame read failure mid-stream | Count consecutive failures. Under 30 (~1 s), keep the last good frame with a red tint and a dropped-frames tag. At 30, release and reopen once, then exit 1. Keep `waitKey(1)` in every path so it never busy-loops |
| Window closed with X | `getWindowProperty(WIN, WND_PROP_VISIBLE) < 1` exits cleanly, else the loop holds the camera invisibly |
| Ctrl-C | `try/finally` always releases the capture and destroys windows |
| Too dark / bright / blurred | Not an error, a reason to abstain. Mean luma below 40 or above 215, or Laplacian variance under a floor, shows the specific reason. Gate behind `--quality-gate`, default on. The floor is another uncalibrated constant and gets the same provisional treatment |

---

## Verification

**Stage 1:**
1. `07_eval.py --task disease --split test` reproduces the existing `test_metrics.json`
   top-1 to within floating-point noise. If it does not, the evaluation path disagrees
   with the training path and everything downstream is suspect.
2. Per-class support column sums to the test split size, 4,120 for disease and 4,445 for
   malignancy.
3. The `flat_name` join covers 100% of test files with zero key collisions.
4. `runs/thresholds.json` exists, and re-running with the same seed produces the same
   threshold.

**Stage 2, four of five work before Model B exists:**
1. **V0 mapping proof.** `--selftest`, no camera, no Model B. Prints the 27-row table and
   exits non-zero on any assertion failure.
2. **V1 known image to known name.** `--image-dir data\yolo\dataset_disease\test\basal_cell_carcinoma --expect basal_cell_carcinoma`.
   Assert **modally**, not per image: over N images of class X, the most frequently
   predicted class must be X. Per-image assertion would flake constantly at 0.64 top-1.
   The modal test separates a broken mapping (consistently wrong class, often a constant
   offset) from an ordinarily inaccurate model (correct mode, wide spread).
   Use `basal_cell_carcinoma` (164 test images) and `melanocytic_nevus` (78) as the
   primary checks. **`melanoma` has only 32 test images**, so treat its modal result as
   weak evidence.
3. **V2 camera path today.** Run with `--model-b runs\malignancy_smoke\weights\best.pt`,
   which exists now, or `--no-model-b`. Proves capture, overlay, FPS, keys, window close,
   cleanup. Predictions are meaningless and the automatic red smoke banner says so.
4. **V3 colour-channel proof.** A BGR/RGB swap does not crash, it degrades quietly. Run
   `--image X` twice, once with a deliberate `--bgr-swap`, and confirm the prediction
   changes. If swapping channels changes nothing, the pipeline is ignoring the input.
5. **V4 after Model B finishes.** Re-run V0 and V1 against the real weights, confirm the
   3-class assertion, spot-check the OR on melanoma test images.

---

## Honest boundaries, to be stated in the app UI and the eval report

Not a medical device, not a diagnosis, no regulatory clearance, no claim of clinical
validity. Labels come from public datasets of varying quality, and the DermNet folder-11
melanoma/nevus split was hand-sorted by a layperson and has not been cross-checked.

**The domain gap is the central caveat.** Training images are dermatology-atlas
photographs: lesion centered, in focus, filling the frame, evenly lit. A live webcam frame
is none of those. Auto white-balance drift hits the model directly, because colour
sensitivity was deliberately preserved during training. Motion blur and rolling-shutter
noise have no analogue in the training set. Most fixed-focus webcams cannot focus closer
than ~20 cm, so "move closer" runs into a hard optical limit. At a natural sitting
distance the center crop is often a shirt or a wall.

Val top-1 of 0.639 is measured on the same four sources the model trained on, with the
same capture modality. Real-world accuracy under this much covariate shift will be
materially lower, and **it is unquantified until someone evaluates genuinely
out-of-domain images.** Never print "64% accurate" in the UI.

Three caveats the UI carries permanently, not in a dismissible splash:

- Melanoma has 214 images total and 149 in train. The deadliest class is the thinnest.
- Skin types 5-6 are ~15% of the labelled subset. Performance on darker skin is
  unvalidated and likely worse.
- The OR rule is deliberately recall-biased. False MALIGNANT flags are expected behavior,
  not bugs.

**Out of scope:** lesion detection or localization (these are whole-image classifiers,
there is no bounding box), multi-lesion frames, longitudinal tracking, any record keeping,
and dermoscopy input, which is out of distribution by construction.

---

## Blocked until training finishes

Stage 1 needs `runs/malignancy/weights/best.pt`, which does not exist yet. Disease is at
epoch 40/60 with best top-1 0.6513 at epoch 39, so early stopping may trigger at epoch 51.
Malignancy has not started. Both legs complete somewhere between 09:30 and 11:30.

Stage 2's V0 through V3 can all be done before then.

**Housekeeping:** `nul` in the repo root is a 122-byte junk file from a redirect mistake,
safe to delete.
