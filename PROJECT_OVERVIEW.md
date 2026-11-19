# Skin Lesion Classifier — Project Overview

A two-model skin image classifier built from four public dermatology datasets and
trained locally on an RTX 4060 Laptop (8 GB VRAM). Model A names the condition.
Model B says whether it is cancer.

This document explains what every script does, how the raw images became a training
set, how that set was split, and why the project ships two models instead of one.

For the design reasoning behind each choice, see `PIPELINE_PLAN.md`. For the decision
to avoid Kaggle and work from a reduced Fitzpatrick17k set, see
`PLAN_B_LOCAL_NO_KAGGLE.md`. This file is the operational summary of what was actually
built.

---

## 1. Why two models

### The constraint

Ultralytics `yolo classify` supports exactly one label per image. A single YOLO-cls
model cannot emit both a disease name and a malignancy verdict. Something had to give.

### The two options considered

**Derive malignancy from the predicted disease** using a lookup table. One training run,
no extra cost. The problem is that the malignancy answer inherits every disease-level
mistake. A lesion the model has never seen collapses to whatever class sits nearest in
feature space, and the cancer verdict rides along with it.

**Train a second, dedicated malignancy model.** Costs one more training run, but
optimises directly for the output that carries the clinical weight.

### The decision: both, combined with OR

| | Model A | Model B |
|---|---|---|
| Question | "what is it" | "is it cancer" |
| Classes | 27 disease groups | `benign`, `malignant`, `non-neoplastic` |
| Training images | 27,480 | 29,640 |
| Epochs | 60 | 40 |
| Dataset tree | `data/yolo/dataset_disease` | `data/yolo/dataset_malignancy` |

Model B trains on the **full** merged set, including every image whose disease class was
dropped for being too rare. A lesion too uncommon to name still carries a valid
malignancy label, and that label is Model B's entire job. This is why Model B sees 2,160
more images than Model A.

The two outputs combine at inference:

```
malignant_flag = (Model B says malignant)
              OR (lookup[Model A top-1] == malignant)
```

An OR, not an average, and the asymmetry is deliberate. A false positive costs the user
a needless dermatologist visit. A false negative costs them a missed cancer. The system
is biased toward recall on the malignant class, and any UI built on it should say so
plainly.

### Why malignancy is derived from disease, never from source

In `scripts/02c_build_label_map.py`, the malignancy label is computed from the canonical
disease name, not from the source dataset's own annotation. This makes it impossible for
the same condition to carry two different malignancy labels depending on which dataset
the image came from.

---

## 2. Data sources

Four public sources, no Kaggle involvement.

| Source | Images fetched | What it contributes |
|---|---|---|
| DermNet (HF `exper1ment/dermnet`) | 19,559 | The backbone. 23 clinically coherent folders, the largest single source |
| SCIN (Google, `gs://dx-scin-public-data`) | 10,364 from 5,033 cases | Consumer-submitted photos, dermatologist-weighted labels |
| PAD-UFES-20 (Mendeley) | 2,298 | Malignant lesions photographed **on smartphones** |
| Fitzpatrick17k (atlas share only) | 3,887 | Fitzpatrick skin-type annotations |
| **Raw total** | **36,108** | |

### Two things worth knowing about the sources

**Fitzpatrick17k is incomplete.** Only 3,887 of the original 16,577 rows were
recoverable, which is 23.4%. The `www.dermaamin.com` share, 12,631 rows, is DNS-dead.
Only the `atlasdermatologico.com.br` URLs still resolve. A request has been submitted to
the dataset maintainers; if a full archive arrives, Stage 1 should be re-run against the
complete set. That would add roughly 573 melanoma-family images, which is the single
biggest available fix for the weak melanoma class.

**No dermoscopy data, on purpose.** Merging ISIC or HAM10000 would push the count past
60k and add thousands of confirmed cancers. It would also poison the model. Dermoscopy
images are captured through a contact lens with polarised light at fixed magnification;
clinical photos are not. If nearly all malignant examples arrive through a lens and
nearly all benign examples do not, the cheapest separating feature the model can learn
is *capture modality*, not pathology. Validation accuracy looks excellent and collapses
on real phone input. PAD-UFES-20 is in the mix specifically to break a milder version of
the same correlation: it supplies malignant lesions shot on smartphones, so "malignant"
is not perfectly predicted by any one source.

---

## 3. What each file does

### Stage 0 — acquire

| File | Job |
|---|---|
| `scripts/00a_fetch_dermnet.py` | Pulls DermNet from Hugging Face into an ImageFolder tree at `data/raw/dermnet/`. Resume-safe: non-empty existing files are skipped |
| `scripts/00b_fetch_scin.py` | Fetches SCIN from its public GCS bucket over plain HTTPS, no auth. Every object is verified against the md5Hash reported in the bucket listing |
| `scripts/00c_fetch_fitz_atlas.py` | Recovers the live 24% of Fitzpatrick17k. Throttled to **one request per second** because the host is a small academic server. Each download is verified against the CSV's md5hash |

PAD-UFES-20 has no fetch script. It was downloaded manually from Mendeley through a
browser and placed at `data/raw/pad_ufes_20/`.

All three fetchers are resume-safe. An existing file whose hash matches is skipped
without a network request.

### Stage 1 — manifest

`scripts/01_manifest.py` merges all four sources into one row-per-image table at
`data/manifest_raw.csv`. Every row carries `path`, `source`, `raw_label`, `group_id`,
`fitz_scale`, `width`, `height`, `bytes`. Source-specific columns are kept and left
blank where they do not apply.

It also emits `data/label_map_skeleton.csv`, one row per `(source, raw_label)` pair with
counts. That skeleton is where Stage 2 starts.

The script **exits non-zero** if any of these fail:

- DermNet rows are exactly 19,559
- SCIN images on disk are exactly 10,379, and every labelled case's images exist
- PAD-UFES-20 rows are exactly 2,298 and every `img_id` resolves to a file
- Fitzpatrick17k ok-rows divided by 3,905 is at least 0.95

These are assertions, not warnings. A silent partial download is the kind of failure
that surfaces three stages later as an unexplained accuracy drop.

SCIN rows are filtered by `SCIN_MIN_WEIGHT = 0.5`, a dermatologist-agreement floor on
the top condition. Of 5,033 cases, 1,972 are unlabelled and 837 fall below that floor.

### Stage 2 — labelling

| File | Job |
|---|---|
| `scripts/02a_sort_folder11.py` | A local web app on `http://127.0.0.1:8765` for hand-sorting DermNet folder 11 ("Melanoma Skin Cancer Nevi and Moles"), one image at a time. Keys: `M` melanoma, `N` nevus, `U` or `?` unsure. Labels append to CSV as pressed, so the session can be stopped and resumed |
| `scripts/02b_contact_sheets.py` | Builds 3x2 contact sheets of the same folder for faster bulk review, with a CSV mapping `(sheet, cell)` back to image paths |
| `scripts/02c_build_label_map.py` | The label map itself: every `(source, raw_label)` to `(disease, malignancy)` |

**The label map is written as code, not as a spreadsheet.** A Python dict is diffable,
reviewable and re-runnable. A hand-edited spreadsheet is none of those. Edit the dicts,
re-run, and Stage 4 picks up the change. Output is `data/label_map.csv`: 330
`(source, raw_label)` pairs across DermNet 23, SCIN 211, Fitzpatrick17k 90,
PAD-UFES-20 6.

Two cases are resolved **per image** rather than per label:

- DermNet folder 11 mixes melanoma and benign moles in one folder. `data/dermnet_folder11_labels.csv` decides each image individually. Unsure images are excluded from both models.
- DermNet's "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions" folder is entirely malignant, but *which* malignancy is unknown. Those rows go to Model B only.

SCIN labels not listed in the map are excluded from both models. The 164-label tail is
804 images, too thin per label, and contains a few neoplastic diagnoses, so defaulting
them to "non-neoplastic" would be actively wrong for some.

The annotator for folder 11 is a layperson. The CSV records that, and the plan calls for
cross-checking those labels against Model B's predictions.

### Stage 3 — deduplication

`scripts/03_dedup.py` removes near-duplicate images across all four sources.

| Setting | Value |
|---|---|
| Hash | 64-bit pHash (`imagehash.phash`, hash_size 8) |
| Duplicate threshold | Hamming distance <= 5 |
| Candidate search | 4-band LSH on 16-bit slices, any shared band is a candidate |
| Grouping | union-find |
| Survivor per group | highest resolution |

Results:

| Metric | Count |
|---|---|
| Candidate pairs checked | 120,479 |
| Duplicate pairs confirmed | 1,739 |
| Duplicate groups | 1,431 |
| Images inside groups | 2,989 |
| Chained groups dissolved (diameter > 5) | 4 |
| Cross-source groups | 1 |
| Groups with differing labels | 495 |
| **Rows dropped** | **1,558** |

Per source: DermNet 1,522, SCIN 16, PAD-UFES-20 16, Fitzpatrick17k 4. Output is
`data/manifest_dedup.csv` at 34,550 rows.

**This runs before the split, not after.** That ordering is the whole point: a duplicate
that survives into both train and test turns the reported accuracy into a partial
memorisation score.

Hashes are cached in `data/phash.csv`, so re-runs skip already-hashed files.

### Stage 4 — apply labels, filter, split, resize

`scripts/04_split.py` is where the dataset is actually decided. It applies its rules in
strict order.

1. **Apply the label map** to get `(disease, malignancy, model_a, model_b)`. Folder 11 is resolved per image.
2. **Resolve duplicate-group conflicts.** If members of a duplicate group disagree on disease, `model_a = 0`. If they disagree on malignancy, the row is dropped outright, because the cancer label cannot be trusted either way.
3. **Drop images with short side below 200 px.**
4. **Drop Model A classes below 100 images.** Those rows stay for Model B.
5. **Split** with `StratifiedGroupKFold`.
6. **Resize and write** to `data/processed/`, long edge 512 px, JPEG quality 92.

Constants: `MIN_SIDE = 200`, `MIN_IMAGES = 100`, `LONG_EDGE = 512`, `JPEG_Q = 92`,
`N_FOLDS = 20`, `SEED = 42`.

The attrition, in order:

| Step | Rows | Dropped |
|---|---|---|
| `manifest_dedup` in | 34,550 | |
| After label map | 29,702 | 4,848 excluded by map or marked unsure |
| Duplicate-group conflicts | | 33 dropped on malignancy conflict, 448 set to Model B only |
| After size filter | 29,640 | 29 below 200 px |
| **`manifest_final`** | **29,640** | |

Five disease classes fell below the 100-image floor and became Model B only:
`cutaneous_lymphoma`, `kaposi_sarcoma`, `keratosis_pilaris`, `miliaria`,
`photodermatosis`. That is 339 rows.

Final contribution by source:

| Source | Rows | Model A | Model B |
|---|---|---|---|
| dermnet | 17,829 | 16,020 | 17,829 |
| scin | 5,665 | 5,517 | 5,665 |
| fitzpatrick17k | 3,876 | 3,685 | 3,876 |
| padufes20 | 2,270 | 2,258 | 2,270 |
| **Total** | **29,640** | **27,480** | **29,640** |

### Stage 5 — YOLO layout

`scripts/05_layout.py` materialises the two ImageFolder trees that `yolo classify`
expects, both generated from the single `manifest_final.csv` so they cannot drift apart.

```
data/yolo/dataset_disease/{train,val,test}/<disease>/<img>.jpg       model_a == 1
data/yolo/dataset_malignancy/{train,val,test}/<malignancy>/<img>.jpg  model_b == 1
data/yolo/classes_disease.txt        27 lines
data/yolo/classes_malignancy.txt      3 lines
```

Files are **NTFS hard links** into `data/processed`, not copies. Same volume, no admin
rights needed, zero extra bytes on disk. It falls back to a real copy if linking fails.
Re-running is safe; existing links are kept.

### Stage 6 — train

`scripts/06_train.py` trains one model per invocation.

```bash
python scripts/06_train.py --task disease    --smoke   # 1 epoch on 10%, ~3 min
python scripts/06_train.py --task disease              # full run
python scripts/06_train.py --task malignancy           # full run
```

Run the two full runs **sequentially, never in parallel**. Two trainers on one 8 GB card
contend for VRAM and both slow down more than either gains.

`scripts/run_full_training_skin.cmd` runs both legs back to back, detached, against the
project-local `skin` virtual environment. It refuses to start if a trainer is already
live.

---

## 4. How the data is split

### The method

`StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)`, stratified on
disease, **grouped on `group_id`**. The 20 folds are assigned:

| Folds | Split | Share |
|---|---|---|
| 0–13 | train | 70% |
| 14–16 | val | 15% |
| 17–19 | test | 15% |

Written once, seeded, never re-randomised.

### Why grouping is mandatory

`group_id` ties together every image that belongs to the same case or lesion. SCIN
submits multiple photos per case. PAD-UFES-20 photographs the same lesion more than once.
Without grouping, two shots of one mole land on opposite sides of the train/test line,
and the model gets graded on images it has effectively already seen.

The verification line in `data/split_summary.txt` is the one that matters:

```
split: Counter({'train': 20749, 'val': 4446, 'test': 4445})  groups spanning splits: 0
```

**Zero groups span splits.** Combined with deduplication running before the split, that
closes both leakage paths.

### Realised split sizes

| Tree | Classes | train | val | test |
|---|---|---|---|---|
| `dataset_disease` | 27 | 19,239 | 4,121 | 4,120 |
| `dataset_malignancy` | 3 | 20,749 | 4,446 | 4,445 |

### Class balance, and what was done about it

The disease distribution is steep. `eczema_dermatitis` has 3,579 images;
`granulomatous_disorder` has 156. Model B is worse: `non-neoplastic` holds 22,324 of
29,640 rows.

Ultralytics' classification trainer exposes no class weights and no sampler hook, so
balancing happens at the data layer instead. `06_train.py` generates a `<dataset>_bal/`
tree whose **train split only** is oversampled with hard links:

```
replication factor = min(4.0, sqrt(max_count / count))
```

Square-root balancing lifts rare classes without drowning the common ones in near-copies,
and the 4x ceiling stops the smallest class from becoming mostly duplicates. `val/` and
`test/` are linked 1:1 and left untouched, so the evaluation set still reflects the real
distribution.

| Tree | train before | train after |
|---|---|---|
| `dataset_disease_bal` | 19,239 | 33,704 |
| `dataset_malignancy_bal` | 20,749 | 28,268 |

The realised per-class counts are printed at the start of every run, so the intent can be
verified rather than assumed. Because the extra files are hard links, the balanced disease
tree costs 2,166 MB logically and zero additional bytes on disk.

### Skin tone coverage

`fitz_scale` is the Fitzpatrick skin type, 1 (lightest) to 6 (darkest), with `-1` meaning
unknown. In `manifest_final.csv`:

| fitz_scale | Images |
|---|---|
| unknown (-1) | 18,777 |
| 1 | 1,015 |
| 2 | 3,126 |
| 3 | 2,965 |
| 4 | 2,124 |
| 5 | 1,199 |
| 6 | 434 |

Of the 10,863 images with a known type, types 5–6 are 1,633, about 15%. That is a live
fairness risk, not a hypothetical one, and it is why the evaluation stage is specified to
break every malignancy metric out by skin-type bucket.

---

## 5. Training configuration

Both models fine-tune `yolo11s-cls.pt`.

| Setting | Model A (disease) | Model B (malignancy) |
|---|---|---|
| Epochs | 60 | 40 |
| Early-stop patience | 12 | 10 |
| Image size | 320 | 320 |
| Batch | 64 | 64 |
| Dataloader workers | 4 | 4 |
| LR schedule | cosine | cosine |
| Seed | 42 | 42 |

### Augmentation, and one deliberate omission

| Parameter | Value |
|---|---|
| `hsv_h` | 0.015 |
| `hsv_s` | 0.4 |
| `hsv_v` | 0.3 |
| `fliplr` | 0.5 |
| `flipud` | 0.5 |
| `scale` | 0.4 |
| `erasing` | 0.2 |
| `auto_augment` | **off** |

RandAugment is switched off on purpose. Hue and saturation carry real diagnostic signal
in dermatology — erythema, pigmentation, vascularity — and they also encode skin tone.
RandAugment includes Solarize and Posterize and other aggressive colour operations that
would destroy both. Colour jitter is therefore kept mild while geometric augmentation,
which costs nothing diagnostically, runs freely.

---

## 6. Outputs

Every training run writes to `runs/<name>/`:

| File | Contents |
|---|---|
| `weights/best.pt`, `weights/last.pt` | Model checkpoints |
| `results.csv` | Per-epoch top-1, top-5, train loss, val loss, learning rates |
| `results.png` | Those curves plotted |
| `confusion_matrix.png`, `confusion_matrix_normalized.png` | On the val split |
| `test_metrics.json` | Held-out test top-1 and top-5, plus the class list |
| `args.yaml` | Every hyperparameter, for reproducing the run |
| `train_batch*.jpg`, `val_batch*_labels.jpg`, `val_batch*_pred.jpg` | Sample images, for eyeballing that augmentation is doing what was intended |

Held-out test evaluation lands in a sibling directory, `runs/<name>_test/`.

Logs live in `logs/`: `06_train_disease.log`, `06_train_malignancy.log`,
`06_train_progress.log`, and the completion marker `06_train_ALL_DONE.flag`.

---

## 7. Environment

Training runs from a project-local uv virtual environment at `skin/`, Python 3.12.

```bash
uv venv skin --python 3.12
uv pip install --python skin\Scripts\python.exe --torch-backend=auto -r requirements-skin.txt
```

`requirements-skin.txt` pins 68 packages. The load-bearing ones are torch 2.14.0+cu132,
torchvision 0.29.0+cu132 and ultralytics 8.4.156.

The `--torch-backend=auto` flag matters. Without it uv resolves CPU-only torch and
training silently falls off the GPU.

Two launchers exist. `scripts/run_full_training.cmd` is the original and calls a bare
`python`, meaning whatever is on PATH. `scripts/run_full_training_skin.cmd` calls the
`skin` interpreter explicitly and refuses to double-launch. **Prefer the second one.**

One Windows caveat: never edit a `.cmd` file while `cmd.exe` is executing it. The shell
seeks the batch file by byte offset between lines, so an in-flight edit shifts that
offset and runs a corrupted line. Write a new file and switch on the next run.

---

## 8. Results

Both models trained on 2026-09-20 on the RTX 4060 Laptop. Numbers below are on the
held-out test split, which neither model saw during training or early stopping.

### Headline

| | Model A (disease) | Model B (malignancy) |
|---|---|---|
| Classes | 27 | 3 |
| Epochs run | 60 of 60 | 40 of 40 |
| Wall clock | 3 h 45 m | 2 h 15 m |
| Best val top-1 | 0.6554 (epoch 49) | 0.9199 (epoch 34) |
| **Test top-1** | **0.6364** | **0.9201** |
| **Test macro-F1** | **0.6275** | **0.8453** |
| Test images | 4,120 | 4,445 |

Neither model early-stopped. Both ran their full schedule, so patience never triggered.

**Ignore Model B's top-5 of 1.000.** With three classes, the top five always contain the
right answer. The number is arithmetically forced and carries no information.

### What the macro-F1 tells you

For Model A, macro-F1 of 0.6275 sits only 0.009 below top-1 of 0.6364. That gap is the
interesting number, not either figure alone.

Macro-F1 weights a 24-image class the same as a 539-image one, so it collapses when a
model quietly stops predicting rare classes and banks accuracy on common ones. It did not
collapse here. The square-root oversampling described in section 4 did what it was meant
to do: the model is not ignoring the tail.

The val-to-test gap is also small, 0.6554 to 0.6364. The model generalises to held-out
data about as well as it fits the data it tuned on.

### Model A, per-class on test

Best five by F1:

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| nail_disease | 0.907 | 0.942 | 0.924 | 155 |
| hair_disease | 0.840 | 0.840 | 0.840 | 25 |
| seborrheic_keratosis_benign_tumor | 0.778 | 0.788 | 0.783 | 330 |
| basal_cell_carcinoma | 0.779 | 0.774 | 0.777 | 164 |
| acne_rosacea | 0.742 | 0.738 | 0.740 | 164 |

Worst five by F1:

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| contact_dermatitis | 0.278 | 0.313 | 0.295 | 166 |
| bacterial_infection | 0.380 | 0.338 | 0.358 | 145 |
| trauma_scar_bruise | 0.481 | 0.394 | 0.433 | 33 |
| exanthem_drug_eruption | 0.494 | 0.389 | 0.436 | 113 |
| granulomatous_disorder | 0.526 | 0.417 | 0.465 | 24 |

The weak classes are not the rare ones. `contact_dermatitis` has 166 test images and is
the worst class in the set, while `hair_disease` has 25 and sits near the top. The
failures track *visual separability*, not sample count. Contact dermatitis, eczema and
bacterial infection look alike in a photograph, and the label taxonomy folds 16 raw
labels into `eczema_dermatitis` alone. Adding images will not fix this; the classes
genuinely overlap.

### Melanoma

| Metric | Value |
|---|---|
| Recall | 0.688 (22 of 32) |
| Precision | 0.688 |
| Test images | 32 |

Where the 10 misses went: 4 to `melanocytic_nevus`, 3 to
`seborrheic_keratosis_benign_tumor`, 1 to `basal_cell_carcinoma`, 1 to
`squamous_cell_carcinoma`, 1 elsewhere.

That split matters more than the recall figure. Seven misses landed on **benign** classes,
which is the dangerous direction and exactly the `melanoma` to `melanocytic_nevus`
confusion the plan warned about. Two landed on other **malignant** classes, where the OR
rule still raises the cancer flag, so those are caught despite the wrong name.

**32 test images is a very small sample.** The difference between 22 and 23 correct is 3
percentage points of recall. Treat this number as directional, not precise.

### Model B, per-class on test

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| non-neoplastic | 0.945 | 0.974 | 0.959 | 3,349 |
| malignant | 0.860 | 0.809 | 0.834 | 593 |
| benign | 0.802 | 0.692 | 0.743 | 503 |

**Recall on `malignant` is 0.809.** This is the number the project lives or dies by, and
it means roughly one malignant lesion in five is missed by Model B alone, before the OR
rule adds Model A's contribution.

`benign` is the weakest class at 0.692 recall. Benign neoplasms get confused with
non-neoplastic conditions, which is the least costly of the available error modes.

### Training conditions, for the record

The disease run took 3 h 45 m against an estimated 1-1.5 h. Three things slowed it, none
of which affected the results:

- The GPU ran power-capped the whole time at 2,505 MHz against a 3,105 MHz maximum, under
  the ASUS Armoury Crate "Silent" operating mode.
- A mains power cut put the laptop on battery mid-run. Power returned before the battery
  was exhausted and no epoch was lost.
- The real bottleneck was the dataloader. With `workers: 4` on a 20-core machine, four
  worker processes sat pegged at ~97% of one core each while roughly 15 cores idled, and
  the GPU averaged 25% utilisation waiting for data. **Raising `--workers` is the single
  biggest speed-up available for future runs.**

### One bug found

`scripts/06_train.py:153` crashed with `RuntimeError: CUDA error: out of memory` on the
disease run, after training had completed successfully. The line builds a second `YOLO`
object for the held-out test pass while the trained model is still resident on the 8 GB
card.

Training was unaffected and `best.pt` is intact, but `test_metrics.json` was never
written and the script exited 1. The test pass was re-run standalone once the GPU was
idle, and the disease figures above come from that re-run. The malignancy run hit no such
problem.

**This should be fixed before the next training run**, by releasing the trained model or
running the test pass in a separate process.

---

## 9. Known gaps

**`scripts/07_eval.py` still does not exist.** The figures in section 8 were computed
ad hoc after training and are not reproducible by any committed script. Re-deriving them
means re-running throwaway code.

What section 8 now covers: macro-F1 on both models, per-class precision and recall,
melanoma recall, and where the melanoma misses land. What is still missing entirely:

- **Sensitivity at a fixed 90% specificity, and the full ROC curve.** The default 0.5 threshold is almost certainly wrong for a screening tool. The operating threshold should be chosen on val and frozen before test is touched. Nothing has been calibrated, so any application built on these models is currently using an arbitrary threshold.
- **Every Model B metric broken out by `fitz_scale` bucket** (1–2, 3–4, 5–6). A model at 85% malignant recall on types 1–2 and 60% on types 5–6 is not an 80% model. It is a model that fails the people already worst served by dermatology. This is the largest remaining unknown.
- **Image-by-image reading of the melanoma misses.** Seven of the ten landed on benign classes. Which seven, and whether they share a visual property, is unexamined.

`PLAN_EVAL_AND_LIVE_CAMERA.md` specifies this work as Stage 1 and has not been executed.

**Melanoma is thin.** 214 images total, 149 in train and 32 in test. Its measured recall
of 0.688 rests on those 32 images, so the confidence interval is wide. This is the class
where a miss costs the most and the one the dataset supports least.

**Fitzpatrick17k is at 23.4% coverage.** If the maintainers grant the full archive,
re-running Stage 1 onward would add roughly 573 melanoma-family images.

**Folder 11 labels come from a layperson.** They should be cross-checked against Model B's
predictions before being trusted.

**Stage 8, export, has not been run.** When it is, ship `label_map.csv`, the class-index
ordering, the frozen decision threshold and the test metrics alongside the weights.
Weights without the label ordering are unusable, and that is a routine way to lose a week.
