# Raw Data, Sources, and Processing — Full Walkthrough

Audience: a project partner who has never touched this repo.
Goal: after reading this you should know **what the raw data is, where every image came
from, why those sources (and not others), and exactly what each processing stage does and
why**.

Everything here is reproducible from `scripts/00a` → `scripts/05` in order.

---

## 0. One-paragraph summary

We are training two YOLO11 **classification** models on ordinary (non-dermoscopic) photos
of skin:

- **Model A — disease**: 27 classes (eczema, psoriasis, BCC, melanoma, …)
- **Model B — malignancy**: 3 classes (`benign` / `malignant` / `non-neoplastic`)

Four public datasets were downloaded (**36,108 raw images**), merged into one manifest,
label-harmonised into a single taxonomy, de-duplicated by perceptual hash, filtered, and
split 70/15/15 with patient/case grouping. The result is **29,640 cleaned images** in
`data/processed/`, described row-by-row in `data/manifest_final.csv`.

---

## 1. The raw data

### 1.1 What was downloaded

| Source | Origin | Raw images | Survives to final | What it brings |
|---|---|---|---|---|
| **DermNet** | Hugging Face `exper1ment/dermnet` | 19,559 | ~17.8k | The backbone. 23 clinically coherent condition folders. |
| **SCIN** (Google) | public GCS bucket `gs://dx-scin-public-data` | 10,364 | ~5.7k | Real **smartphone** photos from consumers + dermatologist label weights. |
| **Fitzpatrick17k** | scraped from `atlasdermatologico.com.br` | 3,887 | ~3.9k | **Skin-tone (Fitzpatrick 1–6) labels** — our only fairness axis. |
| **PAD-UFES-20** | Mendeley `zr7vgbcyr2`, manual browser download | 2,298 | ~2.3k | Phone photos of **cancer** (BCC, SCC, MEL, ACK) + patient metadata. |
| **Total** | | **36,108** | **29,640** | |

Links:
- DermNet — <https://huggingface.co/datasets/exper1ment/dermnet>
- SCIN — <https://github.com/google-research-datasets/scin>
- Fitzpatrick17k — <https://github.com/mattgroh/fitzpatrick17k>
- PAD-UFES-20 — <https://data.mendeley.com/datasets/zr7vgbcyr2/1>

### 1.2 Why these four, and why not the obvious ones

**The product constraint drives everything**: the model must run on a **phone camera /
webcam** photo of skin. So every image in training must *look like* a phone photo.

- **ISIC / HAM10000 / PH2 deliberately excluded.** Those are **dermoscopy** — shot
  through a contact lens with polarised light. They are the biggest, cleanest skin-cancer
  sets in existence, and they are *wrong for us*: a model trained on mixed dermoscopy +
  phone photos learns **the camera**, not the disease. The dermoscopy border is a
  shortcut feature that correlates perfectly with "this was a cancer-clinic image", and
  the model will take it. Accuracy would look great and collapse in deployment.
- **DermNet** → chosen as the backbone because it is the largest clinical-photo set we
  can legally pull, and its folder structure is already a sane coarse taxonomy, so it can
  *define* the class list instead of being forced into someone else's.
- **SCIN** → chosen because it is the only large set of genuinely consumer-captured
  photos (bad lighting, odd angles, hair, clothing in frame). This is the distribution
  our users actually produce. It also ships per-case **dermatologist agreement weights**,
  so we can filter weak labels instead of trusting all of them.
- **PAD-UFES-20** → chosen to fix the malignant tail. Clinical photo sets are
  overwhelmingly benign/inflammatory; PAD-UFES-20 is biopsy-confirmed cancer shot on
  phones, which is exactly the class we are most afraid of getting wrong.
- **Fitzpatrick17k** → chosen because without it we have **no skin-tone metadata at all**
  and cannot even *measure* whether the model fails on dark skin. Only ~23.4% of the
  original 16,577 URLs still resolve (3,887 of 3,905 reachable `atlasdermatologico` URLs
  were recovered, 99.5%); `dermaamin.com` is dead. We took what survives — the point is
  the `fitz_scale` column, not the volume.

**Known limitation to state out loud:** `fitz_scale` is `-1` (unknown) for 18,777 of
29,640 final images. We can audit tone-wise performance on the Fitzpatrick subset only.

### 1.3 What the raw images physically look like

Short-side pixels, measured at manifest time:

| Source | min | p10 | median | max |
|---|---|---|---|---|
| dermnet | 314 | 469 | 477 | 720 |
| scin | 102 | 504 | **810** | 1080 |
| padufes20 | 147 | 372 | 780 | 3474 |
| fitzpatrick17k | 177 | 355 | 376 | 1123 |

So: DermNet is uniformly small and web-sized, SCIN is big phone-camera output, PAD-UFES
is mixed, Fitzpatrick is small atlas thumbnails. This spread is why Stage 4 normalises
everything to a **512 px long edge** — training at a fixed size anyway, and keeping
3474 px originals in the training tree only costs disk and decode time.

### 1.4 Raw layout on disk

```
data/raw/
  dermnet/{train,test}/<23 class folders>/*.jpg   + _manifest.csv
  scin/images/*.png  + scin_cases.csv, scin_labels.csv, _manifest.csv
  pad_ufes_20/imgs_part_{1,2,3}/*.png + metadata.csv
  fitzpatrick17k/images/<md5hash>.jpg + _manifest.csv, _failed.csv, _rejected/
  fitzpatrick17k.csv            (official 16,577-row label file)
data/raw.rar                    (18.8 GB archive of the above — the cold backup)
```

`data/raw/` is **never modified** after download. Every later stage writes somewhere else.

### 1.5 How each source was fetched (and the integrity rule)

| Script | Source | Integrity check | Notes |
|---|---|---|---|
| `00a_fetch_dermnet.py` | HF `datasets` | SHA1 per file | Exports the parquet dataset to an ImageFolder tree. Resume-safe. |
| `00b_fetch_scin.py` | GCS JSON list API over plain HTTPS, 8 threads | every object verified against the **md5Hash GCS reports** | No auth needed; bucket is public. |
| `00c_fetch_fitz_atlas.py` | direct HTTP scrape | each download verified against the **md5hash in the official CSV** | Throttled to **1 request/second** — small academic host, do not hammer it. Mismatches go to `_rejected/`. |
| (manual) | PAD-UFES-20 | — | Mendeley blocks scripted download; pulled through a browser. |

Every fetcher is **resume-safe**: an existing file with a matching hash is skipped without
a request. Rerunning after a crash costs nothing.

---

## 2. Processing pipeline

Run order; each stage reads the previous stage's CSV and writes its own.

```
00a/00b/00c  fetch           -> data/raw/**
01_manifest                  -> manifest_raw.csv      (36,108 rows)
02a/02b      folder-11 sort  -> dermnet_folder11_labels.csv
02c_build_label_map          -> label_map.csv         (330 raw labels -> 27 classes)
03_dedup                     -> manifest_dedup.csv    (34,550 rows)
04_split                     -> manifest_final.csv + data/processed/**  (29,640 rows)
05_layout                    -> data/yolo/dataset_{disease,malignancy}/
06_train                     -> data/yolo/*_bal/ + runs/
```

### Stage 1 — unified manifest (`01_manifest.py`)

Four sources with four different metadata shapes become **one CSV, one row per image**.

Columns: `path, source, raw_label, group_id, fitz_scale, width, height, bytes,
split_hint, case_id, label_weight, label_dict, patient_id, lesion_id, md5hash,
three_partition_label, nine_partition_label`.

Why this exists: every downstream decision (dedup, label map, split, filter) is a
transformation on **one table**, not on four folder trees. That makes every drop auditable
and every stage re-runnable.

Key decision — **`group_id`**: the leak-prevention key.
- PAD-UFES-20 → `patient_id` / `lesion_id` (multiple photos of the same lesion exist)
- SCIN → `case_id` (one case = several photos of the same rash)
- DermNet / Fitzpatrick → the image's own hash (no patient info available)

SCIN filtering at this stage:
- 1,972 unlabelled cases and 837 low-agreement cases (`weight < 0.5`) are marked; only
  **strong-label** images (4,578) are trusted downstream.
- 42 rows (15 files) shared between cases dropped — a file that belongs to two cases
  cannot be grouped safely.

The script **exits non-zero** if any hard count check fails (DermNet == 19,559,
PAD-UFES == 2,298 with every `img_id` resolving, Fitzpatrick recovery ≥ 95% of reachable
URLs). The pipeline is not allowed to proceed on silently truncated data.

Also emits `label_map_skeleton.csv`: all **330 distinct `(source, raw_label)` pairs** with
counts — the starting point for Stage 2c.

### Stage 2a/2b — hand-sorting DermNet folder 11

DermNet's folder *"Melanoma Skin Cancer Nevi and Moles"* mixes **melanoma** (malignant)
and **nevi/moles** (benign) in one folder. Mapping the whole folder either way poisons
Model B on the single most important distinction in the project.

So it was labelled **per image**:
- `02a_sort_folder11.py` — tiny local web app on `127.0.0.1:8765`, one image at a time,
  keys `M` / `N` / `U`(unsure). Appends to CSV on every keypress → fully resumable.
- `02b_contact_sheets.py` — 3×2 contact sheets at 480 px for faster bulk review; labels
  can be typed per sheet (`label 12 "M N U N N M"`).

Output: `data/dermnet_folder11_labels.csv` (`path, label, annotator, ts`).

**Honesty flag, keep it in the paper:** the annotator is recorded as `lay-user`, not a
dermatologist. `unsure` images are **dropped from both models** rather than guessed.

### Stage 2c — the label map (`02c_build_label_map.py`)

330 raw labels → **27 disease classes** + a malignancy class.

Design decisions, each with a reason:

1. **The map is code, not a spreadsheet.** A dict in a `.py` file is diffable,
   reviewable in PR, and re-runnable. A hand-edited XLSX is none of those.
2. **DermNet's 23 groups are the backbone**, because it is the largest source and its
   folders are already clinically coherent. Other sources fold into those groups.
3. **A specific label gets its own class only if it is both more specific AND big
   enough**: `basal_cell_carcinoma`, `squamous_cell_carcinoma`, `actinic_keratosis`,
   `melanoma`, `melanocytic_nevus`. This is why we end at 27, not 23.
4. **Malignancy is derived from the canonical disease, never taken from the source.**
   Consequence: the same condition can never carry two different malignancy labels
   depending on which dataset it came from. This is the single most important
   consistency rule in the project.
5. **Per-image exceptions** (resolved by image, not by label):
   - folder 11 → from the hand-label CSV; `unsure` → excluded from both models.
   - *"Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions"* → every
     image is malignant, but **which** malignancy is unknown → `model_b` only, excluded
     from Model A.
6. **SCIN's 164-label tail (804 images) is excluded, not defaulted.** A blanket
   "non-neoplastic" default would be *wrong* for the few neoplastic diagnoses hiding in
   that tail — and a wrong malignancy label is more costly than a missing image.

Output `label_map.csv`: `source, raw_label, n_images, disease, malignancy, model_a,
model_b, note`.

### Stage 3 — perceptual-hash dedup (`03_dedup.py`)

Public skin datasets re-host each other's images. The same photo appearing in train *and*
test would silently inflate every metric we report.

Method:
- **64-bit pHash** (`imagehash.phash`, `hash_size=8`), cached in `phash.csv` so re-runs
  are free.
- Candidate pairs via **4-band LSH** on 16-bit slices (share any band → candidate), then
  exact Hamming verification, then **union-find** merge.
- Threshold: **Hamming ≤ 5** — catches re-compression, resize and minor crops without
  collapsing genuinely different lesions.
- Within a group the **highest-resolution** member is kept.
- Chained groups with diameter > 5 are dissolved (4 of them) — transitivity would
  otherwise merge unrelated images.

Results: 120,479 candidate pairs → 1,739 duplicate pairs → **1,431 groups / 2,989 images**;
**1,558 dropped** (dermnet 1,522, scin 16, padufes20 16, fitz 4) → **34,550 rows**.
1 cross-source group; **495 groups whose members carry different labels** — kept for
Stage 4 to adjudicate.

**It runs before the split, by design.** Dedup-after-split cannot fix a leak that already
happened.

Audit trail: `dedup_groups.csv` lists every group member with its phash and a `kept` flag.

### Stage 4 — labels, filters, split, resize (`04_split.py`)

Applied strictly in this order:

1. **Apply `label_map`** → `disease, malignancy, model_a, model_b`; folder 11 resolved
   per image. Not mapped / `unsure` → dropped (**4,848 rows**).
2. **Duplicate-group conflicts** (from Stage 3's 495 disagreeing groups):
   - members' diseases differ → `model_a = 0` (**448 rows**, kept for Model B)
   - members' **malignancies** differ → **drop the row** (**33 rows**). If two sources
     disagree on benign vs malignant, neither label is trustworthy.
3. **Size filter**: short side < **200 px** → drop (**29 rows**). Below that, upscaling to
   512 is inventing pixels.
4. **Rare-class floor**: Model A classes with < **100** images → `model_a = 0`, row stays
   for Model B. Dropped: `cutaneous_lymphoma, kaposi_sarcoma, keratosis_pilaris,
   miliaria, photodermatosis` (339 rows). A 20-image class produces a meaningless
   per-class metric.
5. **Split** — `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)`,
   stratified on `disease`, **grouped on `group_id`**. Folds 0–13 → train, 14–16 → val,
   17–19 → test = **70 / 15 / 15**.
   - Stratified so rare classes exist in all three splits.
   - Grouped so the same patient/case/lesion never straddles splits.
   - Verified: `groups spanning splits: 0`.
   - Seed fixed and the split is **written into `manifest_final.csv` once, never
     re-randomised** — every experiment compares on the identical test set.
6. **Resize + write** → `data/processed/`, long edge **512 px**, **JPEG q92**, aspect
   preserved. 29,640 images, 0 errors, ~54 s on 12 workers.

Final: **29,640 rows** → train 20,749 / val 4,446 / test 4,445.

Drop ledger (every dropped image is accounted for):
```
excluded_by_label_map_or_unsure   4,848
dup_group_malignancy_conflict        33
short_side_lt_200                    29
```

`manifest_final.csv` is the **single source of truth**: `path, raw_path, source, disease,
malignancy, model_a, model_b, split, fold, group_id, fitz_scale, width, height,
raw_label, case_id, patient_id, lesion_id, label_weight, md5hash, dup_group`.

### Stage 5 — materialise the two training trees (`05_layout.py`)

`yolo classify` wants `root/{train,val,test}/<class>/*.jpg`, so:

| Tree | Rows | Classes | train | val | test |
|---|---|---|---|---|---|
| `yolo/dataset_disease` (Model A) | `model_a == 1` | 27 | 19,239 | 4,121 | 4,120 |
| `yolo/dataset_malignancy` (Model B) | `model_b == 1` | 3 | 20,749 | 4,446 | 4,445 |

Both trees are generated from the **one manifest**, so they cannot drift apart. Files are
**NTFS hard links** into `data/processed` — same volume, no admin rights, **zero extra
bytes**. Falls back to a copy if linking fails. Re-running keeps existing links.

### Stage 6 — class balancing (`06_train.py --task ...`)

Model A is heavily skewed: `eczema_dermatitis` 3,579 vs `granulomatous_disorder` 156.
Model B: non-neoplastic 22,324 vs benign 3,342.

Balancing is done **at the data layer**, producing `dataset_<task>_bal/`:
- repetition factor `rep = min(MAX_REP, sqrt(max_class / n))` — **square-root**
  oversampling, which lifts rare classes without drowning the frequent ones in
  near-identical copies (full inverse-frequency would duplicate melanoma ~24×).
- duplicates are hard links named `<stem>__dup<k>.jpg` → zero extra bytes.
- **only `train/` is oversampled.** `val/` and `test/` are copied 1:1 from the unbalanced
  tree, so reported metrics are on the true distribution.

---

## 3. Final dataset at a glance

**Model A — 27 disease classes, 27,480 images**

Largest → smallest: eczema_dermatitis 3,579 · psoriasis_lichen_planus 2,566 ·
seborrheic_keratosis_benign_tumor 2,190 · fungal_infection 1,836 · infestation_bite 1,456 ·
viral_warts_molluscum 1,234 · systemic_genetic_disease 1,185 · contact_dermatitis 1,106 ·
basal_cell_carcinoma 1,088 · acne_rosacea 1,077 · nail_disease 1,032 ·
bacterial_infection 976 · vasculitis_purpura 857 · pigmentation_disorder 817 ·
actinic_keratosis 767 · exanthem_drug_eruption 758 · urticaria 736 ·
lupus_connective_tissue 673 · bullous_disease 636 · vascular_tumor 629 · herpes_std 594 ·
melanocytic_nevus 523 · squamous_cell_carcinoma 394 · melanoma 214 ·
trauma_scar_bruise 211 · hair_disease 190 · granulomatous_disorder 156

**Model B — malignancy, 29,640 images**

| class | train | val | test | total |
|---|---|---|---|---|
| non-neoplastic | 15,626 | 3,349 | 3,349 | 22,324 |
| malignant | 2,784 | 597 | 593 | 3,974 |
| benign | 2,339 | 500 | 503 | 3,342 |

Melanoma alone: train 149 / val 33 / test 32.

**Fitzpatrick coverage (final set):** unknown 18,777 · I 1,015 · II 3,126 · III 2,965 ·
IV 2,124 · V 1,199 · VI 434.

### Risks a reviewer will ask about — state them first

- **Melanoma is tiny (214 images, 32 in test).** Any melanoma recall number has a huge
  confidence interval. Report CIs, do not report a point estimate alone.
- **Folder-11 labels are lay-annotated.** Cross-check against Model B predictions; treat
  disagreements as a review queue.
- **63% of images have no skin-tone label.** Fairness analysis is limited to the
  Fitzpatrick subset.
- **Source ↔ class correlation.** PAD-UFES is nearly all cancer, SCIN is nearly all
  inflammatory. The model can partly cheat on image style. Worth a per-source breakdown
  of the test metrics.
- **No dermoscopy by design** — so do not compare our numbers against ISIC leaderboards;
  different task, different input distribution.

---

## 4. File map

| Path | What it is |
|---|---|
| `data/raw/` | untouched downloads (+ `data/raw.rar`, 18.8 GB cold backup) |
| `data/processed/` | 29,640 cleaned images, long edge 512, JPEG q92 |
| `data/manifest_raw.csv` | 36,108 rows — every raw image, unified schema |
| `data/manifest_dedup.csv` | 34,550 rows — after pHash dedup |
| `data/manifest_final.csv` | **29,640 rows — source of truth** (labels + split + metadata) |
| `data/label_map_skeleton.csv` | the 330 raw `(source, label)` pairs with counts |
| `data/label_map.csv` | raw label → disease / malignancy / model_a / model_b |
| `data/dermnet_folder11_labels.csv` | per-image melanoma/nevus/unsure hand labels |
| `data/phash.csv` | pHash cache, one row per raw image |
| `data/dedup_groups.csv` | every duplicate group member + `kept` flag (audit) |
| `data/*_summary.txt` | printed stats from stages 1, 3, 4 |
| `data/work/folder11_sheets/` | contact sheets used for the manual sort |
| `data/yolo/dataset_{disease,malignancy}/` | ImageFolder trees for training |
| `data/yolo/*_bal/` | sqrt-oversampled train split (hard links) |
| `data/yolo/classes_*.txt` | fixed class order |
| `logs/` | fetch logs per stage |

### Reproducing on another machine

```bash
pip install -r requirements-skin.txt
python scripts/00a_fetch_dermnet.py
python scripts/00b_fetch_scin.py
python scripts/00c_fetch_fitz_atlas.py      # PAD-UFES-20: download manually from Mendeley
python scripts/01_manifest.py
python scripts/02c_build_label_map.py       # 02a/02b only if re-doing the folder-11 sort
python scripts/03_dedup.py
python scripts/04_split.py
python scripts/05_layout.py
python scripts/06_train.py --task disease
python scripts/06_train.py --task malignancy
```

**Shortcut if you just want the data:** copy `data/processed/`, `manifest_final.csv` and
`label_map.csv`, then run `scripts/05_layout.py`. That rebuilds `data/yolo/` exactly.
Do **not** zip `data/yolo/` — the hard links become real copies and the archive balloons.
`*_bal/` is regenerated by `06_train.py`, never ship it.
