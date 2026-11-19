# Skin Disease + Malignancy Classifier — Pipeline Plan

**Target platform:** local workstation (RTX 4060 Laptop) — primary; Kaggle Notebooks as fallback. See §4.
**Goal:** From a single clinical/smartphone photo of a skin lesion, predict
1. **What condition is it?** (disease name, multi-class)
2. **Is it cancer?** (`benign` / `malignant` / `non-neoplastic`)

**Status:** planning complete, data acquisition pending Kaggle credentials (§4.5).
**Last updated:** 2026-09-20

---

## 1. Dataset Selection (Option A — clinical / smartphone domain)

All three sources are **clinical photographs**, not dermoscopy. This is deliberate: the
deployment target is a phone camera, so every training image must come from the same
optical domain. Mixing in dermoscopy data (ISIC, HAM10000) is explicitly rejected — see
§3.2.

| Source | Images | Disease classes | Malignancy label | Kaggle slug |
|---|---|---|---|---|
| Fitzpatrick17k | 16,577 | 114 | Yes, 3-way (native column) | `nazmussadat013/fitzpatrick-17k-dataset` |
| DermNet | ~19,500 | 23 | Derive via manual map | `shubhamgoel27/dermnet` |
| PAD-UFES-20 | 2,298 | 6 | Derive via map (3 are cancer) | `mahdavi1202/skin-cancer` |
| **Total** | **~38,375** | — | — | — |

Mirror alternatives if a slug goes dead:
- Fitzpatrick17k — `nazmusresan/fitzpatrick17k`, `mobaswiralfarabi/fitzpatrick17k-original`
- PAD-UFES-20 — `orvile/pad-ufes-20`, `moeinmadadi/pad-ufes-20`

### 1.1 Verified Fitzpatrick17k label structure

The label CSV was downloaded from GitHub and analysed. Confirmed contents
(`data/raw/fitzpatrick17k.csv`, 16,577 rows):

Columns: `md5hash, fitzpatrick_scale, fitzpatrick_centaur, label,
nine_partition_label, three_partition_label, qc, url, url_alphanum`

`three_partition_label` — this is the malignancy target, available for free:

```
non-neoplastic   12,080
malignant         2,263
benign            2,234
```

`nine_partition_label` — a useful intermediate granularity:

```
inflammatory                   10,886
malignant epidermal             1,352
genodermatoses                  1,194
benign dermal                   1,067
benign epidermal                  931
malignant melanoma                573
benign melanocyte                 236
malignant cutaneous lymphoma      182
malignant dermal                  156
```

`label` — 114 conditions, between 53 and 653 images each.
60 classes have >= 100 images; all 114 have >= 50.

The 11 malignant conditions:

```
squamous cell carcinoma               581
basal cell carcinoma                  468
melanoma                              261
mycosis fungoides                     182
actinic keratosis                     175
kaposi sarcoma                        156
superficial spreading melanoma ssm    118
malignant melanoma                    111
lentigo maligna                        83
solid cystic basal cell carcinoma      66
basal cell carcinoma morpheiform       62
```

Note the taxonomy is messy on purpose: `melanoma`, `malignant melanoma`,
`superficial spreading melanoma ssm` and `lentigo maligna` are four separate labels
for overlapping concepts. Same for the three basal cell carcinoma variants. These must
be consolidated in Stage 2.

Skin tone coverage (`fitzpatrick_scale`, `-1` = unlabelled):

```
-1   565
 1  2,947
 2  4,808
 3  3,308
 4  2,781
 5  1,533
 6    635
```

Types 5-6 are under-represented (13% combined). Report per-skin-tone metrics
separately (§7.3) rather than pretending the headline number holds for everyone.

---

## 2. Blocking Finding: Do Not Scrape Fitzpatrick17k From Source

The official repo ships **URLs only, not images**. Those URLs were tested:

| Host | Images | Status |
|---|---|---|
| `www.dermaamin.com` | 12,631 (76%) | **DNS SERVFAIL — domain does not resolve.** Unrecoverable. |
| `atlasdermatologico.com.br` | 3,905 (24%) | Alive, `image/jpeg`, ~50-95 KB each |
| (blank / malformed) | 41 | N/A |

```
$ nslookup www.dermaamin.com
*** dns.google can't find www.dermaamin.com: Server failed
```

**Consequence:** the download script in the official repo now recovers at most 24% of
the dataset. The Kaggle mirrors are the only practical source for the full 16,577
images. Treat the GitHub CSV as the *label authority* and the Kaggle mirror as the
*image store*, then join on `md5hash`.

**Verify the mirror before trusting it.** Mirrors are user uploads with no guarantee of
completeness. Stage 1 must assert the join covers the expected row count and report the
shortfall rather than silently training on a subset.

---

## 3. Design Decisions

### 3.1 Two outputs, two models

YOLO's classification mode (`yolo classify`) supports exactly one label per image, so a
single YOLO-cls model cannot emit both heads. Two options were considered:

- **Derive malignancy from the predicted disease** via a lookup table. Simple, but the
  malignancy answer inherits every disease-level error, and a lesion the model has
  never seen collapses to whatever class is nearest.
- **Train a second, dedicated malignancy model.** Costs one more training run but
  optimises directly for the output that actually carries clinical weight.

**Decision: do both, and combine them.**

- **Model A — disease classifier.** N classes (see §5.2). Answers "what is it".
- **Model B — malignancy classifier.** 3 classes: `benign` / `malignant` /
  `non-neoplastic`. Trained on the full merged set, so every one of the ~38k images
  contributes, including the ones whose disease class was dropped for being too rare.

Final decision rule:

```
malignant_flag = (Model B says malignant)
              OR (lookup[Model A top-1] == malignant)
```

An OR, not an average. A false positive costs the user a needless dermatologist visit;
a false negative costs them a missed cancer. Bias toward recall on the malignant class
and say so plainly in the UI.

### 3.2 Why no dermoscopy data

Merging ISIC/HAM10000 into this set is tempting — it would push the count past 60k and
add thousands of confirmed cancers. It must not be done here.

Dermoscopy images are captured through a contact lens with polarised light at fixed
magnification. Clinical photos are not. If nearly all malignant examples arrive from
dermoscopy and nearly all benign examples from clinical photos, the cheapest separating
feature is *capture modality*, not pathology. The model learns to detect lens vignetting
and reports a high validation accuracy that evaporates on real phone input. This is
textbook shortcut learning, and it is invisible unless the test set is domain-matched.

PAD-UFES-20 is in the mix specifically to prevent a milder version of the same failure:
it supplies malignant lesions **photographed on smartphones**, so "malignant" is not
perfectly correlated with any single source.

### 3.3 Grouped splitting is mandatory

PAD-UFES-20 contains 2,298 images of 1,641 lesions from 1,373 patients — multiple
photographs of the same lesion. A random split puts two shots of one mole in train and
val, and the reported accuracy becomes partly a memorisation score.

Split on `patient_id` using `sklearn.model_selection.StratifiedGroupKFold`. For the
other two sources, group on `md5hash` / file hash after deduplication.

---

## 4. Execution Environment

### 4.1 Local hardware (verified 2026-09-20)

```
GPU    NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB VRAM, driver 616.92
CPU    12th Gen Intel Core i7-12700H, 14 cores / 20 threads
RAM    15.6 GB
Disk   D: 138 GB free (project drive), C: 169 GB free
Python 3.12.2
torch / ultralytics — not installed yet
```

**Fit assessment.** 8 GB VRAM is comfortable for this workload. `yolo11s-cls`
(~6M params) at `imgsz=320 batch=64` uses roughly 3 GB with AMP on. `yolo11m-cls`
also fits at `batch=32`. Throughput is in the same band as a Kaggle T4 and ahead of a
P100.

Expected wall-clock on this machine, ~38k images at 320 px:

| Job | Estimate |
|---|---|
| pHash dedup, 38k images (CPU) | ~15 min |
| Resize pass (CPU) | ~20 min |
| Model A, 60 epochs | 1–1.5 h |
| Model B, 40 epochs | ~45 min |

Two local constraints worth respecting:

- **16 GB RAM.** Windows spawns dataloader workers as separate processes, each holding
  its own copy of the dataset index. Keep `workers=4`. Going to 8 will start swapping.
- **Laptop thermals.** Sustained full GPU load for 90 minutes will throttle on most
  laptop chassis. Run on mains power with a cooling pad if available. Throttling slows
  training roughly 2x; it does not make it fail.

### 4.2 Three ways to run the pipeline

All three share the same code. Only paths and the launch command differ, so switching
between them is a ten-minute change, not a rewrite.

| | Where prep runs | Where training runs | GPU quota | Chassis heat | Setup |
|---|---|---|---|---|---|
| **1. All local** | local CPU | local RTX 4060 | none | yes | `pip install`, done |
| **2. All Kaggle** | Kaggle CPU notebook | Kaggle GPU notebook | 30 h/week | none | 3 notebooks + `kaggle kernels push` |
| **3. Hybrid** | local CPU | Kaggle GPU notebook | training only | none | upload processed set as a Kaggle Dataset |

### 4.3 Recommendation: option 1, all local

- No quota, no 12-hour session ceiling. Iterate as many times as the problem needs.
- One script and one folder instead of three chained notebooks.
- Data stays on disk. When the confusion matrix points at `melanoma → nevus` errors,
  the images can be opened and looked at, which is how those errors actually get
  understood.

Fall back to option 3 if the laptop becomes unusable for other work during training,
or if a `yolo11m-cls` run turns out to need more than one sitting.

### 4.4 Kaggle platform constraints (options 2 and 3)

| Constraint | Value | Implication |
|---|---|---|
| GPU quota | ~30 h/week (P100 or T4 x2) | Budget ~4 h per training run; roughly 7 runs/week |
| Session wall clock | 12 h GPU, 9 h CPU | One run must finish inside a session |
| `/kaggle/working` | 20 GB, persists as notebook output | Weights and manifests go here |
| `/kaggle/input` | Read-only | Never write to it |
| `/kaggle/temp` | Scratch, wiped at session end | Use for the unpacked image tree |
| Internet | **Off by default** | Toggle on for `pip install` and pretrained weights |
| Output dataset | 200 GB max | Preprocessed set fits easily |

Under option 2, split the work across notebooks. A single notebook that preprocesses
and trains re-does the preprocessing on every restart and burns GPU quota on CPU work.

- **Notebook 1 — `prep`** (CPU only, no GPU quota consumed). Stages 1-4. Saves the
  cleaned, resized, split dataset as a Kaggle Dataset.
- **Notebook 2 — `train`** (GPU). Attaches Notebook 1's output. Stages 5-6.
- **Notebook 3 — `eval`** (GPU or CPU). Stage 7. Attaches Notebook 2's weights.

Chaining this way means a failed training run costs a GPU session, not a rebuild.

### 4.5 Kaggle credentials — required under every option

Dataset download needs an API token even when nothing else touches Kaggle.

1. Open <https://www.kaggle.com/settings> (log in first; a free account is enough).
2. Scroll to the **API** section.
3. Click **Create New Token**. The browser downloads `kaggle.json`, shaped like
   `{"username":"...","key":"..."}`.
4. Place it at `C:\Users\mumin\.kaggle\kaggle.json`. In PowerShell:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle"
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

5. Verify without printing the key: `kaggle datasets list -s dermnet --max-size 1`
   should return a table, not `Authentication required`.

**Treat this file as a password.** Do not commit it, do not paste its contents into a
chat, and add `.kaggle/` and `kaggle.json` to `.gitignore` if the project is ever put
under git. Some datasets additionally require accepting terms once on their Kaggle
page before the CLI is allowed to download them; the CLI error says so explicitly when
this is the case.

### 4.6 Driving Kaggle from the local terminal

With the token in place, the whole Kaggle workflow runs from the local shell. The web
editor is never required.

```bash
# pull a dataset to local disk
kaggle datasets download -d shubhamgoel27/dermnet -p data/raw --unzip

# publish a local folder as a Kaggle Dataset (option 3: the processed set)
kaggle datasets create -p ./processed_data

# push a local notebook/script to Kaggle and run it there (options 2 and 3)
kaggle kernels push -p ./notebook_folder

# poll a run
kaggle kernels status <username>/skin-train

# pull a finished run's output (weights, logs) back to local
kaggle kernels output <username>/skin-train -p ./results
```

`kaggle kernels push` reads a `kernel-metadata.json` beside the notebook. Its
`dataset_sources` list is how datasets get attached programmatically — the equivalent
of the sidebar's **Add Data**, but reproducible.

---

## 5. Pipeline Stages

### Stage 0 — Acquire

**Option 1 (local, primary).** Requires the token from §4.5. The `kaggle` CLI is
already installed in the project's Python.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics imagehash pandas pyarrow scikit-learn
kaggle datasets download -d nazmussadat013/fitzpatrick-17k-dataset -p data/raw --unzip
kaggle datasets download -d shubhamgoel27/dermnet                  -p data/raw --unzip
kaggle datasets download -d mahdavi1202/skin-cancer                -p data/raw --unzip
```

About 25 GB of raw data. After each unzip, list the top two directory levels and
record the actual layout in this document — mirror layouts are not standardised and
Stage 1 must be written against what is really there.

**Options 2 and 3 (Kaggle).** Use **Add Data** in the notebook sidebar and search each
slug, or list the slugs under `dataset_sources` in `kernel-metadata.json` (§4.6). No
download code, no internet toggle, no quota.

The authoritative label CSV is already present at `data/raw/fitzpatrick17k.csv`
(downloaded from GitHub, no auth required). Prefer it over any CSV inside a mirror.

### Stage 1 — Build a unified manifest

Walk all three input trees and emit one Parquet file. One row per image, never a
directory structure as the source of truth — directory layouts differ per mirror and
break silently.

```
path            absolute path to the image file
source          fitzpatrick17k | dermnet | padufes20
raw_label       label string exactly as the source gives it
disease         canonical label, filled in Stage 2
malignancy      benign | malignant | non-neoplastic, filled in Stage 2
group_id        patient_id (PAD-UFES-20) or file hash (others)
fitz_scale      1-6, or -1 where unknown (Fitzpatrick17k only)
width, height   from the image header, not a full decode
```

Join Fitzpatrick17k images to the GitHub CSV on `md5hash`. **Assert the join rate** and
fail loudly below ~95%:

```python
matched = manifest.md5hash.isin(fitz_csv.md5hash).sum()
assert matched / len(fitz_csv) > 0.95, f"mirror only covers {matched}/{len(fitz_csv)}"
```

### Stage 2 — Reconcile labels

This is where the real work is, and it is manual. Budget an hour.

**Build `label_map.csv` by hand** — one row per distinct `raw_label` across all three
sources (roughly 143 before merging), mapping to `(disease, malignancy)`. Commit it.
A hand-checked CSV is auditable; a chain of `str.replace` calls is not.

Three jobs it has to do:

1. **Collapse synonyms within Fitzpatrick17k.** `melanoma`, `malignant melanoma`,
   `superficial spreading melanoma ssm` and `lentigo maligna` → one `melanoma` class
   (563 images combined, versus 261 for the largest alone). Same for the three BCC
   variants → `basal cell carcinoma` (596 combined).
2. **Bridge vocabularies across sources.** DermNet's `Melanoma Skin Cancer Nevi and
   Moles`, Fitzpatrick17k's `melanoma`, and PAD-UFES-20's `MEL` are the same condition
   under three naming schemes.
3. **Assign malignancy where it is not native.** Fitzpatrick17k supplies
   `three_partition_label` directly — use it as the reference and copy its conventions.
   PAD-UFES-20: `BCC`, `MEL`, `SCC` → `malignant`; `ACK` → `malignant` (actinic
   keratosis is pre-cancerous, and Fitzpatrick17k already labels it malignant — match
   that convention so the two sources do not contradict each other); `NEV`, `SEK` →
   `benign`. DermNet: map its 23 classes by hand.

Watch for DermNet's `Melanoma Skin Cancer Nevi and Moles`, which merges a malignancy
with two benign conditions in a single folder. It cannot be assigned a clean malignancy
label. Either exclude it from Model B's training set or route it to a
`mixed` bucket that Model B ignores. Do not guess.

### Stage 3 — Deduplicate

Fitzpatrick17k has documented duplicate and near-duplicate images (see the *Scientific
Data* audit, §9). Cross-source duplicates are also plausible, since these atlases quote
each other.

```python
import imagehash
from PIL import Image
manifest["phash"] = [str(imagehash.phash(Image.open(p))) for p in manifest.path]
```

Group by Hamming distance <= 5, keep one representative per group, and record the drop
count. Run deduplication **before** splitting, never after — a duplicate straddling the
train/val boundary inflates every metric downstream.

### Stage 4 — Filter, split, resize

**Filter for Model A.** Drop disease classes with < 100 images after merging. A class
with 50 examples cannot be learned and pollutes the confusion matrix. Expect roughly
60-75 surviving classes. **Model B keeps every image** — a class too rare to name still
carries a valid malignancy label, and that is Model B's only job.

**Split** 70/15/15 with `StratifiedGroupKFold`, stratified on `disease`, grouped on
`group_id`. Write the assignment into the manifest as a `split` column and never
re-randomise — a changing split makes two training runs incomparable.

**Resize** the long edge to 512 px, keep the aspect ratio, save as JPEG quality 92.
Full-resolution DermNet images waste decode time at a 224-320 px training resolution.
Then publish `/kaggle/working` as a Kaggle Dataset for Notebook 2.

### Stage 5 — Materialise the YOLO layout

`yolo classify` expects ImageFolder layout. Build it from the manifest with symlinks
where the filesystem allows, copies otherwise:

```
dataset_disease/          dataset_malignancy/
  train/<disease>/*.jpg     train/<malignancy>/*.jpg
  val/<disease>/*.jpg       val/<malignancy>/*.jpg
  test/<disease>/*.jpg      test/<malignancy>/*.jpg
```

Two trees, same underlying images, different groupings. Generate both from the one
manifest so they can never drift apart.

### Stage 6 — Train

Local (option 1). `device=0` is the RTX 4060; `workers=4` is the RAM ceiling from §4.1.

```bash
# Model A — disease
yolo classify train model=yolo11s-cls.pt data=data/yolo/dataset_disease \
  epochs=60 imgsz=320 batch=64 patience=12 workers=4 device=0 \
  cos_lr=True label_smoothing=0.1 \
  project=runs name=disease

# Model B — malignancy
yolo classify train model=yolo11s-cls.pt data=data/yolo/dataset_malignancy \
  epochs=40 imgsz=320 batch=64 patience=10 workers=4 device=0 \
  cos_lr=True \
  project=runs name=malignancy
```

Kaggle (options 2 and 3): same commands with `pip install ultralytics` first (internet
ON), `data=/kaggle/temp/...`, `project=/kaggle/working/runs`, and `workers` can rise to
the notebook's CPU count.

Run Model A and Model B **sequentially**, not in parallel. Two trainers on one 8 GB
card will contend for VRAM and both slow down more than they would gain.

Start with `yolo11s-cls`. Move to `yolo11m-cls` only after the small model's error
profile has been read — if it is confusing visually similar conditions, more capacity
may help; if it is failing on rare classes, more data or better sampling will help and
capacity will not.

**Class imbalance.** `non-neoplastic` outnumbers `malignant` 5.3:1 in Fitzpatrick17k
alone, and it is worse for individual diseases. Ultralytics' classification trainer
exposes no class-weight argument, so handle it at the data layer: build a
`WeightedRandomSampler` with weights `1/sqrt(class_count)`, or oversample minority
classes on disk during Stage 5. Verify by logging the realised class distribution of
the first epoch's batches — do not assume the sampler is doing what you intended.

**Augmentation.** Colour jitter must stay mild. Hue and saturation carry diagnostic
signal in dermatology (erythema, pigmentation, vascularity) and also encode skin tone.
Aggressive colour augmentation destroys the first and can worsen fairness on the second.
Set `hsv_h=0.015 hsv_s=0.4 hsv_v=0.3`. Geometric augmentation is safe and useful:
`fliplr=0.5 flipud=0.5 degrees=30 scale=0.4`.

### Stage 7 — Evaluate

Accuracy is close to useless here. Report:

**7.1 Model A (disease)** — top-1 and top-5, plus macro-F1. Macro-F1 is the headline:
it weights a 120-image class equally with a 650-image one, and it is the number that
falls when the model quietly gives up on rare conditions.

**7.2 Model B (malignancy)** — per-class precision/recall, and specifically
**recall on `malignant`**. This is the metric the project lives or dies by. Also report
sensitivity at a fixed 90% specificity operating point, and plot the full ROC — the
default 0.5 threshold is almost certainly wrong for a screening tool, and the threshold
should be chosen on val and frozen before touching test.

**7.3 Fairness** — everything in 7.2, broken out by `fitz_scale` bucket (1-2, 3-4, 5-6).
A model at 85% malignant recall on types 1-2 and 60% on types 5-6 is not an 80% model;
it is a model that fails the people already worst served by dermatology. Types 5-6 are
only 13% of the data, so this is a live risk, not a hypothetical one.

**7.4 Confusion matrix** on the disease head, read manually. Adjacent-class confusion
(two eczema variants) is tolerable. `melanoma → benign nevus` confusion is the
dangerous cell and should be inspected image by image.

### Stage 8 — Export

```bash
yolo export model=/kaggle/working/runs/disease/weights/best.pt format=onnx imgsz=320
```

Ship alongside the weights: `label_map.csv`, the class-index ordering, the frozen
decision threshold from 7.2, and the test-set metrics. Weights without the label
ordering are unusable, and this is a routine way to lose a week.

---

## 6. Execution Order

Times are for option 1 (local). The Kaggle column shows where each stage lands under
option 2.

| # | Stage | Local | Kaggle equivalent | Time | Blocked by |
|---|---|---|---|---|---|
| 0 | Acquire | `kaggle datasets download` | Add Data | 30-60 min (25 GB) | `kaggle.json` |
| 1 | Manifest | `scripts/01_manifest.py` | Notebook 1 (CPU) | 20 min | 0 |
| 2 | Label map | **Manual**, `label_map.csv` | same | 1-2 h | 1 |
| 3 | Deduplicate | `scripts/03_dedup.py` | Notebook 1 (CPU) | 15-40 min | 1 |
| 4 | Filter/split/resize | `scripts/04_split.py` | Notebook 1 (CPU) | 30 min | 2, 3 |
| 5 | YOLO layout | `scripts/05_layout.py` | Notebook 2 | 10 min | 4 |
| 6 | Train A + B | `yolo classify train` on RTX 4060 | Notebook 2 (GPU) | 2-2.5 h | 5 |
| 7 | Evaluate | `scripts/07_eval.py` | Notebook 3 | 1 h | 6 |
| 8 | Export | `yolo export` | Notebook 3 | 10 min | 7 |

Stage 2 is the critical path and cannot be automated away. Everything downstream
inherits its errors, and a bad label map produces a model that is confidently wrong in a
way no metric will reveal.

---

## 7. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Kaggle mirror is incomplete | Training on a silent subset | Assert join rate in Stage 1 (§5) |
| Label map errors | Undetectable systematic error | Hand-build, commit, review the CSV |
| Duplicate leakage | Metrics inflated, real accuracy unknown | pHash dedup **before** split (Stage 3) |
| PAD-UFES-20 patient leakage | Inflated metrics | Group split on `patient_id` |
| Source-correlated labels | Model learns source, not disease | Domain-matched sources only (§3.2) |
| Poor recall on dark skin | Harms the users most in need | Report per-`fitz_scale` metrics (§7.3) |
| Rare-class collapse | High accuracy, useless model | Macro-F1 as headline; min 100 images/class |
| Laptop thermal throttling (option 1) | Training 2x slower, not failed | Mains power, cooling pad; run A and B sequentially |
| 16 GB RAM exhaustion (option 1) | Dataloader swap, crawl or OOM | `workers=4`; resize to 512 px in Stage 4 first |
| GPU quota exhausted (options 2-3) | Week-long stall | Split prep (CPU) from train (GPU) |
| `kaggle.json` leaked | Account compromise | Never commit; `.gitignore` it; rotate at kaggle.com/settings if exposed |

---

## 8. Scope Note

This is a screening aid, not a diagnostic device. It is trained on internet-sourced
atlas photographs with no histopathological confirmation for most images, and its worst
performance will be on exactly the presentations that are hardest to photograph well.
Any interface built on it states that a dermatologist decides, and never suppresses a
recommendation to seek one.

---

## 9. References

- Fitzpatrick17k — <https://github.com/mattgroh/fitzpatrick17k>
- Fitzpatrick17k quality audit — <https://www.nature.com/articles/s41597-025-04382-5>
- DermNet (Kaggle) — <https://www.kaggle.com/datasets/shubhamgoel27/dermnet>
- PAD-UFES-20 (source) — <https://data.mendeley.com/datasets/zr7vgbcyr2/1>
- PAD-UFES-20 (paper) — <https://www.sciencedirect.com/science/article/pii/S235234092031115X>
- Ultralytics classification docs — <https://docs.ultralytics.com/tasks/classify/>
