# Plan B — Fully Local, Zero Kaggle Dependency

**Companion to:** `PIPELINE_PLAN.md` (the main plan). This document changes only what
Kaggle-free sourcing forces to change: data acquisition (Stage 0-1) and the label
reconciliation that follows from it (Stage 2). Stages 3-8, the two-model design, the
evaluation protocol and the local hardware assessment are unchanged and are not
repeated here — read them in the main plan.

**Verdict:** feasible. About 32,000 clinical images from four sources, none of which
require a Kaggle account. Cost of going Kaggle-free: one manual browser download,
a weaker melanoma signal, and the loss of 76% of Fitzpatrick17k unless its maintainers
answer a data request.

**Last updated:** 2026-09-20. Every source below was probed live on that date.

---

## 1. What Changes and Why

The main plan's three sources were chosen for one property: clinical/smartphone
photographs, not dermoscopy. That constraint survives. What does not survive is the
Kaggle mirror as the delivery mechanism, and that matters for exactly one dataset.

| Source | Kaggle-free status | Consequence |
|---|---|---|
| DermNet | **Full copy on Hugging Face**, verified 19,559 images, 23 classes | No change |
| PAD-UFES-20 | **Official Mendeley source**, metadata via API, images via browser | One manual download |
| Fitzpatrick17k | **No full copy anywhere outside Kaggle.** 76% of source URLs dead. | Partial (24%) now; full only via maintainer request |
| SCIN (new) | **Public Google Cloud bucket**, no auth | Added to recover lost breadth |

Fitzpatrick17k was carrying two loads in the main plan: 114-condition breadth and
2,263 malignant images with a native 3-way malignancy label. Without it, breadth comes
from SCIN and DermNet, and malignancy comes from PAD-UFES-20 and one DermNet folder.
Section 3 quantifies what that costs.

---

## 2. Verified Sources

### 2.1 DermNet — Hugging Face `exper1ment/dermnet`

Probed via the Hugging Face datasets-server API.

```
format   parquet, columns: image (Image), label (ClassLabel, 23 names)
splits   train 15,557   test 4,002   total 19,559
size     1.85 GB
auth     none
```

The 23 class names match the Kaggle DermNet folder names exactly, so the label map in
the main plan's Stage 2 transfers without edits. `34data/dermnet` (19,559 rows,
1.76 GB, raw JPEG bytes + filename) is a second copy if the first goes away.

```python
from datasets import load_dataset
ds = load_dataset("exper1ment/dermnet")           # ~1.85 GB, cached under ~/.cache/huggingface
ds["train"].features["label"].names               # the 23 strings
```

For Stage 1 the images must exist as files on disk, not as parquet rows. Export once:

```python
import os
for split in ("train", "test"):
    for i, ex in enumerate(ds[split]):
        cls = ds[split].features["label"].int2str(ex["label"])
        out = f"data/raw/dermnet/{split}/{cls}/{i:06d}.jpg"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ex["image"].convert("RGB").save(out, quality=95)
```

### 2.2 PAD-UFES-20 — Mendeley Data (official)

`metadata.csv` is served directly by the Mendeley API with no authentication:

```
https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/fa850265-57da-48f0-ba3e-998b3e44b1f6/file_downloaded
```

Verified contents: 2,298 rows, 1,373 unique `patient_id`, 1,641 unique `lesion_id`.
The `patient_id` column the main plan's grouped split depends on is present.

Diagnosis distribution:

```
BCC   845   malignant
ACK   730   malignant (pre-cancerous; matches the Fitzpatrick17k convention)
NEV   244   benign
SEK   235   benign
SCC   192   malignant
MEL    52   malignant
```

**The image archives are not exposed by the Mendeley API** — only `metadata.csv` is
listed at the root, and the folder endpoints return 404. The S3 "download all" URL
returns 403 without a browser-issued session. This is the one manual step in Plan B:

1. Open <https://data.mendeley.com/datasets/zr7vgbcyr2/1> in a browser.
2. Click **Download All** (about 3.5 GB).
3. Extract into `data/raw/pad_ufes_20/`. Expect `imgs_part_1/`, `imgs_part_2/`,
   `imgs_part_3/` and `metadata.csv`.

Filenames follow `PAT_<patient>_<lesion>_<n>.png` and match `img_id` in the CSV, so
the join is exact.

### 2.3 SCIN — Google Cloud Storage `dx-scin-public-data` (new source)

Google's 2024 crowdsourced dermatology dataset. Real smartphone photos donated by US
internet users with informed consent, labelled by up to three dermatologists each.

```
bucket    gs://dx-scin-public-data, public, no auth, listable over HTTP
images    dataset/images/  — 10,379 PNG files, 12.58 GB
cases     dataset/scin_cases.csv   — 5,033 cases, 1-3 images each (20,815 image slots, 10,379 filled)
labels    dataset/scin_labels.csv  — 5,033 rows, weighted multi-label condition dict per case
```

Every file is fetchable as
`https://storage.googleapis.com/dx-scin-public-data/<object name>`. No SDK required;
a 40-line Python downloader with `concurrent.futures` covers it. Listing is paginated
through `https://storage.googleapis.com/storage/v1/b/dx-scin-public-data/o?prefix=dataset/images/`.

**Label structure, verified.** `weighted_skin_condition_label` is a Python-dict literal
per case: `{"Eczema": 0.67, "Allergic Contact Dermatitis": 0.33}`. Only **3,061 of
5,033 cases** carry a non-empty label; the rest were not gradable. Taking the
top-weighted condition per case yields 211 distinct conditions, and only **six** of
them reach 100 cases:

```
Eczema                          488
Allergic Contact Dermatitis     270
Urticaria                       214
Insect Bite                     185
Folliculitis                    142
Psoriasis                       109
```

SCIN's malignant content is negligible — Basal Cell Carcinoma 6 cases, Melanoma 1,
Actinic Keratosis 12. **It contributes nothing to Model B.** What it does contribute is
several thousand genuine phone photographs of the common inflammatory conditions a
consumer app will actually be pointed at, in a capture domain closer to deployment than
any atlas. Use it for Model A breadth and for the `non-neoplastic` class of Model B.

Handling rules for Stage 2:

- Keep a case only if its top-weighted condition has weight >= 0.5. Below that the
  dermatologists disagreed and the label is noise.
- Map the case label onto all of its 1-3 images. Group on `case_id` for the split —
  three photos of one rash must not straddle train and val.
- Conditions with fewer than 100 images after mapping fold into their DermNet
  equivalent where one exists (SCIN `Tinea` → DermNet fungal folder), else drop for
  Model A. They remain in Model B as `non-neoplastic`.
- Expect roughly 5,000-6,500 usable images after these rules, not 10,379.

### 2.4 Fitzpatrick17k — partial recovery

`data/raw/fitzpatrick17k.csv` (already downloaded, 16,577 rows) lists a source URL per
image. Tested 2026-09-19 and 2026-09-20:

| Host | Rows | Status |
|---|---|---|
| `www.dermaamin.com` | 12,631 | DNS `SERVFAIL`. Gone. |
| `atlasdermatologico.com.br` | 3,905 | Alive. Sampled URLs return `200 image/jpeg`, 45-95 KB. |
| blank | 41 | — |

Hugging Face has no full copy: `spycoder/fitzpatrick` (16,577 rows) is the label CSV
with a `path` column and **no image files** in the repo; `ZYXue/Fitzpatrick_17k` is a
5,148-row subset. Every other HF entry is a derivative that assumes you already have the
images.

Two things to do, in parallel:

1. **Scrape the 3,905 live images now.** Throttle to one request per second — this is a
   small Brazilian academic site, not a CDN. Verify each download against the CSV's
   `md5hash` column; that column is the MD5 of the image bytes, so a mismatch means a
   changed or placeholder image and the row is dropped. About 70 minutes at 1 req/s.
2. **Submit the maintainers' data request form** linked from the GitHub README. They
   distribute the full archive to researchers on request. Turnaround is days to weeks and
   not guaranteed. If it arrives, Stage 1 re-runs with the full set and Plan B converges
   on the main plan's numbers.

The partial set is still worth having: `atlasdermatologico.com.br` supplies a share of
the malignant rows (SCC, BCC, melanoma variants) and is the only Fitzpatrick17k content
that reaches Model B.

---

## 3. What the Merged Set Looks Like

| Source | Images (usable) | Model A role | Model B role |
|---|---|---|---|
| DermNet | 19,559 | 23 broad classes | folder 1 → `malignant`; folder 11 → excluded (mixed); rest mapped |
| SCIN | ~5,000-6,500 | common inflammatory conditions, phone domain | `non-neoplastic` |
| PAD-UFES-20 | 2,298 | 6 lesion types | **primary cancer source**: 1,819 malignant |
| Fitzpatrick17k (partial) | ~3,900 | whatever survives the 100-image floor | a few hundred malignant |
| **Total** | **~31,000-32,000** | | |

Malignancy supply for Model B, approximate:

```
PAD-UFES-20    BCC 845 + ACK 730 + SCC 192 + MEL 52        = 1,819
DermNet        "Actinic Keratosis Basal Cell Carcinoma and
                other Malignant Lesions" folder              ~ 1,100
Fitzpatrick17k atlasdermatologico share of 11 malignant labels ~   400
                                                             -------
                                                             ~ 3,300
```

That is more malignant images than the main plan had (2,263 from Fitzpatrick17k alone),
because PAD-UFES-20 is cancer-heavy by construction. **The distribution inside
"malignant" is the problem.** BCC and actinic keratosis dominate. Melanoma — the cancer
that kills — is roughly 52 (PAD) + ~100 (atlas) + whatever can be salvaged from
DermNet's mixed melanoma/nevus folder. Under 200 clean melanoma images is thin.
Report melanoma recall as its own line in Stage 7, separate from overall malignant
recall, and expect it to be the weakest number in the evaluation. If the Fitzpatrick17k
request comes through, its 573 melanoma-family images are the single biggest upgrade
available.

DermNet folder 11, `Melanoma Skin Cancer Nevi and Moles`, deserves a decision rather
than a default. It mixes a malignancy with two benign conditions under one label and
cannot be assigned a malignancy value. Options: exclude it from Model B entirely (clean,
loses maybe 400 melanoma images), or run a quick manual pass sorting its ~460 images
into melanoma / not-melanoma by eye with the filenames as hints (DermNet filenames often
carry the condition). The manual pass is about an hour and is the only way to recover
those images for Model B. Recommended.

---

## 4. Stage 0 — Acquire (replaces the main plan's Stage 0)

Disk budget: roughly 1.9 + 3.5 + 12.6 + 0.3 GB = **18 GB raw**, then ~5 GB processed.
`D:` has 138 GB free.

```bash
# environment (one time)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics datasets huggingface_hub imagehash pandas pyarrow scikit-learn pillow

# 1. DermNet — Hugging Face, ~1.85 GB
python scripts/00a_fetch_dermnet.py        # load_dataset + export to data/raw/dermnet/

# 2. SCIN — GCS over HTTP, ~12.6 GB, ~10k files, parallel 8 workers
python scripts/00b_fetch_scin.py           # -> data/raw/scin/images/, scin_cases.csv, scin_labels.csv

# 3. Fitzpatrick17k partial — 3,905 URLs at 1 req/s, md5-verified
python scripts/00c_fetch_fitz_atlas.py     # -> data/raw/fitzpatrick17k/images/<md5>.jpg

# 4. PAD-UFES-20 metadata — direct
curl -L -o data/raw/pad_ufes_20/metadata.csv \
  "https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/fa850265-57da-48f0-ba3e-998b3e44b1f6/file_downloaded"

# 5. PAD-UFES-20 images — MANUAL: browser "Download All" from
#    https://data.mendeley.com/datasets/zr7vgbcyr2/1 -> extract to data/raw/pad_ufes_20/
```

Each fetch script writes a `_manifest.csv` beside its output with `path, sha1, bytes` so
Stage 1 can detect a truncated or partial download instead of silently proceeding.

---

## 5. Stage 1-2 Adjustments

**Stage 1 (manifest)** gains a `case_id` column populated for SCIN and set equal to
`group_id`. `group_id` is `patient_id` for PAD-UFES-20, `case_id` for SCIN, and the
file's own hash for DermNet and Fitzpatrick17k. Everything else in the main plan's
Stage 1 stands, including the join-rate assertion for Fitzpatrick17k — except that the
expected denominator is now 3,905, not 16,577.

**Stage 2 (label map)** grows from ~143 raw labels to ~143 + 211 SCIN conditions. Do
not hand-map all 211. Map the ones with >= 100 images (six of them, plus a dozen in the
50-99 band worth folding into DermNet folders) and send the tail to `non-neoplastic`
for Model B only. The map file gains a `source` column so that the same string from two
sources can be routed differently if they turn out to mean different things.

---

## 6. Execution Order (Plan B)

| # | Stage | Time | Blocked by |
|---|---|---|---|
| 0a | Fetch DermNet (HF) | 15 min | — |
| 0b | Fetch SCIN (GCS) | 30-45 min | — |
| 0c | Scrape Fitzpatrick17k atlas share | ~70 min, background | — |
| 0d | PAD-UFES-20 metadata | 1 min | — |
| 0e | **PAD-UFES-20 images — manual browser step** | 10 min | user |
| 0f | Submit Fitzpatrick17k data request | 5 min | user; async, days |
| 1 | Manifest | 20 min | 0a-0e |
| 2 | Label map (incl. DermNet folder 11 manual sort) | 2-3 h | 1 |
| 3-8 | As main plan | as main plan | 2 |

0a through 0d run in parallel and unattended. Total unattended wall clock about 75
minutes, bounded by the throttled scrape.

---

## 7. Plan B Risk Additions

| Risk | Impact | Mitigation |
|---|---|---|
| Melanoma under-represented (<200) | Weak recall on the deadliest class | Report melanoma recall separately; manual sort of DermNet folder 11; Fitzpatrick17k request |
| SCIN labels are top-1 of a disagreement | Label noise in inflammatory classes | Weight threshold 0.5; drop below |
| SCIN 1-3 photos per case | Leakage across split | Group on `case_id` |
| `atlasdermatologico.com.br` rate-limits or blocks | Lose 3,905 images | 1 req/s throttle; resume-safe script; md5 verify |
| HF mirror removed | Lose DermNet | `34data/dermnet` as fallback; keep the exported files |
| Mendeley "Download All" fails in browser | Lose PAD-UFES-20, the cancer backbone | Retry; the three `imgs_part_N` zips can also be fetched individually from the web UI |

---

## 8. Sources

- DermNet (HF) — <https://huggingface.co/datasets/exper1ment/dermnet>
- DermNet fallback (HF) — <https://huggingface.co/datasets/34data/dermnet>
- PAD-UFES-20 (Mendeley) — <https://data.mendeley.com/datasets/zr7vgbcyr2/1>
- SCIN (GitHub, docs) — <https://github.com/google-research-datasets/scin>
- SCIN bucket — <https://storage.googleapis.com/dx-scin-public-data/dataset/scin_cases.csv>
- Fitzpatrick17k (labels + data request form) — <https://github.com/mattgroh/fitzpatrick17k>
