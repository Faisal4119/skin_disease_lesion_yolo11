# Skin Lesion Dataset — Sources and Splits

This file covers where the data came from, how it was cleaned and split, and where each
file lives.

---

## 1. Where the data came from

| Source | Origin | Raw | Final | Role |
|---|---|---|---|---|
| **DermNet** | Hugging Face `exper1ment/dermnet` | 19,560 | 17,829 | Main dataset, 23 classes |
| **SCIN** (Google) | Public GCS bucket `dx-scin-public-data` | 10,384 | 5,665 | Phone photos of common conditions (rashes, eczema, etc.) |
| **Fitzpatrick17k** | `atlasdermatologico.com.br` (only 24% of the original URLs still work) | 3,889 | 3,876 | Skin tone (Fitzpatrick 1-6) labels |
| **PAD-UFES-20** | Mendeley, downloaded manually through a browser | 2,299 | 2,270 | Cancer images taken on phones (BCC, SCC, MEL, ACK) |
| **Total** | | **36,108** | **29,640** | |

Dermoscopy data (ISIC, HAM10000) was left out on purpose. Those images are taken through
a lens, so mixing them in would teach the model the camera type instead of the disease.

Source links:
- DermNet — <https://huggingface.co/datasets/exper1ment/dermnet>
- SCIN — <https://github.com/google-research-datasets/scin>
- Fitzpatrick17k — <https://github.com/mattgroh/fitzpatrick17k>
- PAD-UFES-20 — <https://data.mendeley.com/datasets/zr7vgbcyr2/1>

---

## 2. How the data was separated

### 2.1 Label reconciliation — `scripts/02c_build_label_map.py`

- The four sources had 330 distinct labels between them. These were mapped to 27
  disease classes.
- Malignancy (`benign` / `malignant` / `non-neoplastic`) is derived from the disease,
  not taken from the source. That way the same condition can never get two different
  labels from two different sources.
- DermNet folder 11 ("Melanoma Skin Cancer Nevi and Moles") mixed melanoma and nevus
  images. It was sorted by hand, one image at a time (`data/dermnet_folder11_labels.csv`).
  Images marked unsure were dropped.
- From SCIN, only cases with a dermatologist agreement weight of at least 0.5 were kept.

### 2.2 Duplicate removal — `scripts/03_dedup.py`

- Images were hashed with pHash. Two images with a Hamming distance of 5 or less count as
  duplicates.
- 1,558 images were removed (DermNet 1,522, SCIN 16, PAD-UFES-20 16, Fitzpatrick17k 4).
- This ran **before** the split, so the same image can't end up in both train and test.

### 2.3 Filtering — `scripts/04_split.py`

- Images with a short side under 200 px were dropped.
- Classes with fewer than 100 images were dropped from Model A (5 classes:
  `cutaneous_lymphoma`, `kaposi_sarcoma`, `keratosis_pilaris`, `miliaria`,
  `photodermatosis`). They were kept for Model B.
- Images were resized to a 512 px long edge and saved as JPEG at quality 92.

### 2.4 Train / Val / Test split — `scripts/04_split.py`

- `StratifiedGroupKFold` with 20 folds and seed 42. Folds 0-13 are train, 14-16 are val,
  and 17-19 are test.
- **70 / 15 / 15**, which gives train 20,749, val 4,446 and test 4,445.
- Images are grouped by patient or case ID, so photos of the same patient or the same
  lesion never land in two different splits. This was checked:
  `groups spanning splits: 0`.

### 2.5 Two folders for the two models — `scripts/05_layout.py`

| Tree | Classes | train | val | test |
|---|---|---|---|---|
| `dataset_disease` (Model A) | 27 | 19,239 | 4,121 | 4,120 |
| `dataset_malignancy` (Model B) | 3 | 20,749 | 4,446 | 4,445 |

In the `_bal` folders, only the train split is oversampled (rare classes are repeated, up
to 4x). Val and test are the same as in the original trees.

---

## 3. Paths

```
L:\personal_project\yolo_model\datasheet\data
```

| Path | Contents |
|---|---|
| `data\raw\` | Original downloads, unchanged |
| `data\processed\` | Final cleaned and resized images (29,640) |
| `data\yolo\dataset_disease\` | Train/val/test for Model A |
| `data\yolo\dataset_malignancy\` | Train/val/test for Model B |
| `data\yolo\*_bal\` | Oversampled copies used for training |
| `data\yolo\classes_*.txt` | Class order |
| `data\manifest_final.csv` | Label, split, source and fitz_scale for every image |
| `data\label_map.csv` | Mapping from raw label to disease/malignancy |

Folder structure: `<tree>\{train,val,test}\<class>\*.jpg`

---

## 4. Sharing with a teammate

- The path above is on a local drive and won't work on another PC. Upload the data to
  Google Drive or OneDrive, or send it as a zip.
- The smallest useful package is `data\processed\`, `manifest_final.csv` and
  `label_map.csv`. Running `scripts/05_layout.py` on the other PC rebuilds the `yolo\`
  folder from those.
- The files in `yolo\` are hard links. Zipping them turns each link into a real copy and
  makes the archive much larger. The `*_bal\` folders don't need to be sent.
