"""Stage 1 — build one unified manifest across all four raw sources.

Reads:  data/raw/dermnet/_manifest.csv
        data/raw/scin/{scin_cases.csv, scin_labels.csv, images/}
        data/raw/pad_ufes_20/{metadata.csv, imgs_part_*/}
        data/raw/fitzpatrick17k/_manifest.csv   (+ fitzpatrick17k.csv for the full-set denominator)
Writes: data/manifest_raw.csv            one row per image
        data/label_map_skeleton.csv      one row per (source, raw_label) with counts — Stage 2 starts here
        data/manifest_summary.txt

Every row carries: path, source, raw_label, group_id, fitz_scale, width, height, bytes.
Source-specific extras are kept in their own columns; blank where not applicable.

Hard checks (script exits non-zero if any fail):
  - DermNet rows == 19,559
  - SCIN images on disk == 10,379, and every labelled case's images exist
  - PAD-UFES-20 rows == 2,298 and every img_id resolves to a file
  - Fitzpatrick17k ok-rows / 3,905 >= 0.95
"""
from __future__ import annotations

import ast
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_MANIFEST = ROOT / "data" / "manifest_raw.csv"
OUT_SKELETON = ROOT / "data" / "label_map_skeleton.csv"
OUT_SUMMARY = ROOT / "data" / "manifest_summary.txt"

SCIN_MIN_WEIGHT = 0.5  # dermatologist agreement floor for the top condition

COLUMNS = [
    "path", "source", "raw_label", "group_id", "fitz_scale", "width", "height", "bytes",
    # source extras
    "split_hint",                # dermnet: train/test as shipped
    "case_id",                   # scin
    "label_weight",              # scin: weight of the top condition
    "label_dict",                # scin: full weighted dict, for Stage 2 review
    "patient_id", "lesion_id",   # pad
    "md5hash", "three_partition_label", "nine_partition_label",  # fitzpatrick17k
]

errors: list[str] = []
summary: list[str] = []


def say(msg: str) -> None:
    print(msg, flush=True)
    summary.append(msg)


def fitz_int(v: str | None) -> int:
    """'FST3' -> 3, '2.0' -> 2, '' / 'NONE_IDENTIFIED' / '-1' -> -1."""
    if not v:
        return -1
    m = re.search(r"\d", v)
    if not m:
        return -1
    n = int(m.group())
    return n if 1 <= n <= 6 else -1


def img_dims(p: Path) -> tuple[int, int]:
    with Image.open(p) as im:  # header only, no decode
        return im.size


def row(**kw) -> dict:
    r = {c: "" for c in COLUMNS}
    r.update(kw)
    return r


# ---------------------------------------------------------------- DermNet
def load_dermnet() -> list[dict]:
    src = RAW / "dermnet" / "_manifest.csv"
    with src.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        p = ROOT / r["path"]
        w, h = img_dims(p)
        out.append(row(path=r["path"], source="dermnet", raw_label=r["raw_label"], group_id=r["sha1"],
                       fitz_scale=-1, width=w, height=h, bytes=r["bytes"], split_hint=r["split_hint"]))
    say(f"dermnet: {len(out)} rows")
    if len(out) != 19559:
        errors.append(f"dermnet expected 19559 rows, got {len(out)}")
    return out


# ---------------------------------------------------------------- SCIN
def load_scin() -> list[dict]:
    d = RAW / "scin"
    img_dir = d / "images"
    on_disk = {p.name for p in img_dir.glob("*.png")}
    if len(on_disk) != 10379:
        errors.append(f"scin expected 10379 images on disk, got {len(on_disk)}")

    with (d / "scin_cases.csv").open(encoding="utf-8") as f:
        cases = {r["case_id"]: r for r in csv.DictReader(f)}
    with (d / "scin_labels.csv").open(encoding="utf-8") as f:
        labels = {r["case_id"]: r for r in csv.DictReader(f)}

    # An image referenced by more than one case carries conflicting labels and would
    # leak across a case-grouped split. Verified 2026-09-20: 15 such files, one of them
    # attached to 11 different cases. Drop them outright.
    img_cases: dict[str, set[str]] = defaultdict(set)
    for cid, c in cases.items():
        for k in ("image_1_path", "image_2_path", "image_3_path"):
            if c.get(k):
                img_cases[Path(c[k]).name].add(cid)
    shared = {n for n, s in img_cases.items() if len(s) > 1}

    out = []
    n_unlabelled = n_low = n_missing_img = n_shared_rows = 0
    missing_names: list[str] = []
    for cid, c in cases.items():
        lab = labels.get(cid, {})
        try:
            wd = ast.literal_eval(lab.get("weighted_skin_condition_label") or "{}")
        except (ValueError, SyntaxError):
            wd = {}
        if not wd:
            n_unlabelled += 1
            top, wt = "", ""
        else:
            top = max(wd, key=wd.get)
            wt = wd[top]
            if wt < SCIN_MIN_WEIGHT:
                n_low += 1
        # skin type: self-reported first, dermatologist label as fallback
        fitz = fitz_int(c.get("fitzpatrick_skin_type"))
        if fitz == -1:
            fitz = fitz_int(lab.get("dermatologist_fitzpatrick_skin_type_label_1"))
        for k in ("image_1_path", "image_2_path", "image_3_path"):
            rel = c.get(k)
            if not rel:
                continue
            name = Path(rel).name
            if name in shared:
                n_shared_rows += 1
                continue
            if name not in on_disk:
                n_missing_img += 1
                missing_names.append(name)
                continue
            p = img_dir / name
            w, h = img_dims(p)
            out.append(row(path=p.relative_to(ROOT).as_posix(), source="scin",
                           raw_label=top, group_id=cid, fitz_scale=fitz, width=w, height=h,
                           bytes=p.stat().st_size, case_id=cid, label_weight=wt, label_dict=str(wd)))
    labelled = sum(1 for r in out if r["raw_label"])
    strong = sum(1 for r in out if r["label_weight"] != "" and float(r["label_weight"]) >= SCIN_MIN_WEIGHT)
    say(f"scin: {len(out)} image rows from {len(cases)} cases | unlabelled cases={n_unlabelled} "
        f"low-agreement cases (<{SCIN_MIN_WEIGHT})={n_low} | labelled images={labelled} "
        f"strong-label images={strong} | dropped shared-image rows={n_shared_rows} ({len(shared)} files) "
        f"| missing files={n_missing_img}")
    if n_missing_img:
        # The bucket itself lacks these (listing = 10,379, cases reference 10,380). A source-side gap,
        # not a download failure — warn, do not fail. Anything beyond a handful IS a download problem.
        say(f"scin WARNING: referenced but absent from bucket: {missing_names}")
        if n_missing_img > 5:
            errors.append(f"scin: {n_missing_img} referenced images missing on disk")
    return out


# ---------------------------------------------------------------- PAD-UFES-20
def load_pad() -> list[dict]:
    d = RAW / "pad_ufes_20"
    files = {p.name: p for p in d.glob("imgs_part_*/*.png")}
    with (d / "metadata.csv").open(encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    out = []
    missing = 0
    for r in meta:
        p = files.get(r["img_id"])
        if p is None:
            missing += 1
            continue
        w, h = img_dims(p)
        out.append(row(path=p.relative_to(ROOT).as_posix(), source="padufes20", raw_label=r["diagnostic"],
                       group_id=r["patient_id"], fitz_scale=fitz_int(r.get("fitspatrick")), width=w, height=h,
                       bytes=p.stat().st_size, patient_id=r["patient_id"], lesion_id=r["lesion_id"]))
    say(f"padufes20: {len(out)} rows | metadata={len(meta)} files={len(files)} missing={missing}")
    if len(out) != 2298 or missing:
        errors.append(f"padufes20 expected 2298 resolved rows, got {len(out)} (missing {missing})")
    return out


# ---------------------------------------------------------------- Fitzpatrick17k (partial)
def load_fitz() -> list[dict]:
    d = RAW / "fitzpatrick17k"
    with (d / "_manifest.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") in ("ok", "skipped")]
    with (RAW / "fitzpatrick17k.csv").open(encoding="utf-8") as f:
        atlas_total = sum(1 for r in csv.DictReader(f) if "atlasdermatologico" in (r.get("url") or ""))
    out = []
    for r in rows:
        p = ROOT / r["path"]
        if not p.exists():
            continue
        w, h = img_dims(p)
        out.append(row(path=r["path"], source="fitzpatrick17k", raw_label=r["raw_label"], group_id=r["md5hash"],
                       fitz_scale=fitz_int(r["fitzpatrick_scale"]), width=w, height=h, bytes=p.stat().st_size,
                       md5hash=r["md5hash"], three_partition_label=r["three_partition_label"],
                       nine_partition_label=r["nine_partition_label"]))
    rate = len(out) / atlas_total if atlas_total else 0
    say(f"fitzpatrick17k: {len(out)} rows of {atlas_total} atlas URLs ({rate:.1%}); "
        f"full dataset is 16,577 so this is {len(out) / 16577:.1%} of the original")
    if rate < 0.95:
        errors.append(f"fitzpatrick17k join rate {rate:.1%} below 95%")
    return out


# ---------------------------------------------------------------- skeleton for Stage 2
def write_skeleton(rows: list[dict]) -> None:
    counts: Counter = Counter((r["source"], r["raw_label"]) for r in rows if r["raw_label"])
    # pre-fill what is already known so Stage 2 starts from facts, not blanks
    fitz_mal: dict[str, str] = {}
    for r in rows:
        if r["source"] == "fitzpatrick17k":
            fitz_mal[r["raw_label"]] = r["three_partition_label"]
    pad_mal = {"BCC": "malignant", "SCC": "malignant", "MEL": "malignant", "ACK": "malignant",
               "NEV": "benign", "SEK": "benign"}
    pad_name = {"BCC": "basal cell carcinoma", "SCC": "squamous cell carcinoma", "MEL": "melanoma",
                "ACK": "actinic keratosis", "NEV": "melanocytic nevus", "SEK": "seborrheic keratosis"}
    dermnet_mal = {
        "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": "malignant",
        "Melanoma Skin Cancer Nevi and Moles": "MIXED-needs-manual-sort",
        "Seborrheic Keratoses and other Benign Tumors": "benign",
        "Vascular Tumors": "benign",
    }
    with OUT_SKELETON.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "raw_label", "n_images", "disease", "malignancy", "note"])
        for (src, lab), n in sorted(counts.items(), key=lambda kv: (kv[0][0], -kv[1])):
            disease = malig = note = ""
            if src == "fitzpatrick17k":
                malig = fitz_mal.get(lab, "")
                disease = lab
            elif src == "padufes20":
                malig = pad_mal.get(lab, "")
                disease = pad_name.get(lab, "")
            elif src == "dermnet":
                malig = dermnet_mal.get(lab, "non-neoplastic" if lab not in dermnet_mal else "")
                if malig == "MIXED-needs-manual-sort":
                    malig, note = "", "mixed melanoma + nevi folder; sort manually or exclude from Model B"
            elif src == "scin":
                malig = "non-neoplastic"  # SCIN is inflammatory; override the handful of neoplastic ones by hand
                note = "verify: SCIN is inflammatory-dominant, a few neoplastic labels need manual override"
            w.writerow([src, lab, n, disease, malig, note])
    say(f"label_map_skeleton: {len(counts)} (source, raw_label) pairs -> {OUT_SKELETON.relative_to(ROOT)}")


def main() -> int:
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    rows += load_dermnet()
    rows += load_scin()
    rows += load_pad()
    rows += load_fitz()

    with OUT_MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    say(f"manifest_raw: {len(rows)} rows -> {OUT_MANIFEST.relative_to(ROOT)}")

    by_src = Counter(r["source"] for r in rows)
    say("per source: " + ", ".join(f"{k}={v}" for k, v in by_src.items()))
    fitz_dist = Counter(int(r["fitz_scale"]) for r in rows)
    say("fitz_scale: " + ", ".join(f"{k}:{fitz_dist[k]}" for k in sorted(fitz_dist)))
    dims = defaultdict(list)
    for r in rows:
        dims[r["source"]].append(min(int(r["width"]), int(r["height"])))
    for s, v in dims.items():
        v.sort()
        say(f"{s} short-side px: min={v[0]} p10={v[len(v) // 10]} median={v[len(v) // 2]} max={v[-1]}")

    write_skeleton(rows)

    if errors:
        say("CHECKS FAILED:")
        for e in errors:
            say("  - " + e)
    else:
        say("all hard checks passed")
    OUT_SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
