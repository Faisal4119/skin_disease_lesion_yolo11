"""Stage 4 — apply labels, filter, split, resize.

Reads  data/manifest_dedup.csv
       data/label_map.csv                 (Stage 2c)
       data/dermnet_folder11_labels.csv   (Stage 2b, per-image melanoma/nevus/unsure)
       data/dedup_groups.csv              (Stage 3, all members of every duplicate group)
Writes data/processed/<source>/...jpg    long edge 512 px, JPEG q92
       data/manifest_final.csv           one row per kept image
       data/split_summary.txt

Rules, in order:
  1. label_map -> (disease, malignancy, model_a, model_b). Folder 11 resolved per image.
  2. Duplicate-group conflicts: members' canonical diseases differ -> model_a = 0;
     malignancies differ -> drop the row (cannot trust the cancer label either way).
  3. Short side < MIN_SIDE px -> drop.
  4. Model A classes with < MIN_IMAGES images -> model_a = 0 (row stays for Model B).
  5. StratifiedGroupKFold(20, shuffle, seed 42) on disease, grouped on group_id:
     folds 0-13 train (70%), 14-16 val (15%), 17-19 test (15%). Written once, never re-randomised.
  6. Resize + write.
"""
from __future__ import annotations

import csv
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
MANIFEST = D / "manifest_dedup.csv"
LABEL_MAP = D / "label_map.csv"
F11 = D / "dermnet_folder11_labels.csv"
GROUPS = D / "dedup_groups.csv"
PROCESSED = D / "processed"
OUT = D / "manifest_final.csv"
SUMMARY = D / "split_summary.txt"

MIN_SIDE = 200
MIN_IMAGES = 100
LONG_EDGE = 512
JPEG_Q = 92
N_FOLDS = 20
TRAIN_FOLDS, VAL_FOLDS = 14, 3      # 14/20 = 70%, 3/20 = 15%, rest 15%
SEED = 42
WORKERS = 12

PER_IMAGE = "__per_image__"
EXCLUDE = "__exclude__"
F11_MAP = {"melanoma": ("melanoma", "malignant"), "nevus": ("melanocytic_nevus", "benign")}

lines: list[str] = []


def say(m: str) -> None:
    print(m, flush=True)
    lines.append(m)


def load_label_map() -> dict[tuple[str, str], dict]:
    with LABEL_MAP.open(encoding="utf-8") as f:
        return {(r["source"], r["raw_label"]): r for r in csv.DictReader(f)}


def load_f11() -> dict[str, str]:
    last: dict[str, str] = {}
    if F11.exists():
        with F11.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                last[r["path"]] = r["label"]
    return last


def canonical(src: str, raw: str, path: str, lm: dict, f11: dict) -> tuple[str, str, int, int] | None:
    """-> (disease, malignancy, model_a, model_b) or None when the image is excluded."""
    m = lm.get((src, raw))
    if m is None or m["disease"] == EXCLUDE:
        return None
    if m["disease"] == PER_IMAGE:
        lab = f11.get(path, "unsure")
        if lab not in F11_MAP:
            return None
        dis, mal = F11_MAP[lab]
        return dis, mal, 1, 1
    return m["disease"], m["malignancy"], int(m["model_a"]), int(m["model_b"])


def resize_one(args: tuple[str, str]) -> tuple[str, int, int, str]:
    src_rel, dst_rel = args
    dst = ROOT / dst_rel
    try:
        if dst.exists() and dst.stat().st_size > 0:
            with Image.open(dst) as im:
                return dst_rel, im.width, im.height, "skipped"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(ROOT / src_rel) as im:
            im = im.convert("RGB")
            im.thumbnail((LONG_EDGE, LONG_EDGE), Image.LANCZOS)
            im.save(dst, format="JPEG", quality=JPEG_Q, optimize=True)
            return dst_rel, im.width, im.height, "ok"
    except Exception as e:  # noqa: BLE001
        return dst_rel, 0, 0, f"ERR:{e}"


def main() -> int:
    lm = load_label_map()
    f11 = load_f11()
    with MANIFEST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    say(f"manifest_dedup rows: {len(rows)}")

    # ---- 1. labels
    drops: Counter = Counter()
    kept: list[dict] = []
    for r in rows:
        c = canonical(r["source"], r["raw_label"], r["path"], lm, f11)
        if c is None:
            drops["excluded_by_label_map_or_unsure"] += 1
            continue
        r = dict(r)
        r["disease"], r["malignancy"], r["model_a"], r["model_b"] = c
        kept.append(r)
    say(f"after label map: {len(kept)}  dropped={drops['excluded_by_label_map_or_unsure']}")

    # ---- 2. duplicate-group conflicts (members may already be gone; use the audit file)
    members: dict[str, list[dict]] = defaultdict(list)
    with GROUPS.open(encoding="utf-8") as f:
        for g in csv.DictReader(f):
            members[g["dup_group"]].append(g)
    conflict_a: set[str] = set()
    conflict_drop: set[str] = set()
    for gid, ms in members.items():
        cans = []
        for g in ms:
            c = canonical(g["source"], g["raw_label"], g["path"], lm, f11)
            if c is not None:
                cans.append(c)
        if len(cans) < 2:
            continue
        if len({c[1] for c in cans}) > 1:
            conflict_drop.add(gid)
        elif len({c[0] for c in cans}) > 1:
            conflict_a.add(gid)
    n_a = n_d = 0
    tmp = []
    for r in kept:
        gid = r.get("dup_group", "")
        if gid and gid in conflict_drop:
            n_d += 1
            continue
        if gid and gid in conflict_a and r["model_a"]:
            r["model_a"] = 0
            n_a += 1
        tmp.append(r)
    kept = tmp
    drops["dup_group_malignancy_conflict"] = n_d
    say(f"dup conflicts: {n_d} rows dropped (malignancy conflict), {n_a} rows Model A only-off (disease conflict)")

    # ---- 3. size
    tmp = []
    for r in kept:
        if min(int(r["width"]), int(r["height"])) < MIN_SIDE:
            drops[f"short_side_lt_{MIN_SIDE}"] += 1
            continue
        tmp.append(r)
    kept = tmp
    say(f"after size filter: {len(kept)}  dropped={drops[f'short_side_lt_{MIN_SIDE}']}")

    # ---- 4. Model A class floor
    a_counts = Counter(r["disease"] for r in kept if r["model_a"])
    small = {d for d, n in a_counts.items() if n < MIN_IMAGES}
    n_floor = 0
    for r in kept:
        if r["model_a"] and r["disease"] in small:
            r["model_a"] = 0
            n_floor += 1
    say(f"Model A floor {MIN_IMAGES}: dropped classes {sorted(small)} ({n_floor} rows -> Model B only)")

    # ---- 5. split (grouped, stratified on disease, across everything Model B sees)
    y = np.array([r["disease"] for r in kept])
    groups = np.array([r["group_id"] for r in kept])
    fold = np.full(len(kept), -1)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(kept)), y, groups)):
        fold[test_idx] = k
    assert (fold >= 0).all()
    for r, k in zip(kept, fold):
        r["fold"] = int(k)
        r["split"] = "train" if k < TRAIN_FOLDS else ("val" if k < TRAIN_FOLDS + VAL_FOLDS else "test")
    # leakage check: no group in two splits
    g2s: dict[str, set[str]] = defaultdict(set)
    for r in kept:
        g2s[r["group_id"]].add(r["split"])
    leaks = sum(1 for s in g2s.values() if len(s) > 1)
    say(f"split: {Counter(r['split'] for r in kept)}  groups spanning splits: {leaks}")
    if leaks:
        say("ERROR: grouped split leaked")
        return 1

    # ---- 6. resize
    jobs = []
    for r in kept:
        rel = Path(r["path"])
        dst_rel = (Path("data/processed") / rel.relative_to("data/raw")).with_suffix(".jpg").as_posix()
        r["raw_path"] = r["path"]
        r["path"] = dst_rel
        jobs.append((r["raw_path"], dst_rel))
    t0 = time.time()
    st: Counter = Counter()
    dims: dict[str, tuple[int, int]] = {}
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, (dst_rel, w, h, s) in enumerate(ex.map(resize_one, jobs, chunksize=32), 1):
            st["ok" if s == "ok" else ("skipped" if s == "skipped" else "err")] += 1
            dims[dst_rel] = (w, h)
            if i % 5000 == 0 or i == len(jobs):
                say(f"  resized {i}/{len(jobs)}  ok={st['ok']} skipped={st['skipped']} err={st['err']}  {time.time() - t0:.0f}s")
    final = []
    for r in kept:
        w, h = dims.get(r["path"], (0, 0))
        if w == 0:
            drops["resize_error"] += 1
            continue
        r["width"], r["height"] = w, h
        final.append(r)

    # ---- write
    cols = ["path", "raw_path", "source", "disease", "malignancy", "model_a", "model_b", "split", "fold",
            "group_id", "fitz_scale", "width", "height", "raw_label", "case_id", "patient_id", "lesion_id",
            "label_weight", "md5hash", "dup_group"]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(final)
    say(f"\nmanifest_final: {len(final)} rows -> {OUT.relative_to(ROOT)}")
    say("drop reasons: " + ", ".join(f"{k}={v}" for k, v in drops.items()))

    say("\nModel A  (disease x split)")
    ta = defaultdict(Counter)
    for r in final:
        if r["model_a"]:
            ta[r["disease"]][r["split"]] += 1
    say(f"  {'disease':36s} {'train':>6s} {'val':>5s} {'test':>5s} {'total':>6s}")
    for d, c in sorted(ta.items(), key=lambda kv: -sum(kv[1].values())):
        say(f"  {d:36s} {c['train']:6d} {c['val']:5d} {c['test']:5d} {sum(c.values()):6d}")
    say(f"  classes: {len(ta)}  images: {sum(sum(c.values()) for c in ta.values())}")

    say("\nModel B  (malignancy x split)")
    tb = defaultdict(Counter)
    for r in final:
        if r["model_b"]:
            tb[r["malignancy"]][r["split"]] += 1
    for m, c in tb.items():
        say(f"  {m:15s} {c['train']:6d} {c['val']:5d} {c['test']:5d} {sum(c.values()):6d}")
    mel = Counter(r["split"] for r in final if r["disease"] == "melanoma")
    say(f"  melanoma alone: {dict(mel)}")

    say("\nfitz_scale coverage: " + str(dict(Counter(int(r['fitz_scale']) for r in final))))
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
