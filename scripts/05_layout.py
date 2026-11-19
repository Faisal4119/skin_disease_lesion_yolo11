"""Stage 5 — materialise the two ImageFolder trees `yolo classify` expects.

Reads  data/manifest_final.csv
Writes data/yolo/dataset_disease/{train,val,test}/<disease>/<img>.jpg      (model_a == 1)
       data/yolo/dataset_malignancy/{train,val,test}/<malignancy>/<img>.jpg (model_b == 1)
       data/yolo/classes_disease.txt, classes_malignancy.txt  (sorted class order)

Files are NTFS hard links to data/processed (same volume, no admin needed, zero extra
bytes). Falls back to a copy if linking fails. Both trees are generated from the one
manifest so they cannot drift apart. Re-running is safe: existing links are kept.
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest_final.csv"
YOLO = ROOT / "data" / "yolo"


def place(src: Path, dst: Path) -> str:
    if dst.exists():
        return "kept"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def flat_name(rel: str) -> str:
    # data/processed/dermnet/train/Eczema Photos/000123.jpg -> dermnet__train__Eczema_Photos__000123.jpg
    p = Path(rel)
    parts = p.relative_to("data/processed").parts
    return "__".join(x.replace(" ", "_") for x in parts)


def main() -> int:
    with MANIFEST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    st: Counter = Counter()
    counts = {"disease": defaultdict(Counter), "malignancy": defaultdict(Counter)}
    for r in rows:
        src = ROOT / r["path"]
        name = flat_name(r["path"])
        if r["model_a"] == "1":
            dst = YOLO / "dataset_disease" / r["split"] / r["disease"] / name
            st["disease_" + place(src, dst)] += 1
            counts["disease"][r["split"]][r["disease"]] += 1
        if r["model_b"] == "1":
            dst = YOLO / "dataset_malignancy" / r["split"] / r["malignancy"] / name
            st["malignancy_" + place(src, dst)] += 1
            counts["malignancy"][r["split"]][r["malignancy"]] += 1

    for task in ("disease", "malignancy"):
        classes = sorted({c for s in counts[task].values() for c in s})
        (YOLO / f"classes_{task}.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
        # every class must exist in every split, else ultralytics maps indices inconsistently
        for split, c in counts[task].items():
            missing = [k for k in classes if c[k] == 0]
            if missing:
                print(f"ERROR: {task}/{split} missing classes {missing}")
                return 1
        tot = {s: sum(c.values()) for s, c in counts[task].items()}
        print(f"{task:11s} classes={len(classes):2d}  " + "  ".join(f"{s}={n}" for s, n in tot.items()))
    print("ops:", dict(st))
    return 0


if __name__ == "__main__":
    sys.exit(main())
