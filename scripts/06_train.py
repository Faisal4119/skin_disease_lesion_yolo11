"""Stage 6 — train Model A (disease) or Model B (malignancy) with Ultralytics YOLO-cls.

Usage
    python scripts/06_train.py --task disease    --smoke     # 1 epoch on 10%: proves the pipeline end to end (~3 min)
    python scripts/06_train.py --task malignancy --smoke
    python scripts/06_train.py --task disease                # full run, ~1-1.5 h on the RTX 4060
    python scripts/06_train.py --task malignancy             # full run, ~45 min

Run the two full runs one after the other, never at the same time: two trainers on one
8 GB card contend for VRAM and both slow down more than they gain.

Class imbalance
    The classification trainer exposes no class weights and no sampler hook, so the
    balancing is done at the data layer: a `<dataset>_bal/` tree is generated whose
    train/ split oversamples minority classes with NTFS hard links (zero extra bytes).
    Replication factor per class = min(MAX_REP, sqrt(max_count / count)), i.e. square-
    root balancing — it lifts rare classes without drowning the common ones in copies.
    val/ and test/ are linked 1:1, untouched. The realised class distribution is printed
    so the intent can be verified, not assumed.

Augmentation
    Hue/saturation carry diagnostic signal in dermatology (erythema, pigment, vascularity)
    and encode skin tone, so colour jitter is kept mild and RandAugment (which includes
    Solarize/Posterize/strong colour ops) is switched off. Geometric augmentation is free:
    both flips, random-resized-crop down to 60% area, mild random erasing.

Outputs
    runs/<task>/weights/best.pt, last.pt, results.csv, confusion_matrix.png
    runs/<task>/test_metrics.json   (top1/top5 on the held-out test split)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = ROOT / "data" / "yolo"
RUNS = ROOT / "runs"

MAX_REP = 4.0  # never replicate a class more than 4x

DEFAULTS = {
    "disease":    {"epochs": 60, "patience": 12},
    "malignancy": {"epochs": 40, "patience": 10},
}


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_balanced(task: str) -> Path:
    """Create data/yolo/dataset_<task>_bal with a sqrt-oversampled train split."""
    src = YOLO_DIR / f"dataset_{task}"
    dst = YOLO_DIR / f"dataset_{task}_bal"
    counts = {d.name: sum(1 for _ in d.iterdir()) for d in (src / "train").iterdir() if d.is_dir()}
    mx = max(counts.values())
    print(f"\n[{task}] balancing train split  (max class = {mx})")
    print(f"  {'class':36s} {'orig':>6s} {'rep':>5s} {'after':>7s}")
    realised: Counter = Counter()
    for cls, n in sorted(counts.items(), key=lambda kv: kv[1]):
        rep = min(MAX_REP, math.sqrt(mx / n))
        target = int(round(n * rep))
        files = sorted((src / "train" / cls).iterdir())
        k = 0
        for i in range(target):
            f = files[i % n]
            copy_idx = i // n
            name = f.name if copy_idx == 0 else f"{f.stem}__dup{copy_idx}{f.suffix}"
            link_or_copy(f, dst / "train" / cls / name)
            k += 1
        realised[cls] = k
        print(f"  {cls:36s} {n:6d} {rep:5.2f} {k:7d}")
    for split in ("val", "test"):
        for cls_dir in (src / split).iterdir():
            for f in cls_dir.iterdir():
                link_or_copy(f, dst / split / cls_dir.name / f.name)
    print(f"  train total: {sum(counts.values())} -> {sum(realised.values())}")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["disease", "malignancy"], required=True)
    ap.add_argument("--model", default="yolo11s-cls.pt")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-balance", action="store_true", help="train on the raw class distribution")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 10%% of the data, small batch")
    ap.add_argument("--name", help="run name (default: task, or task_smoke)")
    a = ap.parse_args()

    from ultralytics import YOLO  # deferred: slow import, keep --help fast

    task = a.task
    epochs = a.epochs or DEFAULTS[task]["epochs"]
    patience = DEFAULTS[task]["patience"]
    fraction = 1.0
    batch = a.batch
    if a.smoke:
        epochs, fraction, batch = 1, 0.1, min(a.batch, 32)
    name = a.name or (f"{task}_smoke" if a.smoke else task)

    data = YOLO_DIR / f"dataset_{task}" if a.no_balance else build_balanced(task)

    print(f"\n[{task}] model={a.model} data={data.relative_to(ROOT)} epochs={epochs} imgsz={a.imgsz} "
          f"batch={batch} fraction={fraction} -> runs/{name}")
    t0 = time.time()
    model = YOLO(a.model)
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=a.imgsz,
        batch=batch,
        workers=a.workers,
        device=0,
        project=str(RUNS),
        name=name,
        exist_ok=True,
        seed=42,
        patience=patience,
        cos_lr=True,
        fraction=fraction,
        # colour: mild. geometry: free. RandAugment off (see module docstring).
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        fliplr=0.5, flipud=0.5,
        scale=0.4,
        erasing=0.2,
        auto_augment=None,
        plots=True,
        verbose=True,
    )
    train_min = (time.time() - t0) / 60
    print(f"\n[{task}] training done in {train_min:.1f} min")

    # held-out test split, never seen during training or early stopping
    best = RUNS / name / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(data), split="test", imgsz=a.imgsz, batch=batch, device=0,
                            project=str(RUNS), name=f"{name}_test", exist_ok=True, plots=True)
    metrics = {"task": task, "model": a.model, "epochs": epochs, "imgsz": a.imgsz, "smoke": a.smoke,
               "train_minutes": round(train_min, 1),
               "test_top1": float(m.top1), "test_top5": float(m.top5),
               "classes": sorted(d.name for d in (data / "train").iterdir() if d.is_dir())}
    (RUNS / name / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[{task}] TEST top1={m.top1:.4f} top5={m.top5:.4f}  -> runs/{name}/test_metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
