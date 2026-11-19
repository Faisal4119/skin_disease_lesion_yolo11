"""Stage 0a — fetch DermNet from Hugging Face and export to an ImageFolder tree.

Source: https://huggingface.co/datasets/exper1ment/dermnet  (19,559 images, 23 classes)
Output: data/raw/dermnet/{train,test}/<class>/<idx>.jpg  +  data/raw/dermnet/_manifest.csv

Resume-safe: existing files with non-zero size are skipped.
"""
from __future__ import annotations

import csv
import hashlib
import sys
import time
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "dermnet"
LOG = ROOT / "logs" / "00a_dermnet.log"
REPO = "exper1ment/dermnet"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log(f"loading {REPO} (downloads parquet shards to HF cache on first run)")
    ds = load_dataset(REPO)
    names = ds["train"].features["label"].names
    log(f"classes={len(names)} splits={ {k: len(v) for k, v in ds.items()} }")

    manifest_path = OUT / "_manifest.csv"
    rows: list[dict] = []
    written = skipped = failed = 0
    t0 = time.time()

    for split in ds.keys():
        sub = ds[split]
        total = len(sub)
        for i, ex in enumerate(sub):
            cls = names[ex["label"]]
            safe_cls = cls.replace("/", "-")
            dst = OUT / split / safe_cls / f"{i:06d}.jpg"
            try:
                if dst.exists() and dst.stat().st_size > 0:
                    skipped += 1
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    img = ex["image"]
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(dst, format="JPEG", quality=95)
                    written += 1
                rows.append(
                    {
                        "path": dst.relative_to(ROOT).as_posix(),
                        "source": "dermnet",
                        "split_hint": split,
                        "raw_label": cls,
                        "bytes": dst.stat().st_size,
                        "sha1": sha1_of(dst),
                    }
                )
            except Exception as e:  # noqa: BLE001
                failed += 1
                log(f"FAIL {split}[{i}] {cls}: {e}")
            if (i + 1) % 1000 == 0 or i + 1 == total:
                el = time.time() - t0
                log(f"{split} {i + 1}/{total}  written={written} skipped={skipped} failed={failed}  {el:.0f}s")

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"DONE rows={len(rows)} written={written} skipped={skipped} failed={failed} -> {manifest_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
