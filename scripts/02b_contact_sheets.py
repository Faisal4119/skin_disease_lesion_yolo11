"""Stage 2b — contact sheets of DermNet folder 11 for visual review.

Builds data/work/folder11_sheets/sheet_NNN.jpg, each a 3x2 grid of six images at 480 px
cells, every cell captioned with its global index. Also writes sheet_index.csv mapping
(sheet, cell) -> image path so labels typed per sheet can be joined back.

Usage:
    python scripts/02b_contact_sheets.py            # build sheets
    python scripts/02b_contact_sheets.py label 12 "M N U N N M" [annotator]
        -> appends six labels for sheet 12, cells 1..6, to data/dermnet_folder11_labels.csv
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FOLDER = "Melanoma Skin Cancer Nevi and Moles"
IMG_DIRS = [ROOT / "data" / "raw" / "dermnet" / s / FOLDER for s in ("train", "test")]
WORK = ROOT / "data" / "work" / "folder11_sheets"
INDEX = WORK / "sheet_index.csv"
OUT_CSV = ROOT / "data" / "dermnet_folder11_labels.csv"

COLS, ROWS, CELL, PAD = 3, 2, 480, 8
PER = COLS * ROWS
KEY = {"M": "melanoma", "N": "nevus", "U": "unsure", "?": "unsure"}


def items() -> list[Path]:
    return sorted(p for d in IMG_DIRS for p in d.glob("*.jpg"))


def font(size: int):
    for f in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    paths = items()
    fnt = font(26)
    rows = []
    for s in range(0, len(paths), PER):
        chunk = paths[s:s + PER]
        W = COLS * CELL + (COLS + 1) * PAD
        H = ROWS * (CELL + 34) + (ROWS + 1) * PAD
        sheet = Image.new("RGB", (W, H), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        for c, p in enumerate(chunk):
            im = Image.open(p).convert("RGB")
            im.thumbnail((CELL, CELL))
            r, col = divmod(c, COLS)
            x = PAD + col * (CELL + PAD)
            y = PAD + r * (CELL + 34 + PAD)
            sheet.paste(im, (x + (CELL - im.width) // 2, y + (CELL - im.height) // 2))
            draw.rectangle([x, y + CELL, x + CELL, y + CELL + 32], fill=(40, 40, 40))
            draw.text((x + 8, y + CELL + 3), f"cell {c + 1}   #{s + c}", fill=(255, 220, 90), font=fnt)
            rows.append({"sheet": s // PER, "cell": c + 1, "index": s + c, "path": p.relative_to(ROOT).as_posix()})
        sheet.save(WORK / f"sheet_{s // PER:03d}.jpg", quality=88)
    with INDEX.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sheet", "cell", "index", "path"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(paths)} images -> {len(rows) // PER + (1 if len(rows) % PER else 0)} sheets in {WORK.relative_to(ROOT)}")


def label(sheet: int, keys: str, annotator: str = "claude") -> None:
    with INDEX.open(encoding="utf-8") as f:
        cells = {(int(r["sheet"]), int(r["cell"])): r["path"] for r in csv.DictReader(f)}
    toks = keys.upper().split()
    n_cells = sum(1 for (s, _) in cells if s == sheet)
    if len(toks) != n_cells:
        raise SystemExit(f"sheet {sheet} has {n_cells} cells, got {len(toks)} labels")
    new = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["path", "label", "annotator", "ts"])
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        for i, t in enumerate(toks, 1):
            w.writerow([cells[(sheet, i)], KEY[t], annotator, ts])
    print(f"sheet {sheet}: {' '.join(KEY[t][0].upper() for t in toks)} written")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "label":
        label(int(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "claude")
    else:
        build()
