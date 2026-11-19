"""Stage 0c — recover the live 24% of Fitzpatrick17k from atlasdermatologico.com.br.

Input:  data/raw/fitzpatrick17k.csv  (official label CSV, 16,577 rows)
Output: data/raw/fitzpatrick17k/images/<md5hash>.jpg
        data/raw/fitzpatrick17k/_manifest.csv, _failed.csv

Only rows whose URL points at atlasdermatologico.com.br are attempted; dermaamin.com
no longer resolves. Throttled to ONE request per second — small academic host.
Each download is verified against the CSV's md5hash (MD5 of the image bytes).
Resume-safe: an existing file whose md5 matches is skipped without a request.
"""
from __future__ import annotations

import csv
import hashlib
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CSV_IN = ROOT / "data" / "raw" / "fitzpatrick17k.csv"
OUT = ROOT / "data" / "raw" / "fitzpatrick17k"
IMG = OUT / "images"
REJ = OUT / "_rejected"
LOG = ROOT / "logs" / "00c_fitz_atlas.log"

HOST = "atlasdermatologico.com.br"
DELAY_S = 1.0
UA = {"User-Agent": "Mozilla/5.0 (skin-dataset-fetch; academic use)"}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def md5_hex(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def main() -> int:
    IMG.mkdir(parents=True, exist_ok=True)
    REJ.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    with CSV_IN.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if HOST in (r.get("url") or "")]
    log(f"{len(rows)} rows point at {HOST}")

    ok = skipped = failed = rejected = 0
    manifest: list[dict] = []
    failures: list[dict] = []
    sess = requests.Session()
    t0 = time.time()
    last_req = 0.0

    for i, r in enumerate(rows, 1):
        md5 = r["md5hash"]
        dst = IMG / f"{md5}.jpg"
        if dst.exists() and md5_hex(dst.read_bytes()) == md5:
            skipped += 1
            manifest.append({**_row(r, dst), "status": "skipped"})
        else:
            wait = DELAY_S - (time.time() - last_req)
            if wait > 0:
                time.sleep(wait)
            status = "fail"
            for attempt in range(3):
                try:
                    last_req = time.time()
                    resp = sess.get(r["url"], headers=UA, timeout=45)
                    if resp.status_code != 200 or not resp.headers.get("Content-Type", "").startswith("image/"):
                        raise ValueError(f"http {resp.status_code} {resp.headers.get('Content-Type')}")
                    data = resp.content
                    if md5_hex(data) == md5:
                        dst.write_bytes(data)
                        ok += 1
                        status = "ok"
                        manifest.append({**_row(r, dst), "status": "ok"})
                    else:
                        (REJ / f"{md5}.jpg").write_bytes(data)
                        rejected += 1
                        status = "rejected"
                        failures.append({"md5hash": md5, "url": r["url"], "reason": "md5 mismatch"})
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        failed += 1
                        failures.append({"md5hash": md5, "url": r["url"], "reason": str(e)[:120]})
                    else:
                        time.sleep(3 * (attempt + 1))
        if i % 100 == 0 or i == len(rows):
            el = time.time() - t0
            log(f"{i}/{len(rows)}  ok={ok} skipped={skipped} rejected={rejected} failed={failed}  {el / 60:.1f} min")

    with (OUT / "_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()) if manifest else ["path"])
        w.writeheader()
        w.writerows(manifest)
    if failures:
        with (OUT / "_failed.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["md5hash", "url", "reason"])
            w.writeheader()
            w.writerows(failures)
    log(f"DONE ok={ok} skipped={skipped} rejected={rejected} failed={failed}")
    return 0 if failed == 0 else 1


def _row(r: dict, dst: Path) -> dict:
    return {
        "path": dst.relative_to(ROOT).as_posix(),
        "source": "fitzpatrick17k",
        "md5hash": r["md5hash"],
        "raw_label": r["label"],
        "nine_partition_label": r["nine_partition_label"],
        "three_partition_label": r["three_partition_label"],
        "fitzpatrick_scale": r["fitzpatrick_scale"],
        "bytes": dst.stat().st_size if dst.exists() else 0,
    }


if __name__ == "__main__":
    sys.exit(main())
