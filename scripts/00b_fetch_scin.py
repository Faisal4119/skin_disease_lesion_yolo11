"""Stage 0b — fetch SCIN (Google) from its public GCS bucket over plain HTTPS.

Bucket: gs://dx-scin-public-data  (no auth; listable and fetchable over HTTP)
Output: data/raw/scin/images/*.png, data/raw/scin/scin_cases.csv, scin_labels.csv,
        data/raw/scin/_manifest.csv

Every object is verified against the md5Hash GCS reports in the listing.
Resume-safe: an existing file whose md5 matches is skipped.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "scin"
IMG = OUT / "images"
LOG = ROOT / "logs" / "00b_scin.log"

BUCKET = "dx-scin-public-data"
LIST_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
OBJ_URL = f"https://storage.googleapis.com/{BUCKET}/"
WORKERS = 8
UA = {"User-Agent": "Mozilla/5.0 (skin-dataset-fetch)"}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def list_objects(prefix: str) -> list[dict]:
    items: list[dict] = []
    token = None
    while True:
        params = {"prefix": prefix, "maxResults": 1000}
        if token:
            params["pageToken"] = token
        r = requests.get(LIST_URL, params=params, headers=UA, timeout=60)
        r.raise_for_status()
        d = r.json()
        items.extend(d.get("items", []))
        token = d.get("nextPageToken")
        if not token:
            return items


def fetch_one(obj: dict) -> tuple[str, str, int]:
    """Returns (name, status, bytes). status in {ok, skipped, fail}."""
    name = obj["name"]
    dst = IMG / Path(name).name
    want = obj["md5Hash"]
    if dst.exists() and md5_b64(dst) == want:
        return name, "skipped", dst.stat().st_size
    for attempt in range(4):
        try:
            r = requests.get(OBJ_URL + name, headers=UA, timeout=120, stream=True)
            r.raise_for_status()
            tmp = dst.with_suffix(dst.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
            if md5_b64(tmp) != want:
                tmp.unlink(missing_ok=True)
                raise ValueError("md5 mismatch")
            tmp.replace(dst)
            return name, "ok", dst.stat().st_size
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return name, f"fail:{e}", 0
            time.sleep(2 * (attempt + 1))
    return name, "fail", 0


def main() -> int:
    IMG.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    for csv_name in ("scin_cases.csv", "scin_labels.csv", "scin_label_questions.csv", "scin_app_questions.csv"):
        r = requests.get(OBJ_URL + "dataset/" + csv_name, headers=UA, timeout=120)
        r.raise_for_status()
        (OUT / csv_name).write_bytes(r.content)
        log(f"csv {csv_name} {len(r.content)} bytes")

    objs = list_objects("dataset/images/")
    total_bytes = sum(int(o["size"]) for o in objs)
    log(f"listed {len(objs)} images, {total_bytes / 1e9:.2f} GB")

    ok = skipped = failed = 0
    done_bytes = 0
    failures: list[str] = []
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(fetch_one, o) for o in objs]
        for n, fut in enumerate(as_completed(futs), 1):
            name, status, nbytes = fut.result()
            done_bytes += nbytes
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append(f"{name}\t{status}")
            if n % 250 == 0 or n == len(objs):
                el = time.time() - t0
                rate = done_bytes / 1e6 / max(el, 1)
                log(f"{n}/{len(objs)}  ok={ok} skipped={skipped} failed={failed}  "
                    f"{done_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB  {rate:.1f} MB/s")

    if failures:
        (OUT / "_failed.txt").write_text("\n".join(failures), encoding="utf-8")

    with (OUT / "_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "bytes", "md5_b64"])
        for o in objs:
            p = IMG / Path(o["name"]).name
            if p.exists():
                w.writerow([p.relative_to(ROOT).as_posix(), "scin", p.stat().st_size, o["md5Hash"]])
    log(f"DONE ok={ok} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
