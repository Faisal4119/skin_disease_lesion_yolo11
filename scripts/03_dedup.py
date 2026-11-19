"""Stage 3 — perceptual-hash deduplication across all four sources.

Reads  data/manifest_raw.csv
Writes data/phash.csv            path, phash (cache; re-runs skip hashed files)
       data/dedup_groups.csv     one row per duplicate group member, for audit
       data/manifest_dedup.csv   manifest_raw minus dropped duplicates, + dup_group column
       data/dedup_summary.txt

Method: 64-bit pHash (imagehash.phash, hash_size=8). Two images are duplicates when
their Hamming distance is <= HAMMING_MAX. Candidate pairs are found with 4-band LSH on
16-bit slices of the hash (any shared band -> candidate), then verified exactly, then
merged with union-find. Within a group the highest-resolution image is kept.

Runs BEFORE the split (Stage 4) so a duplicate can never straddle train/val.
"""
from __future__ import annotations

import csv
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest_raw.csv"
PHASH = ROOT / "data" / "phash.csv"
GROUPS = ROOT / "data" / "dedup_groups.csv"
OUT = ROOT / "data" / "manifest_dedup.csv"
SUMMARY = ROOT / "data" / "dedup_summary.txt"

HAMMING_MAX = 5
BANDS = 4          # 64 bits / 4 = 16-bit bands
WORKERS = 12


def phash_one(rel: str) -> tuple[str, str]:
    try:
        with Image.open(ROOT / rel) as im:
            im = im.convert("L")
            im.thumbnail((512, 512))          # hashing does not need full resolution
            return rel, str(imagehash.phash(im, hash_size=8))
    except Exception as e:  # noqa: BLE001
        return rel, f"ERR:{e}"


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def main() -> int:
    lines: list[str] = []

    def say(m: str) -> None:
        print(m, flush=True)
        lines.append(m)

    with MANIFEST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    say(f"manifest_raw rows: {len(rows)}")

    # ---- hashes (cached)
    cache: dict[str, str] = {}
    if PHASH.exists():
        with PHASH.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cache[r["path"]] = r["phash"]
    todo = [r["path"] for r in rows if r["path"] not in cache]
    say(f"cached hashes: {len(cache)}  to compute: {len(todo)}")
    if todo:
        t0 = time.time()
        with ProcessPoolExecutor(WORKERS) as ex:
            for i, (rel, h) in enumerate(ex.map(phash_one, todo, chunksize=64), 1):
                cache[rel] = h
                if i % 5000 == 0 or i == len(todo):
                    say(f"  hashed {i}/{len(todo)}  {time.time() - t0:.0f}s")
        with PHASH.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "phash"])
            for rel in sorted(cache):
                w.writerow([rel, cache[rel]])
    errs = [p for p, h in cache.items() if h.startswith("ERR")]
    if errs:
        say(f"hash errors: {len(errs)} (kept, not deduplicated): {errs[:5]}")

    # ---- LSH candidates + exact verify + union-find
    idx = {r["path"]: i for i, r in enumerate(rows)}
    hv: list[int | None] = [None] * len(rows)
    for r in rows:
        h = cache.get(r["path"], "")
        hv[idx[r["path"]]] = None if (not h or h.startswith("ERR")) else int(h, 16)

    buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(BANDS)]
    for i, h in enumerate(hv):
        if h is None:
            continue
        for b in range(BANDS):
            buckets[b][(h >> (16 * b)) & 0xFFFF].append(i)

    dsu = DSU(len(rows))
    checked = 0
    pairs = 0
    seen: set[tuple[int, int]] = set()
    for b in range(BANDS):
        for members in buckets[b].values():
            if len(members) < 2:
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    i, j = members[x], members[y]
                    key = (i, j) if i < j else (j, i)
                    if key in seen:
                        continue
                    seen.add(key)
                    checked += 1
                    if hamming(hv[i], hv[j]) <= HAMMING_MAX:
                        dsu.union(i, j)
                        pairs += 1
    say(f"candidate pairs checked: {checked}  duplicate pairs: {pairs}")

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        groups[dsu.find(i)].append(i)
    dup_groups = {g: m for g, m in groups.items() if len(m) > 1}

    # Union-find is single-linkage: A~B (d=4) and B~C (d=4) can chain A and C at d=8.
    # Verified 2026-09-20: one such chain glued three PAD nevi to two DermNet images.
    # Require complete linkage — every pair within HAMMING_MAX — else dissolve the group.
    dissolved = 0
    for g in list(dup_groups):
        m = dup_groups[g]
        diameter = max(hamming(hv[a], hv[b]) for a in m for b in m if a < b)
        if diameter > HAMMING_MAX:
            dissolved += 1
            del dup_groups[g]
    say(f"duplicate groups: {len(dup_groups)}  images in groups: {sum(len(m) for m in dup_groups.values())}"
        f"  (dissolved {dissolved} chained groups with diameter > {HAMMING_MAX})")

    # ---- choose representative, write audit
    drop: set[int] = set()
    cross_source = 0
    label_conflict = 0
    per_source_drop: Counter = Counter()
    gid_of: dict[int, int] = {}
    with GROUPS.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dup_group", "kept", "path", "source", "raw_label", "width", "height", "phash"])
        for gnum, (g, members) in enumerate(sorted(dup_groups.items(), key=lambda kv: -len(kv[1]))):
            members.sort(key=lambda i: -(int(rows[i]["width"]) * int(rows[i]["height"])))
            keep = members[0]
            srcs = {rows[i]["source"] for i in members}
            labs = {(rows[i]["source"], rows[i]["raw_label"]) for i in members}
            if len(srcs) > 1:
                cross_source += 1
            if len(labs) > 1:
                label_conflict += 1
            for i in members:
                gid_of[i] = gnum
                if i != keep:
                    drop.add(i)
                    per_source_drop[rows[i]["source"]] += 1
                r = rows[i]
                w.writerow([gnum, int(i == keep), r["path"], r["source"], r["raw_label"], r["width"], r["height"], cache.get(r["path"], "")])
    say(f"cross-source groups: {cross_source}  groups with differing labels: {label_conflict}")
    say(f"dropped: {len(drop)}  per source: {dict(per_source_drop)}")

    kept_rows = []
    for i, r in enumerate(rows):
        if i in drop:
            continue
        r = dict(r)
        r["dup_group"] = gid_of.get(i, "")
        r["dup_dropped"] = len(dup_groups[dsu.find(i)]) - 1 if i in gid_of else 0
        kept_rows.append(r)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(kept_rows[0].keys()))
        w.writeheader()
        w.writerows(kept_rows)
    say(f"manifest_dedup: {len(kept_rows)} rows -> {OUT.relative_to(ROOT)}")
    say("per source after dedup: " + ", ".join(f"{k}={v}" for k, v in Counter(r["source"] for r in kept_rows).items()))
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
