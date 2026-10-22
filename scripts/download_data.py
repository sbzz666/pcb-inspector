"""Download original-author DeepPCB files with bounded parallelism and retries.

Default downloads a deterministic, explicitly labelled CPU portfolio subset.
--full downloads all 1,500 pairs. No labels are inspected to select the subset.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://raw.githubusercontent.com/tangsanli5201/DeepPCB/master/"


def fetch(relative):
    target = ROOT / "data/raw/DeepPCB" / relative
    if target.exists() and target.stat().st_size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            request = urllib.request.Request(BASE + relative, headers={"User-Agent": "PCBInspector-Portfolio/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read()
            if not data:
                raise ValueError("Empty download")
            temp = target.with_suffix(target.suffix + ".part")
            temp.write_bytes(data)
            temp.replace(target)
            return target
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def choose(rows, count):
    # Hash order distributes selection over source groups, independent of labels.
    return sorted(rows, key=lambda row: hashlib.sha256(row.encode()).hexdigest())[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--trainval", type=int, default=400)
    parser.add_argument("--test", type=int, default=150)
    args = parser.parse_args()
    root = ROOT / "data/raw/DeepPCB"
    for filename in ["README.md", "LICENSE", "PCBData/trainval.txt", "PCBData/test.txt"]:
        fetch(filename)
    selections = {}
    for split, count in [("trainval", args.trainval), ("test", args.test)]:
        rows = (root / f"PCBData/{split}.txt").read_text().strip().splitlines()
        selections[split] = rows if args.full else choose(rows, count)
    files = []
    for rows in selections.values():
        for row in rows:
            image, label = row.split()
            stem = image.rsplit(".", 1)[0]
            files.extend([f"PCBData/{stem}_test.jpg", f"PCBData/{stem}_temp.jpg", f"PCBData/{label}"])
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, path in enumerate(pool.map(fetch, files), 1):
            if index % 100 == 0:
                print(f"Downloaded {index}/{len(files)} files", flush=True)
    manifest = {"source": BASE, "full": args.full, "selection": "sha256(row) ascending; no label-based selection", "rows": selections}
    (root / "selection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({s: len(rows) for s, rows in selections.items()}, flush=True)


if __name__ == "__main__":
    main()
