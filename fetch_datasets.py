#!/usr/bin/env python3
"""Rebuild the demo's data/ directory from the original public datasets.

setup.sh calls this only when a dataset is missing from the place the demo
code looks for it, so a copy unpacked from the distribution tar never triggers
a download.

What the demo actually consumes is images only -- the dataset loops just
display frames and run inference, they never read labels. So this reproduces
the exact image slices the models were validated on, not the full YOLO
conversion the training sessions did:

  ch1_images  DeepPCB official test split (`PCBData/test.txt`), the `_test`
              image of each pair                                    -> 500 jpg
  ch3_images  SolDef_AI, the val 20% of a seed-0 shuffle over the labelled
              JSONs -- the same split the model was trained against  -> 86 jpg
  ch2_visa    VisA pcb1: every Anomaly image + the first 200 Normal images in
              sorted order, which is exactly the slice the CH2 loop indexes
                                                          -> 100 + 200 jpg

Originals already present under --cache are reused instead of re-downloaded.
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

DEEPPCB_GIT = "https://github.com/tangsanli5201/DeepPCB.git"          # MIT
VISA_TAR = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"  # 1.9 GB
SOLDEF_KAGGLE = "mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection"
SOLDEF_PAGE = "https://www.kaggle.com/datasets/" + SOLDEF_KAGGLE


def sh(*cmd: str) -> None:
    print("      $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def copy_all(pairs, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for src, name in pairs:
        shutil.copy2(src, dest / name)
    return len(pairs)


def fetch_deeppcb(cache: Path, dest: Path) -> None:
    root = cache / "deeppcb"
    if not (root / "PCBData").is_dir():
        print(f"      cloning {DEEPPCB_GIT}")
        sh("git", "clone", "--depth", "1", DEEPPCB_GIT, str(root))
    pcb = root / "PCBData"
    pairs = []
    for line in (pcb / "test.txt").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        img_rel = line.split()[0]
        stem = Path(img_rel).stem
        img = pcb / Path(img_rel).parent / f"{stem}_test.jpg"
        if img.exists():
            pairs.append((img, f"{stem}.jpg"))
    print(f"      {copy_all(pairs, dest)} images -> {dest}")


def fetch_soldef(cache: Path, dest: Path) -> None:
    labeled = cache / "soldef" / "SolDef_AI" / "Labeled"
    if not labeled.is_dir():
        # No credential-free mirror exists for this one; Kaggle is the
        # publisher's distribution channel and needs ~/.kaggle/kaggle.json.
        if shutil.which("kaggle"):
            print(f"      downloading {SOLDEF_KAGGLE} via the kaggle CLI")
            (cache / "soldef").mkdir(parents=True, exist_ok=True)
            sh("kaggle", "datasets", "download", "-d", SOLDEF_KAGGLE,
               "-p", str(cache / "soldef"), "--unzip")
        else:
            raise SystemExit(
                "ERROR: SolDef_AI is not available for anonymous download.\n"
                f"  Get it from {SOLDEF_PAGE}\n"
                f"  and unzip it so that this path exists:\n    {labeled}\n"
                "  (or `pip install kaggle` + place ~/.kaggle/kaggle.json, then re-run)")
    jsons = sorted(labeled.glob("*.json"))
    if not jsons:
        raise SystemExit(f"ERROR: no LabelMe JSONs under {labeled}")
    # Same shuffle the training session used -- seed 0, val = first 20%.
    random.Random(0).shuffle(jsons)
    n_val = max(1, round(len(jsons) * 0.2))
    pairs = [(j.with_suffix(".jpg"), j.with_suffix(".jpg").name) for j in jsons[:n_val]]
    pairs = [(s, n) for s, n in pairs if s.exists()]
    print(f"      {copy_all(pairs, dest)} images -> {dest}")


def fetch_visa(cache: Path, dest: Path) -> None:
    images = cache / "visa" / "pcb1" / "Data" / "Images"
    if not images.is_dir():
        tar_path = cache / "VisA_20220922.tar"
        if not tar_path.exists():
            print(f"      downloading {VISA_TAR} (1.9 GB)")
            cache.mkdir(parents=True, exist_ok=True)
            sh("curl", "-L", "--fail", "-o", str(tar_path), VISA_TAR)
        print("      extracting pcb1")
        with tarfile.open(tar_path) as tf:
            members = [m for m in tf.getmembers() if m.name.startswith("pcb1/Data/Images/")]
            tf.extractall(cache / "visa", members=members)
    anomaly = sorted((images / "Anomaly").glob("*"))
    normal = sorted((images / "Normal").glob("*"))[:200]
    n = copy_all([(p, p.name) for p in anomaly], dest / "Anomaly")
    n += copy_all([(p, p.name) for p in normal], dest / "Normal")
    print(f"      {n} images -> {dest}")


JOBS = {
    "ch1_images": fetch_deeppcb,
    "ch3_images": fetch_soldef,
    "ch2_visa": fetch_visa,
}


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(here / "data"), help="the demo's data/ directory")
    ap.add_argument("--cache", default=str(Path.home() / "m1" / "datasets"),
                    help="where the original datasets are kept / downloaded to")
    ap.add_argument("--only", choices=sorted(JOBS), action="append",
                    help="fetch just this one (repeatable); default is everything missing")
    args = ap.parse_args()

    dest_root, cache = Path(args.dest), Path(args.cache)
    wanted = args.only or sorted(JOBS)
    failed = []
    for name in wanted:
        dest = dest_root / name
        if dest.exists():
            print(f"[{name}] already present -- skipping")
            continue
        print(f"[{name}] fetching")
        try:
            JOBS[name](cache, dest)
        except (subprocess.CalledProcessError, SystemExit, OSError) as e:
            shutil.rmtree(dest, ignore_errors=True)   # never leave a half-slice behind
            print(f"[{name}] FAILED: {e}", file=sys.stderr)
            failed.append(name)
    if failed:
        print(f"\nincomplete: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nall datasets present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
