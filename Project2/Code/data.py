"""Dataset loader with automatic checks and capture-session grouping.

Usage:
    python data.py                       # print the audit report and the train/test table
    python data.py --montage OUT_DIR     # also write a label-check grid per class
"""
import argparse
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

IMAGE_DIR = Path(__file__).resolve().parent.parent / "Data" / "data" / "images" / "images"
CLASSES = ["road", "stop", "speed", "blue_light", "yellow_light", "sheep"]
DEFAULT_SIZE = (64, 48)  # (width, height); the sign/sheep/light images are already this size
SESSION_GAP = timedelta(seconds=30)  # a larger gap between frames starts a new session
BLOCK_SECONDS = {"road": 2}  # block length per class, default below
DEFAULT_BLOCK_SECONDS = 5
TEST_EVERY = 5  # every 5th block of each class goes to the test set (~20%)

_TS = re.compile(r"(\d{8})_(\d{6})_(\d+)")


def parse_timestamp(path):
    """20261006_134623_888304.jpg -> datetime (the last field is a fraction of a second)."""
    m = _TS.search(Path(path).stem)
    if m is None:
        return None
    date, clock, frac = m.groups()
    return datetime.strptime(date + clock, "%Y%m%d%H%M%S") + timedelta(seconds=float("0." + frac))


def assign_sessions(paths):
    """Give each path a session id: consecutive frames closer than SESSION_GAP share one."""
    stamps = {p: parse_timestamp(p) for p in paths}
    ordered = sorted((p for p in paths if stamps[p] is not None), key=lambda p: stamps[p])
    sessions, current, prev = {}, 0, None
    for p in ordered:
        if prev is not None and stamps[p] - prev > SESSION_GAP:
            current += 1
        sessions[p] = current
        prev = stamps[p]
    for p in paths:  # files with no parsable timestamp each get their own session
        if p not in sessions:
            current += 1
            sessions[p] = current
    return sessions


def assign_blocks(paths, seconds):
    """Group consecutive frames into blocks of `seconds`; a block never spans a session gap.

    Frames in the same block are near-duplicates, so a block must go entirely into train or test.
    """
    stamps = {p: parse_timestamp(p) for p in paths}
    ordered = sorted((p for p in paths if stamps[p] is not None), key=lambda p: stamps[p])
    blocks, current, start, prev = {}, 0, None, None
    for p in ordered:
        t = stamps[p]
        if prev is not None and (t - prev > SESSION_GAP or (t - start).total_seconds() >= seconds):
            current += 1
            start = t
        if start is None:
            start = t
        blocks[p] = current
        prev = t
    for p in paths:  # files with no parsable timestamp each get their own block
        if p not in blocks:
            current += 1
            blocks[p] = current
    return blocks


def load_dataset(image_dir=IMAGE_DIR, size=DEFAULT_SIZE, exclude=(), verbose=True):
    """Read every image, skipping bad files.

    Returns (images, labels, groups, paths):
      images: list of BGR uint8 arrays resized to `size`=(w, h) (None keeps the original size)
      labels: class name per image (from the folder name)
      groups: time-block id per image (unique across classes), for GroupKFold / group splits
      paths:  file path per image
    `exclude` is a collection of filenames to drop (e.g. mislabelled images).
    """
    exclude = set(exclude)
    images, labels, groups, paths = [], [], [], []
    unreadable, orig_sizes = [], Counter()
    for cls in CLASSES:
        files = sorted(
            p for p in (Path(image_dir) / cls).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.name not in exclude
        )
        blocks = assign_blocks(files, BLOCK_SECONDS.get(cls, DEFAULT_BLOCK_SECONDS))
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                unreadable.append(p)
                continue
            orig_sizes[(cls, img.shape[:2])] += 1
            if size is not None:
                img = cv2.resize(img, size)
            images.append(img)
            labels.append(cls)
            groups.append(f"{cls}_{blocks[p]}")
            paths.append(p)
    if verbose:
        report(labels, groups, unreadable, orig_sizes)
    return images, labels, groups, paths


def split_blocks(labels, groups, test_every=TEST_EVERY):
    """Train/test split by block: within each class, every `test_every`-th block (in time order)
    is test. Spreading test blocks over the whole recording keeps both sets mixed (e.g. straight
    and curved road) and keeps near-duplicate frames on the same side. Returns index arrays."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    train, test = [], []
    for cls in CLASSES:
        idx = np.flatnonzero(labels == cls)
        # block ids are numbered in time order within a class ("road_12" -> 12)
        order = sorted({groups[i] for i in idx}, key=lambda g: int(g.rsplit("_", 1)[1]))
        test_blocks = {g for k, g in enumerate(order) if k % test_every == test_every // 2}
        for i in idx:
            (test if groups[i] in test_blocks else train).append(i)
    return np.array(train), np.array(test)


def report(labels, groups, unreadable, orig_sizes):
    train, test = split_blocks(labels, groups)
    labels_a, groups_a = np.asarray(labels), np.asarray(groups)
    print(f"\nImages loaded: {len(labels)}")
    print(f"{'class':<14}{'images':>8}{'blocks':>8}{'train':>8}{'test':>8}")
    for cls in CLASSES:
        m = labels_a == cls
        print(f"{cls:<14}{m.sum():>8}{len(set(groups_a[m])):>8}"
              f"{(labels_a[train] == cls).sum():>8}{(labels_a[test] == cls).sum():>8}")
    print(f"{'total':<14}{len(labels):>8}{len(set(groups)):>8}{len(train):>8}{len(test):>8}")
    print(f"\nUnreadable files: {len(unreadable)}")
    for p in unreadable:
        print("  ", p)
    print("Original image sizes (h, w):")
    for (cls, shape), n in sorted(orig_sizes.items()):
        print(f"   {cls:<14}{shape}: {n}")


def write_montages(out_dir, per_class=30, cols=6, thumb=(160, 120), seed=0):
    """Save one grid of random thumbnails per class so labels can be checked by eye."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for cls in CLASSES:
        files = sorted((IMAGE_DIR / cls).glob("*.jpg"))
        picks = rng.choice(len(files), size=min(per_class, len(files)), replace=False)
        tiles = []
        for i in sorted(picks):
            img = cv2.imread(str(files[i]))
            if img is None:
                continue
            tile = cv2.resize(img, thumb)
            cv2.putText(tile, files[i].stem[-10:], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            tiles.append(tile)
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
        cv2.imwrite(str(out_dir / f"{cls}.jpg"), np.vstack(rows))
    print(f"\nMontages written to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", metavar="OUT_DIR", help="write a label-check grid per class")
    args = ap.parse_args()
    load_dataset()
    if args.montage:
        write_montages(args.montage)
