#!/usr/bin/env python3
"""Validate PEARL validation data without importing or running the model."""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


SPECS = {
    "voc20": ("VOCdevkit/VOC2012", "JPEGImages", "SegmentationClass", "ImageSets/Segmentation/val.txt", 1449, ".jpg", ".png", 20),
    "voc21": ("VOCdevkit/VOC2012", "JPEGImages", "SegmentationClass", "ImageSets/Segmentation/val.txt", 1449, ".jpg", ".png", 20),
    "context59": ("VOCdevkit/VOC2010", "JPEGImages", "SegmentationClassContext", "ImageSets/SegmentationContext/val.txt", 5105, ".jpg", ".png", 59),
    "context60": ("VOCdevkit/VOC2010", "JPEGImages", "SegmentationClassContext", "ImageSets/SegmentationContext/val.txt", 5105, ".jpg", ".png", 59),
    "ade20k": ("ADEChallengeData2016", "images/validation", "annotations/validation", None, 2000, ".jpg", ".png", 150),
    "coco_object": ("coco_stuff164k", "images/val2017", "annotations/val2017", None, 5000, ".jpg", "_instanceTrainIds.png", 80),
    "coco_stuff164k": ("coco_stuff164k", "images/val2017", "annotations/val2017", None, 5000, ".jpg", "_labelTrainIds.png", 170),
}


def verify(name, root, sample):
    relative, image_dir, mask_dir, split, expected, image_ext, mask_ext, max_id = SPECS[name]
    base = root / relative
    images, masks = base / image_dir, base / mask_dir
    errors = []
    if split:
        split_path = base / split
        if not split_path.is_file():
            errors.append(f"missing split: {split_path}")
            keys = []
        else:
            keys = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
            if len(set(keys)) != len(keys):
                errors.append("duplicate split entries")
    else:
        keys = sorted(p.stem for p in images.glob(f"*{image_ext}")) if images.is_dir() else []
    count_images = len(list(images.glob(f"*{image_ext}"))) if images.is_dir() else 0
    count_masks = len(list(masks.glob(f"*{mask_ext}"))) if masks.is_dir() else 0
    if len(keys) != expected:
        errors.append(f"validation entries {len(keys)} != {expected}")
    if not images.is_dir() or not masks.is_dir():
        errors.append("image or mask directory missing")
    missing_images = []
    missing_masks = []
    empty = []
    for key in keys:
        image = images / f"{key}{image_ext}"
        mask = masks / f"{key}{mask_ext}"
        if not image.is_file():
            missing_images.append(key)
        elif image.stat().st_size == 0:
            empty.append(str(image))
        if not mask.is_file():
            missing_masks.append(key)
        elif mask.stat().st_size == 0:
            empty.append(str(mask))
    if missing_images:
        errors.append(f"missing images: {len(missing_images)}; sample {missing_images[:3]}")
    if missing_masks:
        errors.append(f"missing masks: {len(missing_masks)}; sample {missing_masks[:3]}")
    if empty:
        errors.append(f"empty files: {len(empty)}; sample {empty[:3]}")
    sampled_ids = set()
    eligible = [key for key in keys if key not in set(missing_images + missing_masks)]
    if sample > 0 and eligible:
        stride = max(1, len(eligible) // sample)
        for key in eligible[::stride][:sample]:
            try:
                with Image.open(images / f"{key}{image_ext}") as im, Image.open(masks / f"{key}{mask_ext}") as mask:
                    im.load()
                    mask.load()
                    if im.size != mask.size:
                        errors.append(f"size mismatch: {key}: {im.size} vs {mask.size}")
                    if mask.mode not in ("P", "L", "I"):
                        errors.append(f"mask mode {mask.mode}: {key}")
                    ids = set(mask.getdata())
                    sampled_ids.update(ids)
                    bad = ids - set(range(max_id + 1)) - {255}
                    if bad:
                        errors.append(f"invalid raw label IDs {sorted(bad)[:8]}: {key}")
            except (OSError, ValueError) as exc:
                errors.append(f"unreadable pair {key}: {exc}")
    return {
        "dataset": name,
        "status": "COMPLETED" if not errors else "BLOCKED",
        "path": str(base),
        "validation_entries": len(keys),
        "images_in_directory": count_images,
        "masks_in_directory": count_masks,
        "sampled_raw_ids": sorted(sampled_ids),
        "errors": errors[:20],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*SPECS, "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=Path("/data/Multimodal-final-project"))
    parser.add_argument("--dataset-dir", type=Path, help="validate a staged dataset directory")
    parser.add_argument("--sample", type=int, default=16)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    names = list(SPECS) if args.dataset == "all" else [args.dataset]
    output = []
    for name in names:
        relative = SPECS[name][0]
        if args.dataset_dir:
            # Rebase the staged directory to the same shape as DATA_ROOT/datasets.
            root = args.dataset_dir
            spec = SPECS[name]
            SPECS[name] = (".", *spec[1:])
        else:
            root = args.data_root / "datasets"
        output.append(verify(name, root, args.sample))
        if args.dataset_dir:
            SPECS[name] = (relative, *spec[1:])
    if not args.quiet:
        print(json.dumps(output, indent=2))
    return 0 if all(item["status"] == "COMPLETED" for item in output) else 1


if __name__ == "__main__":
    sys.exit(main())
