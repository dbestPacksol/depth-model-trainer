"""Validate RGB/depth pairs and describe them in Ultralytics dataset YAML."""

import math
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def validate_pair(image_path: Path, depth_path: Path) -> None:
    """Catch common data mistakes before a long training job starts."""
    if not depth_path.is_file():
        raise ValueError(f"Missing depth PNG for {image_path}: expected {depth_path}")

    with Image.open(image_path) as image:
        if image.format != "JPEG" or image.mode != "RGB":
            raise ValueError(f"Expected an RGB JPEG: {image_path}")
        image.load()  # Decode now to detect damaged files, rather than just reading the header.
        image_width, image_height = image.size
        if image.getexif().get(274, 1) != 1:
            raise ValueError(f"Bake EXIF orientation into the RGB/depth pair before training: {image_path}")

    with Image.open(depth_path) as depth_image:
        if depth_image.format != "PNG":
            raise ValueError(f"Expected a depth PNG: {depth_path}")
        depth = np.asarray(depth_image)

    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(f"Depth must be a single-channel, 16-bit unsigned PNG: {depth_path}")
    depth_height, depth_width = depth.shape
    if image_width * depth_height != image_height * depth_width:
        raise ValueError(f"RGB and depth must have the same aspect ratio: {image_path} and {depth_path}")
    if not depth.any():
        raise ValueError(f"Depth PNG has no valid pixels (all values are zero): {depth_path}")


def validate_split(dataset_root: Path, split: str) -> int:
    """Require one matching PNG for every JPG, including nested session folders."""
    image_directory = dataset_root / "images" / split
    depth_directory = dataset_root / "depth" / split
    if not image_directory.is_dir() or not depth_directory.is_dir():
        raise ValueError(f"Expected both {image_directory} and {depth_directory}")

    # Reject other image types so Ultralytics cannot silently train on unchecked files.
    files = sorted(path for path in image_directory.rglob("*") if path.is_file())
    images = [path for path in files if path.suffix.lower() == ".jpg"]
    other_images = [path for path in files if path.suffix.lower() not in {".jpg", ".cache"}]
    if other_images:
        raise ValueError(f"Only .jpg files belong in the RGB folders: {other_images[0]}")
    if not images:
        raise ValueError(f"No .jpg images found in {image_directory}")

    expected_depths = set()
    for image_path in images:
        relative_path = image_path.relative_to(image_directory)
        # Ultralytics swaps the LAST 'images' path component when locating depth.
        if "images" in relative_path.parts[:-1]:
            raise ValueError(f"Nested session folders cannot be named 'images': {image_path}")
        depth_path = depth_directory / relative_path.with_suffix(".png")
        validate_pair(image_path, depth_path)
        expected_depths.add(depth_path)

    extra_depths = sorted(path for path in depth_directory.rglob("*.png") if path not in expected_depths)
    if extra_depths:
        raise ValueError(f"Depth PNG has no matching RGB JPG: {extra_depths[0]}")
    return len(images)


def prepare_dataset(dataset_root: Path, depth_scale: float | None = None, max_depth: float | None = None) -> Path:
    """Write a generated YAML beside the dataset without copying or changing images."""
    dataset_root = dataset_root.expanduser().resolve()
    yaml_path = dataset_root / "depth_dataset.generated.yaml"
    saved_configuration = (yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}) if yaml_path.exists() else {}
    if not isinstance(saved_configuration, dict):
        raise ValueError(f"Expected a dataset YAML mapping: {yaml_path}")
    depth_scale = depth_scale if depth_scale is not None else saved_configuration.get("depth_scale", 1000)
    max_depth = max_depth if max_depth is not None else saved_configuration.get("max_depth", 100)
    for name, value in (("depth_scale", depth_scale), ("max_depth", max_depth)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Dataset {name} must be a positive, finite number.")
    for split in ("train", "val"):
        count = validate_split(dataset_root, split)
        print(f"Validated {split}: {count} RGB/depth pairs")

    configuration = {
        "path": dataset_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "depth"},
        "channels": 3,
        "depth_scale": depth_scale,
        "max_depth": max_depth,
    }
    yaml_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    return yaml_path
