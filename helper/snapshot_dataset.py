"""Build training pairs while keeping raw snapshot downloads and source provenance."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from helper.dataset import prepare_dataset, validate_pair
from helper.snapshot_s3 import download_object, write_json


def sample_id(sample: dict, bucket: str) -> str:
    """Never use remote camera names as local filesystem paths."""
    digest = hashlib.sha256(f"{bucket}/{sample['camera_prefix']}".encode()).hexdigest()[:24]
    return f"{sample['snapshot']}_{digest}"


def split_group(sample: dict, split_by: str) -> str:
    if split_by == "snapshot":
        return sample["capture"]
    timestamp = int(sample["snapshot"].removeprefix("snapshot_"))
    day = datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()
    device_prefix = sample["capture"].rsplit("/snapshots/", 1)[0] if "/snapshots/" in sample["capture"] else ""
    return f"{device_prefix}/day/{day}"


def assign_splits(groups: set[str], previous: dict, fraction: float, seed: int) -> dict:
    """Keep old assignments fixed so a resumed import cannot leak across splits."""
    assignments = dict(previous)

    def score(group):
        return int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), 16)

    new_groups = sorted(groups - assignments.keys(), key=score)
    if not assignments:
        validation_count = max(1, min(len(new_groups) - 1, round(len(new_groups) * fraction)))
        if len(new_groups) < 2:
            validation_count = 0
        assignments.update({group: "val" if index < validation_count else "train"
                            for index, group in enumerate(new_groups)})
    else:
        for group in new_groups:
            assignments[group] = "val" if score(group) / (2 ** 256) < fraction else "train"
        # A first import with only one group can gain its missing split later.
        if len(groups) >= 2 and new_groups:
            active_splits = {assignments[group] for group in groups}
            if len(active_splits) == 1:
                assignments[new_groups[0]] = "val" if "train" in active_splits else "train"
    return assignments


def open_import(output: Path, settings: dict, groups: set[str]) -> dict:
    """Require an owned output directory and compatible settings before resuming."""
    settings_path = output / "import_settings.json"
    previous_groups = {}
    if settings_path.exists():
        previous = json.loads(settings_path.read_text(encoding="utf-8"))
        if previous["settings"] != settings:
            raise ValueError("Import settings changed. Use a different --output directory to build a new dataset.")
        previous_groups = previous["groups"]
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty and has no import_settings.json. Choose a new --output directory.")
    assignments = assign_splits(groups, previous_groups, settings["val_fraction"], settings["seed"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(settings_path, {"settings": settings, "groups": assignments})
    for category in ("images", "depth"):
        for split in ("train", "val"):
            (output / category / split).mkdir(parents=True, exist_ok=True)
    return assignments


def create_training_pair(raw_directory: Path, output: Path, split: str, identifier: str) -> tuple[Path, Path]:
    """Convert RGB to JPEG while copying the original 16-bit depth PNG unchanged."""
    image_path = output / "images" / split / f"{identifier}.jpg"
    depth_path = output / "depth" / split / f"{identifier}.png"
    temporary_image = image_path.with_suffix(".jpg.part")
    temporary_depth = depth_path.with_suffix(".png.part")
    try:
        with Image.open(raw_directory / "rgb.png") as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError("Snapshot rgb.png must be an RGB PNG.")
            image.save(temporary_image, format="JPEG", quality=95, subsampling=0)
        validate_pair(temporary_image, raw_directory / "depth.png")
        shutil.copyfile(raw_directory / "depth.png", temporary_depth)
        temporary_image.replace(image_path)
        temporary_depth.replace(depth_path)
    finally:
        temporary_image.unlink(missing_ok=True)
        temporary_depth.unlink(missing_ok=True)
    return image_path, depth_path


def build_dataset(client, samples: list[dict], output: Path, settings: dict) -> dict:
    """Download all discovered images and report incomplete/invalid samples separately."""
    from botocore.exceptions import BotoCoreError, ClientError

    output = output.expanduser().resolve()
    complete_samples = [sample for sample in samples if {"rgb.png", "depth.png"} <= sample["objects"].keys()]
    groups = {split_group(sample, settings["split_by"]) for sample in complete_samples}
    assignments = open_import(output, settings, groups)
    summary = {"imported": 0, "incomplete": 0, "failed": 0}
    report_path = output / "import_report.jsonl"
    # Flush each result so an interrupted run still leaves a useful report.
    with report_path.open("w", encoding="utf-8") as report:
        for index, sample in enumerate(samples, start=1):
            identifier = sample_id(sample, settings["bucket"])
            raw_directory = output / "raw" / identifier
            row = {"sample_id": identifier, "capture": sample["capture"], "camera": sample["camera"],
                   "source": sample["objects"]}
            try:
                for name, source in sample["objects"].items():
                    download_object(client, settings["bucket"], source, raw_directory / name)
                missing = {"rgb.png", "depth.png"} - sample["objects"].keys()
                if missing:
                    row.update(status="incomplete", reason=f"Missing S3 objects: {', '.join(sorted(missing))}")
                    summary["incomplete"] += 1
                else:
                    group = split_group(sample, settings["split_by"])
                    split = assignments[group]
                    image_path, depth_path = create_training_pair(raw_directory, output, split, identifier)
                    row.update(status="imported", split=split, split_group=group,
                               image=image_path.relative_to(output).as_posix(),
                               depth=depth_path.relative_to(output).as_posix())
                    summary["imported"] += 1
            except (BotoCoreError, ClientError, OSError, ValueError) as error:
                row.update(status="failed", reason=str(error))
                summary["failed"] += 1
                print(f"Failed {sample['camera_prefix']}: {error}", flush=True)
            report.write(json.dumps(row, ensure_ascii=False) + "\n")
            report.flush()
            if index % 25 == 0 or index == len(samples):
                print(f"Processed {index}/{len(samples)} camera samples: {summary}", flush=True)

    if not summary["failed"]:
        if len(groups) < 2:
            raise ValueError(
                "Raw images were downloaded, but at least two complete split groups are needed. "
                "Collect more captures/days or use --split-by snapshot in a new output directory. "
                "See import_report.jsonl."
            )
        # Also verifies that both splits have usable images before writing training YAML.
        prepare_dataset(output, settings["depth_scale"], settings["max_depth"])
    return summary
