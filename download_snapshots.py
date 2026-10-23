"""Download camera-server snapshots from S3 and build a local depth dataset."""

import argparse
import math
from pathlib import Path

from helper.snapshot_dataset import build_dataset
from helper.snapshot_s3 import create_s3_client, find_snapshots


# Default S3 configuration. Each value can be overridden on the command line.
DEFAULT_ACCESS_KEY = ""
DEFAULT_SECRET_KEY = ""
DEFAULT_BUCKET_NAME = "computer-vision-sessions"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/snapshots"), help="Local dataset directory.")
    parser.add_argument("--access_key", default=DEFAULT_ACCESS_KEY, help="AWS access key ID.")
    parser.add_argument("--secret_key", default=DEFAULT_SECRET_KEY, help="AWS secret access key.")
    parser.add_argument("--bucket_name", default=DEFAULT_BUCKET_NAME, help="S3 bucket; default: computer-vision-sessions.")
    parser.add_argument("--prefix", default="", help="Optional device/S3 folder prefix. Default: scan the entire bucket.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of capture groups held out for validation.")
    parser.add_argument("--seed", type=int, default=0, help="Repeatable train/validation assignment.")
    parser.add_argument("--split-by", choices=("snapshot", "day"), default="snapshot", help="Group by capture or device/UTC day.")
    parser.add_argument("--depth-scale", type=float, default=1000, help="PNG units per meter; default: 1000 (Orbbec millimeters).")
    parser.add_argument("--max-depth", type=float, default=25, help="Upper validation depth limit in meters.")
    parser.add_argument("--list-only", action="store_true", help="List totals without downloading or creating local files.")
    args = parser.parse_args()
    for setting in ("access_key", "secret_key", "bucket_name"):
        if not getattr(args, setting).strip():
            parser.error(f"--{setting} must not be empty.")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1, excluding both endpoints.")
    for setting in ("depth_scale", "max_depth"):
        value = getattr(args, setting)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{setting.replace('_', '-')} must be positive and finite.")
    return args


def main() -> None:
    args = parse_arguments()
    client = create_s3_client(args.access_key, args.secret_key)
    bucket = args.bucket_name
    prefix = args.prefix.rstrip("/") + "/" if args.prefix else ""

    print(f"Scanning s3://{bucket}/{prefix} for snapshot RGB/depth pairs...", flush=True)
    snapshots = find_snapshots(client, bucket, prefix)
    complete = sum("rgb.png" in sample["objects"] and "depth.png" in sample["objects"] for sample in snapshots)
    print(f"Found {len(snapshots)} camera samples: {complete} complete, {len(snapshots) - complete} incomplete.")
    if args.list_only:
        return
    if not snapshots:
        raise SystemExit("No snapshot images found. Check the bucket, prefix, and uploader progress.")

    settings = {
        "bucket": bucket,
        "prefix": prefix,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "split_by": args.split_by,
        "depth_scale": args.depth_scale,
        "max_depth": args.max_depth,
    }
    summary = build_dataset(client, snapshots, args.output, settings)
    print(f"Dataset: {args.output.expanduser().resolve()}")
    print(f"Imported: {summary['imported']}; incomplete: {summary['incomplete']}; failed: {summary['failed']}")
    print("Review import_report.jsonl for source keys and any skipped/failed samples.")
    if summary["failed"]:
        raise SystemExit("Some samples failed. Fix the reported errors and rerun the same command to resume.")


if __name__ == "__main__":
    main()
