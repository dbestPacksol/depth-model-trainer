"""Sync reviewed RGB/depth pairs and archive/restore trained results using .env."""

import argparse

from helper.cloud_config import cloud_settings
from helper.cloud_storage import download_run, sync_dataset, upload_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset", help="Transfer only reviewed training pairs, never raw snapshots.")
    dataset.add_argument("direction", choices=("push", "pull", "sync"))
    dataset.add_argument("--data", default="datasets/curated")
    dataset.add_argument("--dataset", default="curated", help="Cloud dataset name.")
    runs = commands.add_parser("runs")
    runs.add_argument("direction", choices=("push", "pull", "list"))
    runs.add_argument("--path", help="Local training run directory (push) or restore destination (pull).")
    runs.add_argument("--run-id", help="Required for pull; optional unique ID for push.")
    args = parser.parse_args()
    if args.command == "runs" and args.direction != "list" and not args.path:
        parser.error("runs push/pull requires --path")
    if args.command == "runs" and args.direction == "pull" and not args.run_id:
        parser.error("runs pull requires --run-id")
    client, bucket = cloud_settings()
    if args.command == "dataset":
        result = sync_dataset(client, bucket, args.data, args.dataset, args.direction)
        print(f"Dataset {args.dataset}: {len(result['files']) // 2} pairs; {args.direction} complete.")
    elif args.direction == "push":
        run_id = upload_run(client, bucket, args.path, args.run_id)
        print(f"Archived run: s3://{bucket}/runs/{run_id}/")
    elif args.direction == "pull":
        download_run(client, bucket, args.path, args.run_id)
        print(f"Restored {args.run_id} to {args.path}")
    else:
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="runs/"):
            for item in page.get("Contents", []):
                if item["Key"].endswith("/manifest.json"):
                    print(item["Key"].split("/")[1])


if __name__ == "__main__":
    main()
