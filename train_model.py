"""Train a YOLO26 depth estimator using paired local RGB and depth images."""

import argparse
import math
import os
from pathlib import Path
from time import perf_counter


def parse_arguments() -> argparse.Namespace:
    """Keep the settings someone might change in one easy-to-find place."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dataset root; see README.md for its layout.")
    parser.add_argument("--model", default="yolo26s-depth.pt", help="Pretrained .pt weights or a model .yaml.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size in pixels.")
    parser.add_argument("--batch", type=int, default=8, help="Total batch across GPUs; use a multiple of GPU count.")
    parser.add_argument("--device", default=None, help="For example: 0, 0,1 for both GPUs, or cpu. Default: automatic.")
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else min(4, os.cpu_count() or 1),
                        help="Data loader workers per GPU (default: 4 on Linux, 0 on Windows).")
    parser.add_argument("--cache", choices=("off", "ram"), default="off",
                        help="Cache decoded RGB in RAM; depth PNGs still decode per sample. Default: off.")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                        help="Deterministic operations; --no-deterministic permits faster nondeterministic kernels.")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True,
                        help="Generate diagnostic plots; --no-plots reduces plotting overhead.")
    parser.add_argument("--lr0", type=float, default=0.0001, help="Initial AdamW learning rate.")
    parser.add_argument("--depth-scale", type=float, help="PNG units/meter; default: saved dataset value, or 1000.")
    parser.add_argument("--max-depth", type=float, help="Validation limit in meters; default: saved value, or 100.")
    parser.add_argument("--project", type=Path, default=Path("runs/depth"), help="Training output directory.")
    parser.add_argument("--name", default="train", help="Run name; existing runs get a new numbered directory.")
    parser.add_argument("--check-data", action="store_true", help="Validate the dataset and write YAML without training.")
    parser.add_argument("--sync-s3", action="store_true", help="Sync reviewed pairs with S3 before training.")
    parser.add_argument("--upload-results", action="store_true", help="Archive the completed training run to S3.")
    parser.add_argument("--dataset", default="curated", help="Cloud dataset name for --sync-s3.")
    args = parser.parse_args()

    for setting in ("epochs", "imgsz", "batch", "lr0", "depth_scale", "max_depth"):
        value = getattr(args, setting)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{setting.replace('_', '-')} must be a positive, finite number.")
    if args.workers < 0:
        parser.error("--workers must be zero or greater.")
    devices = args.device.split(",") if args.device else []
    if len(devices) > 1 and all(device.strip().isdigit() for device in devices) and args.batch % len(devices):
        parser.error("--batch must be divisible by the number of GPUs in --device.")
    if Path(args.model).suffix.lower() not in {".pt", ".yaml", ".yml"}:
        parser.error("--model must be .pt weights or a .yaml model definition, not an exported inference model.")
    return args


def main() -> None:
    args = parse_arguments()
    preparation_start = perf_counter()

    cloud = None
    provenance = {}
    if args.sync_s3 or (args.upload_results and not args.check_data):
        from helper.cloud_config import cloud_settings

        cloud = cloud_settings()
    if args.sync_s3:
        import hashlib
        import json
        from helper.cloud_storage import sync_dataset

        manifest = sync_dataset(*cloud, args.data, args.dataset,
                                depth_scale=args.depth_scale, max_depth=args.max_depth)
        provenance = {"dataset": args.dataset, "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()).hexdigest(), "manifest": manifest}

    # Check every pair before downloading weights or starting a training run.
    from helper.dataset import prepare_dataset

    # sync_dataset already validates the local pairs and writes this YAML.
    dataset_yaml = (args.data.expanduser().resolve() / "depth_dataset.generated.yaml" if args.sync_s3
                    else prepare_dataset(args.data, args.depth_scale, args.max_depth))
    print(f"Dataset configuration: {dataset_yaml}")
    print(f"Dataset preparation (including S3 when enabled): {perf_counter() - preparation_start:.1f}s")
    if args.check_data:
        return

    from ultralytics import YOLO

    # A .pt file fine-tunes existing weights; a .yaml file starts with random weights.
    model = YOLO(args.model)
    if model.task != "depth":
        raise ValueError(f"Expected a depth model, but {args.model!r} has task {model.task!r}.")

    # RGB images are the inputs. Paired depth PNGs provide the training targets.
    # A gentle learning rate and warmup help preserve pretrained depth knowledge.
    print(f"Training: device={args.device or 'automatic'}, total batch={args.batch}, "
          f"workers per GPU={args.workers}, RGB cache={args.cache}, deterministic={args.deterministic}")
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache="ram" if args.cache == "ram" else False,
        deterministic=args.deterministic,
        plots=args.plots,
        optimizer="AdamW",
        lr0=args.lr0,
        warmup_bias_lr=args.lr0,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        seed=0,
    )

    # Read the actual run directory because Ultralytics may number the run name.
    run_directory = Path(model.trainer.save_dir)
    print(f"Best model: {run_directory / 'weights' / 'best.pt'}")
    print(f"Last model: {run_directory / 'weights' / 'last.pt'}")
    if args.upload_results:
        from helper.cloud_storage import upload_run

        run_id = upload_run(*cloud, run_directory, provenance=provenance)
        print(f"Archived run: s3://{cloud[1]}/runs/{run_id}/")


if __name__ == "__main__":
    # Required for Windows when data loader workers start child processes.
    main()
