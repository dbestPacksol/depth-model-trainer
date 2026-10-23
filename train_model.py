"""Train a YOLO26 depth estimator using paired local RGB and depth images."""

import argparse
import math
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """Keep the settings someone might change in one easy-to-find place."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dataset root; see README.md for its layout.")
    parser.add_argument("--model", default="yolo26s-depth.pt", help="Pretrained .pt weights or a model .yaml.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size in pixels.")
    parser.add_argument("--batch", type=int, default=8, help="Images per training batch; lower if GPU memory runs out.")
    parser.add_argument("--device", default=None, help="For example: 0 for the first GPU, or cpu. Default: automatic.")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers; 0 is a simple Windows-safe default.")
    parser.add_argument("--lr0", type=float, default=0.0001, help="Initial AdamW learning rate.")
    parser.add_argument("--depth-scale", type=float, help="PNG units/meter; default: saved dataset value, or 1000.")
    parser.add_argument("--max-depth", type=float, help="Validation limit in meters; default: saved value, or 100.")
    parser.add_argument("--project", type=Path, default=Path("runs/depth"), help="Training output directory.")
    parser.add_argument("--name", default="train", help="Run name; existing runs get a new numbered directory.")
    parser.add_argument("--check-data", action="store_true", help="Validate the dataset and write YAML without training.")
    args = parser.parse_args()

    for setting in ("epochs", "imgsz", "batch", "lr0", "depth_scale", "max_depth"):
        value = getattr(args, setting)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{setting.replace('_', '-')} must be a positive, finite number.")
    if args.workers < 0:
        parser.error("--workers must be zero or greater.")
    if Path(args.model).suffix.lower() not in {".pt", ".yaml", ".yml"}:
        parser.error("--model must be .pt weights or a .yaml model definition, not an exported inference model.")
    return args


def main() -> None:
    args = parse_arguments()

    # Check every pair before downloading weights or starting a training run.
    from helper.dataset import prepare_dataset

    dataset_yaml = prepare_dataset(args.data, args.depth_scale, args.max_depth)
    print(f"Dataset configuration: {dataset_yaml}")
    if args.check_data:
        return

    from ultralytics import YOLO

    # A .pt file fine-tunes existing weights; a .yaml file starts with random weights.
    model = YOLO(args.model)
    if model.task != "depth":
        raise ValueError(f"Expected a depth model, but {args.model!r} has task {model.task!r}.")

    # RGB images are the inputs. Paired depth PNGs provide the training targets.
    # A gentle learning rate and warmup help preserve pretrained depth knowledge.
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
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


if __name__ == "__main__":
    # Required for Windows when data loader workers start child processes.
    main()
