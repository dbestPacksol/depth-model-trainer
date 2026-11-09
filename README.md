# depth-model-trainer

Train or fine-tune an Ultralytics YOLO26 depth estimator with a reviewed dataset
shared through this repository's dedicated S3 bucket and cached locally.
The model learns to predict depth from **RGB alone**; depth PNGs are the target
values used during training and validation.

## Install

Activate your Python environment, then install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

For NVIDIA GPU training, install a CUDA-enabled PyTorch build appropriate for
your system using the [PyTorch installer](https://pytorch.org/get-started/locally/).
CPU training is also available with `--device cpu`.

## Prepared collector dataset

The dataset imported on 2026-11-09 from
`/home/dale/Documents/Repos/Packsol/depth-frame-collector/dataset` is cached at
`datasets/curated` and published as the S3 dataset `curated`. It now contains 5,000
original RGB/depth pairs: 4,394 training and 606 validation, at 1280 × 720.
The expansion added 2,000 pairs and preserved all 3,000 original pairs and splits.
Depth is uint16 millimeters (`depth_scale: 1000`); zero pixels remain invalid.
Validation holds out Alina, Park 12 The Collection, Valdok, and Evoq East Cobb
entirely. Preserve these splits when adding data.

The local copy retains the collector's selection manifest, capture metadata,
summaries, and review sheets. S3 dataset sync publishes only the image/depth pairs
and their checksums/settings; it does not back up those supporting records.
Source checksums were checked for all 10,000 RGB/depth files during import.
All unique S3 objects were read back and SHA-256 verified after publication.
Local evidence is saved in
`datasets/curated/s3_verification.json` and `datasets/curated/verification.json`.
The installed Ultralytics 8.4.208 depth loader verified both splits and loaded
sample batches with the expected depth conversion.
The collector's copied `data.yaml` is regenerated with this checkout's path;
`train_model.py` uses `depth_dataset.generated.yaml`.

```bash
# On another machine, restore the same training inputs first.
python sync_s3.py dataset pull --data datasets/curated
python train_model.py --data datasets/curated --check-data

# Start fine-tuning on a machine with a working CUDA installation.
python train_model.py --data datasets/curated --model yolo26s-depth.pt --sync-s3 --upload-results --device 0
```

Dataset preparation does not start training. Pretrained weights download on the
first training invocation if absent locally.

## Configure the repository S3 bucket

The repo's **fresh, dedicated bucket** is the shared cloud source for reviewed
training pairs and archived training results. Set these values in the local
`.env` (see `.env.example`):

```dotenv
ACCESS_KEY=your-access-key
SECRET_KEY=your-secret-key
BUCKET_NAME=your-repository-bucket
```

Commands load `.env` from the repository root, even when invoked from another
working directory. Existing environment variables take precedence. Credentials
are never stored in manifests. `.env` is ignored by Git. Configure the bucket's
region through `AWS_DEFAULT_REGION` if needed. The bucket must already exist;
the scripts do not create buckets or alter bucket policies.

The credentials need `s3:ListBucket` on the bucket and `s3:GetObject`,
`s3:PutObject`, and `s3:AbortMultipartUpload` on `datasets/*` and `runs/*`.
Delete permissions are not required. Keep the bucket private.

```text
s3://BUCKET_NAME/
  datasets/
    curated/                       # Change with --dataset <name>
      manifest.json                # Reviewed paths, depth settings, SHA-256 and sizes
      objects/<sha256>              # Exact RGB JPG/depth PNG bytes, deduplicated by hash
  runs/
    <unique-run-id>/
      manifest.json                # Original output paths, hashes, training provenance
      objects/<sha256>              # Weights, metrics, plots, settings and other outputs
```

Manifests map each original relative path to its object hash. Dataset paths use
`images/{train,val}/...jpg` and `depth/{train,val}/...png`; run paths include
`weights/best.pt`, `weights/last.pt`, `results.csv`, plots and settings when
present. Download commands reconstruct those directories. Generated dataset YAML
uses machine-specific absolute paths and is rebuilt locally rather than uploaded.
Raw snapshots, review reports, caches, and files outside the paired data folders
are excluded from dataset uploads. Run archives exclude hidden files/directories.

## Review frames, then upload selected pairs

`download_snapshots.py` is an **optional acquisition helper**. It reads snapshots
from the camera pipeline's source bucket (`computer-vision-sessions` by default),
converts RGB PNGs to quality-95 JPGs, preserves depth PNG bytes, and creates a
local candidate dataset. This source bucket is separate from `BUCKET_NAME`.
Downloading snapshots does **not** approve them or publish them to the repo bucket.

```powershell
python download_snapshots.py --bucket_name computer-vision-sessions --prefix "device-name/" --output datasets/snapshot-review
```

The helper uses `ACCESS_KEY` and `SECRET_KEY` from `.env` by default. If source
credentials differ, supply `--access_key` and `--secret_key` for this helper only.
Source credentials need ListBucket and GetObject. It scans every listing page
under the prefix; without `--prefix` it scans the entire source bucket.
`--list-only` shows counts without downloading. It recognizes
`<device>/snapshots/snapshot_<Unix timestamp>/<camera>/rgb.png`, `depth.png`, and
optional `frame.json`, ignoring session/event videos. Camera depth defaults to
millimeters (`--depth-scale 1000`); override for other encodings.

The staging directory contains `raw/`, `images/{train,val}/`,
`depth/{train,val}/`, `import_report.jsonl`, `import_settings.json`, and a generated
dataset YAML. Incomplete pairs and failures are reported. Rerunning resumes raw
downloads and regenerates candidate images, preserving capture split assignments.
All cameras from a capture stay together; use `--split-by day` to group by device
and UTC day. At least two complete groups are required. Use `--val-fraction` and
`--seed` to configure the initial split.

**Review the candidates**, then copy only selected matching RGB/depth files into
`datasets/curated/images/{train,val}/` and `datasets/curated/depth/{train,val}/`,
keeping the same relative names and capture/session split boundaries. Do not sync
the staging directory unless you intend to approve every valid pair in it.
For non-default depth encoding, validate the curated directory with
`--depth-scale` and `--max-depth` before its first upload; those settings are then
shared in the cloud manifest.

```powershell
python train_model.py --data datasets/curated --check-data
python sync_s3.py dataset push --data datasets/curated
```

`push` uploads the validated local train/validation pairs into the named cloud
dataset. Both splits must contain usable pairs. Use unique sample names across
captures/devices; split and relative name identify each pair. Additional reviewed
pairs can be added locally and uploaded with the same command.

## Sync the local training dataset

```powershell
# Restore the shared dataset on another machine (or download new cloud pairs).
python sync_s3.py dataset pull --data datasets/curated

# Download cloud additions, then upload local additions.
python sync_s3.py dataset sync --data datasets/curated

# Keep independent dataset versions or depth conventions in separate namespaces.
python sync_s3.py dataset sync --data datasets/experiment --dataset experiment
```

Sync is additive: missing files are transferred, matching files are reused,
and neither local nor remote files are deleted. Different content at the same
relative path, or incompatible depth settings, causes an error. Move a conflicting
local copy aside to accept the cloud copy; use a new sample name or a new dataset
name for intentionally revised data. A pull retains additional local pairs.
Removing a local file does not remove its cloud counterpart; a later pull restores
it. For a revised selection that excludes published pairs, create a new dataset.
An empty bucket can be initialized with `push` or `sync` from a valid local dataset;
`pull` reports that no dataset exists yet.

Transfers verify SHA-256 and sizes. Downloads stage and validate the complete cloud
dataset before installing new pairs. Uploads publish `manifest.json` only after
all files finish, so interrupted uploads cannot expose half a pair. Conditional
manifest writes detect concurrent updates; rerun after that error. Interrupted
uploads may leave unreferenced objects, which are harmless and are not synced.
The commands do not remove historical or unreferenced objects automatically.
Uploads use up to eight concurrent transfers, deduplicate identical pending
content, and print object counts as they complete. The manifest is published only
after all transfers succeed. Retrying an interrupted initial push may re-upload
unreferenced objects from that attempt.

Train with automatic dataset sync and result archiving:

```powershell
python train_model.py --data datasets/curated --sync-s3 --upload-results --device 0
```

`--sync-s3` uses the same additive review policy: every valid local pair in `--data`
is considered selected for sharing. Omit the cloud flags for entirely local
training. `--check-data` never starts training or uploads results, but an explicit
`--sync-s3` still transfers the dataset. `--dataset` chooses the shared dataset.
On the initial upload, training's `--depth-scale` and `--max-depth` set the shared
depth convention; later invocations cannot override it with a different value.

## Archive and restore trained results

Each archive has a unique run ID and preserves the whole output directory,
including checkpoints, metrics, plots, and run settings. Existing run IDs cannot
be overwritten; an updated run must receive a new ID. With both training cloud
flags enabled, the archive also records the dataset manifest and its fingerprint
so the exact cloud training inputs are traceable. Without `--sync-s3`, dataset
provenance is not inferred. An upload failure leaves local results intact; retry
using the standalone archive command:

```powershell
python sync_s3.py runs push --path runs/depth/train --run-id experiment-001
python sync_s3.py runs list
python sync_s3.py runs pull --run-id experiment-001 --path runs/restored/experiment-001
python train_model.py --data datasets/curated --model runs/restored/experiment-001/weights/best.pt --device 0
```

Omit `--run-id` on push to generate an ID from the directory name, UTC time, and
random suffix. Only completed run manifests appear in `runs list`. Restores verify
checksums and refuse to overwrite different local files. These are explicit
archives after successful training; they do not stream checkpoints during a run.

## Dataset directory pattern

Pass the dataset root to `--data`. Both `train` and `val` must contain at least
one RGB/depth pair:

```text
my_dataset/
├── images/
│   ├── train/
│   │   ├── frame_0001.jpg
│   │   └── session_a/              # Optional nested folders
│   │       └── frame_0002.jpg
│   └── val/
│       └── frame_0100.jpg
└── depth/
    ├── train/
    │   ├── frame_0001.png
    │   └── session_a/
    │       └── frame_0002.png
    └── val/
        └── frame_0100.png
```

- RGB files must be RGB `.jpg` images. Depth files must be **single-channel,
  unsigned 16-bit `.png` images**, containing actual distances rather than colored
  depth previews or 8-bit grayscale pictures.
- Match each pair's filename stem and relative subfolder exactly. For example,
  `images/train/session_a/frame_0002.jpg` pairs with
  `depth/train/session_a/frame_0002.png`. Use lowercase `.png` extensions.
  Nested folders cannot be named `images`, because that name controls pairing.
- Depth must be registered to the RGB camera's view: each depth pixel must describe
  the corresponding RGB location. Matching dimensions are simplest; smaller depth
  maps with the same aspect ratio are supported and resized by Ultralytics.
  Bake any EXIF rotation into both images before training.
- For a new manually prepared dataset, the default depth convention is
  **millimeters**: `1500` means `1.5` meters. Imported datasets use their saved scale.
  `0` means missing/invalid depth and is ignored. Each depth image must contain
  at least one valid pixel. For another encoding, use `--depth-scale`, where
  `depth_in_meters = png_value / depth_scale`.
- Split data into training and held-out validation sets yourself. Keep related
  recording sessions together so nearly identical frames do not leak between
  splits. `train_model.py` does not split or move files; the snapshot helper creates
  candidate splits that you review before syncing.
- No bounding-box labels or `labels/` folder are needed.

## Validate the data

```powershell
python train_model.py --data "C:/datasets/my_dataset" --check-data
```

The script checks every pair for missing files, unreadable images, invalid depth
encoding, incompatible aspect ratios, and all-zero depth. It also rejects depth
PNGs without matching JPGs and unexpected files in the RGB folders.
It writes `depth_dataset.generated.yaml` inside the dataset root with absolute
dataset paths, the PNG scale, and the validation depth limit. This generated file
is replaced on each invocation; existing scale/limit values are retained unless
overridden. The dataset root must be writable.
`--check-data` does not load or download a model.

## Fine-tune pretrained weights

```powershell
python train_model.py --data "C:/datasets/my_dataset" --device 0
```

Defaults: `yolo26s-depth.pt`, 20 epochs, image size 640, batch size 8,
AdamW with an initial learning rate of `0.0001` and gentle warmup, and four data
loader workers per GPU on Linux (zero on Windows). These learning-rate settings follow the
[Ultralytics fine-tuning guidance](https://docs.ultralytics.com/tasks/depth/#fine-tuning-on-your-own-data).
The pretrained model downloads on first use if it is not already available.

To use another YOLO26 size or your own previously trained checkpoint:

```powershell
python train_model.py --data "C:/datasets/my_dataset" --model "C:/models/best.pt" --epochs 40 --batch 4 --device 0
```

This starts a new fine-tuning run from those weights. Reduce `--batch` if GPU
memory is insufficient. Omit `--device` to let Ultralytics select a device.
For depth PNGs using a different scale, add e.g. `--depth-scale 256`.
Use `--max-depth 10` to set the upper validation depth limit to 10 meters;
this does not cap the model's predictions.

### Performance on the dual RTX 3090 workstation

Use explicit GPU selection and enough loading workers to keep the GPUs supplied:

```bash
python train_model.py --data datasets/curated --device 0,1 --batch 32 --workers 4 --cache ram --upload-results
```

`--batch` is the total across GPUs: 32 supplies 16 samples to each GPU. The batch
must be divisible by the GPU count. Automatic device selection does not request
both GPUs. The 128-training/32-validation-pair smoke test completed on both 3090s
with these batch/worker/cache settings, using approximately 4.8 GB per GPU.
Increase the batch in steps after checking throughput and memory use; a larger
batch alone does not guarantee faster training.

The original zero-worker setting decodes, augments, and assembles samples in the
training process. On this Ryzen 9 5900X workstation, a warm data-only benchmark
using the real depth dataset at 640 pixels and batch 8 supplied approximately
80 images/sec with zero workers, 271 with four, and 449 with eight. These isolated
loading rates are not full training rates. A separate single-3090 AMP
forward/loss/backward benchmark achieved approximately 133 images/sec at batch 8,
143 at batch 16, and 145 at batch 24. Each test used short warmed-up loops;
optimizer updates, DDP communication, validation, and startup add overhead.
A combined real-data forward/loss/backward test with deterministic operations
enabled measured 47.5 images/sec with zero workers and 79.6 with four (about 1.67×).
Eight workers measured 79.4 images/sec, so four was sufficient for that workload.
This short single-GPU comparison excludes optimizer updates and is not a forecast
of full two-GPU epoch time.

`--cache ram` caches decoded RGB images; depth PNGs are still decoded per sample.
It consumes host RAM separately from GPU memory. Disk image caching is not
exposed because it writes `.npy` files into the RGB directories, which the
reviewed-data validator currently rejects. Four workers per GPU means eight
training workers across both cards; `--workers 8` uses sixteen. Ultralytics can
also create additional validation workers.

AMP is already enabled by Ultralytics. `--no-deterministic` permits
nondeterministic kernels when reproducibility is less important; it is optional,
and was also used for the smoke test. `--no-plots` skips diagnostic plots,
including the startup depth-distribution plot, but keeps validation enabled.
Neither option changes the fine-tuning learning rate.

`--sync-s3` runs before training and includes hashing, staging, and full data
validation. It can cause a long startup with idle GPUs, but does not run during
each training batch. When the dataset is already current, omit it as above;
`--upload-results` still archives the completed run. Preparation time and loader
settings are printed separately. For archived dataset-manifest provenance,
retain `--sync-s3` and accept its preparation cost.

## Train from scratch

Use a model architecture YAML instead of a pretrained checkpoint. For example:

```powershell
python train_model.py --data "C:/datasets/my_dataset" --model yolo26n-depth.yaml --epochs 100 --lr0 0.001 --device 0
```

This initializes random weights and still uses AdamW. Training from scratch
usually requires more data, more epochs, and learning-rate tuning for your dataset.
Prefer pretrained weights for a small custom dataset.

## Results and script layout

Runs are saved under `runs/depth/train/` by default. Existing runs receive a new
numbered folder. Use `--project` and `--name` to choose another location/name.
The script prints the actual paths to `weights/best.pt` and `weights/last.pt`.
Ultralytics also saves run settings, metrics, and plots in that run directory.

`train_model.py` contains the arguments and training call. `helper/dataset.py`
contains the dataset checks and YAML generation. Run `python train_model.py --help`
to see every setting. The data layout follows the
[Ultralytics depth dataset format](https://docs.ultralytics.com/datasets/depth/).

`sync_s3.py` manages reviewed cloud data and run archives through
`helper/cloud_storage.py`; `helper/cloud_config.py` loads repository credentials.
Run `python sync_s3.py --help` for cloud commands.

`download_snapshots.py` coordinates the optional snapshot staging import. Its helpers are
`helper/snapshot_s3.py` for listing, credentials, and downloads, and
`helper/snapshot_dataset.py` for splitting, conversion, and the import report.
Run `python download_snapshots.py --help` for importer settings. To run the tests:

```powershell
python -m unittest discover -s tests -v
```
