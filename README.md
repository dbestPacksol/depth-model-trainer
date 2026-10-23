# depth-model-trainer

Train or fine-tune an Ultralytics YOLO26 depth estimator with a local dataset.
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

## Import camera snapshots from S3

`download_snapshots.py` downloads the camera server's snapshots and builds the
directory pattern used by `train_model.py`.

The inspected Vision-Test pipeline is:

1. `POST /snapshot` in `vision-camera-server` writes one pair per physical camera
   under `data/snapshots/snapshot_<Unix timestamp>/<camera name or ID>/`:
   `rgb.png`, lossless 16-bit `depth.png`, and `frame.json`. The capture directory
   also contains `snapshot.json`.
2. Vision-Test's `docker-compose.yml` mounts the same `vision-camera-data` volume
   at `/app/vision-camera-server/data` in the camera server and `/data` in
   `aws_uploader`. The uploader watches `/data` and targets
   **`computer-vision-sessions`**.
3. `aws_directory_uploader` waits for each file to be unchanged across two scans
   and idle for more than 20 seconds, uploads it, then removes the local file.
   Its S3 key is `<storage prefix>/<path relative to /data>`. The prefix is
   `STORAGE_DIRECTORY` when set, otherwise `BALENA_DEVICE_NAME_AT_INIT`, otherwise
   `Test`. The inspected Compose file does not explicitly set the prefix.

For example, an uploaded image can be:

```text
s3://computer-vision-sessions/<device>/snapshots/snapshot_1792780000/front/rgb.png
s3://computer-vision-sessions/<device>/snapshots/snapshot_1792780000/front/depth.png
```

The Orbbec cameras report depth in millimeters, so `--depth-scale` defaults to
`1000`: a PNG value of `1500` represents `1.5` meters. The camera code copies raw
SDK uint16 values into the PNG without recording a scale in `frame.json`.
Override `--depth-scale` when importing data with another encoding.
The importer preserves the depth PNG bytes exactly
and records the chosen scale in the generated dataset YAML. Training reuses this
saved scale and depth limit unless you explicitly override them.

Supply the access key and secret key on the command line. The default bucket is
`computer-vision-sessions`. The script does not read Compose files or contain
default credentials.

```powershell
python download_snapshots.py --access_key "your-access-key" --secret_key "your-secret-key"
```

Override the bucket when needed:

```powershell
python download_snapshots.py --access_key "your-access-key" --secret_key "your-secret-key" --bucket_name "your-bucket"
```

The credentials need `s3:ListBucket` and `s3:GetObject`; uploader write permissions
alone may not include those. Credentials are not written into the dataset or
printed by the script. S3 access uses the SDK's default region handling.

By default, the importer scans **the entire bucket**, across all device prefixes,
using every listing page. Only the snapshot layout is imported; event/session
videos are ignored. Use `--prefix "device-name/"` to select one device, or add
`--list-only` to inspect sample counts without downloading or creating local files.

The output looks like:

```text
datasets/snapshots/
├── raw/<snapshot timestamp + source hash>/
│   ├── rgb.png                     # Original downloaded PNG
│   ├── depth.png                   # Original downloaded PNG
│   ├── frame.json                  # Downloaded when present in the listing
│   └── *.s3.json                   # Source fingerprints and local SHA-256 hashes
├── images/{train,val}/<sample ID>.jpg
├── depth/{train,val}/<sample ID>.png
├── depth_dataset.generated.yaml
├── import_settings.json            # Saved settings and fixed group assignments
└── import_report.jsonl             # Results and source keys from the latest scan
```

RGB PNGs become quality-95 JPGs without changing their orientation or dimensions.
Depth is copied unchanged. Source hashes prevent filename collisions between
devices and cameras, and remote names are not used as local paths.

The default validation fraction is 20%. All cameras from a single capture stay
in the same split. Use `--split-by day` to keep each device's entire UTC day
together, which is a stronger boundary for nearby, similar captures. At least
two complete groups are needed to produce both splits. The initial split reserves
at least one group for each split; group counts, rather than exact image counts,
determine the fraction. Set `--val-fraction` and `--seed` to adjust it.

Rerun the **same command** to resume. Verified local raw files are reused, newly
arrived files are downloaded, and existing groups keep their split assignments.
Images are regenerated from the raw copies. Incomplete S3 pairs are downloaded
to `raw/`, reported, and excluded from training until their missing partner arrives.
Files upload independently, so an incomplete pair can simply mean an upload is
still in progress. Invalid images and download failures are reported, other
samples continue, and the command exits with an error so failures are visible.
Conditional downloads reject objects that changed since listing; rerunning scans
their new versions. New dataset YAML is written only after all complete samples
pass and both splits contain usable images.

Use a new output directory to change import settings. Local files from earlier
imports are retained even if their source objects later disappear from S3.
The script reads S3 and does not change remote objects or the producer/uploader.
`snapshot.json` is not required or downloaded; pairing uses the actual listed image
keys because individual files can arrive out of order.

Then train directly:

```powershell
python train_model.py --data datasets/snapshots --device 0
```

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
  splits. `train_model.py` does not split or move files; the S3 importer creates
  the splits described above.
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
AdamW with an initial learning rate of `0.0001` and gentle warmup, and zero data
loader workers. These learning-rate settings follow the
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

`download_snapshots.py` coordinates the S3 import. Its helpers are
`helper/snapshot_s3.py` for listing, credentials, and downloads, and
`helper/snapshot_dataset.py` for splitting, conversion, and the import report.
Run `python download_snapshots.py --help` for importer settings. To run the tests:

```powershell
python -m unittest discover -s tests -v
```
