"""Exercise cloud publication and restore with real RGB/uint16 depth pairs."""

import hashlib
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import boto3
import numpy as np
import yaml
from botocore.exceptions import ClientError
from botocore.stub import Stubber, ANY
from PIL import Image

from helper.cloud_storage import download_run, read_manifest, safe_path, sync_dataset, upload_run


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.uploads = 0
        self.fail_after = None

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        data = self.objects[Key]
        return {"Body": io.BytesIO(data), "ETag": hashlib.md5(data).hexdigest()}

    def put_object(self, Bucket, Key, Body, IfMatch=None, IfNoneMatch=None, **kwargs):
        if (IfNoneMatch and Key in self.objects) or (IfMatch and hashlib.md5(self.objects[Key]).hexdigest() != IfMatch):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body

    def upload_file(self, path, bucket, key):
        self.uploads += 1
        if self.fail_after is not None and self.uploads > self.fail_after:
            raise OSError("Interrupted transfer")
        self.objects[key] = Path(path).read_bytes()

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[key])


class CloudStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "reviewed"
        self.client = MemoryS3()
        for split in ("train", "val"):
            self.add_pair(self.root, split, "session/frame")

    def add_pair(self, root, split, name):
        image = root / "images" / split / f"{name}.jpg"
        depth = root / "depth" / split / f"{name}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        depth.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 8), (120, 80, 40)).save(image)
        Image.fromarray(np.full((8, 12), 1500, dtype=np.uint16)).save(depth)

    def test_push_pull_sync_preserve_bytes_settings_and_exclude_raw(self):
        from helper.dataset import prepare_dataset
        prepare_dataset(self.root, 256, 80)
        (self.root / "raw").mkdir()
        (self.root / "raw/unreviewed.png").write_bytes(b"unreviewed")
        manifest = sync_dataset(self.client, "bucket", self.root, direction="push")
        self.assertEqual(len(manifest["files"]), 4)
        self.assertEqual(manifest["settings"], {"depth_scale": 256, "max_depth": 80})
        uploads = self.client.uploads
        self.assertEqual(sync_dataset(self.client, "bucket", self.root), manifest)
        self.assertEqual(self.client.uploads, uploads)
        restored = self.base / "other computer"
        sync_dataset(self.client, "bucket", restored, direction="pull")
        for relative in manifest["files"]:
            self.assertEqual((self.root / relative).read_bytes(), (restored / relative).read_bytes())
        self.assertEqual(yaml.safe_load((restored / "depth_dataset.generated.yaml").read_text())["path"], restored.as_posix())
        self.add_pair(restored, "train", "new")
        sync_dataset(self.client, "bucket", restored)
        sync_dataset(self.client, "bucket", self.root)
        self.assertTrue((self.root / "images/train/new.jpg").exists())

    def test_interrupted_upload_does_not_publish_and_retry_works(self):
        self.client.fail_after = 1
        with self.assertRaises(OSError):
            sync_dataset(self.client, "bucket", self.root, direction="push")
        self.assertNotIn("datasets/curated/manifest.json", self.client.objects)
        self.client.fail_after = None
        sync_dataset(self.client, "bucket", self.root)
        self.assertIn("datasets/curated/manifest.json", self.client.objects)

    def test_identical_content_is_uploaded_once_but_all_paths_are_published(self):
        manifest = sync_dataset(self.client, "bucket", self.root, direction="push")
        self.assertEqual(len(manifest["files"]), 4)
        self.assertEqual(self.client.uploads, 2)  # Same RGB and depth in both splits.
        destination = self.base / "deduplicated-restore"
        sync_dataset(self.client, "bucket", destination, direction="pull")
        for relative in manifest["files"]:
            self.assertEqual((destination / relative).read_bytes(), (self.root / relative).read_bytes())

    def test_different_local_pair_is_never_overwritten(self):
        sync_dataset(self.client, "bucket", self.root)
        path = self.root / "images/train/session/frame.jpg"
        Image.new("RGB", (12, 8), "blue").save(path)
        before = dict(self.client.objects)
        for direction in ("pull", "push", "sync"):
            with self.subTest(direction=direction), self.assertRaisesRegex(ValueError, "conflict"):
                sync_dataset(self.client, "bucket", self.root, direction=direction)
        self.assertEqual(self.client.objects, before)

    def test_corrupt_download_never_installs_pairs(self):
        manifest = sync_dataset(self.client, "bucket", self.root)
        entry = next(iter(manifest["files"].values()))
        self.client.objects[f"datasets/curated/objects/{entry['sha256']}"] = b"corrupt"
        destination = self.base / "restore"
        with self.assertRaisesRegex(ValueError, "checksum"):
            sync_dataset(self.client, "bucket", destination, direction="pull")
        self.assertFalse(destination.exists())

    def test_orphan_pair_and_settings_conflicts_rejected(self):
        sync_dataset(self.client, "bucket", self.root)
        config = self.root / "depth_dataset.generated.yaml"
        saved = yaml.safe_load(config.read_text())
        saved["depth_scale"] = 256
        config.write_text(yaml.safe_dump(saved))
        with self.assertRaisesRegex(ValueError, "settings differ"):
            sync_dataset(self.client, "bucket", self.root)
        config.unlink()
        (self.root / "depth/train/session/frame.png").unlink()
        with self.assertRaisesRegex(ValueError, "Missing depth"):
            sync_dataset(self.client, "bucket", self.root, direction="push")

    def test_empty_bucket_pull_and_invalid_paths(self):
        with self.assertRaisesRegex(ValueError, "No curated"):
            sync_dataset(self.client, "bucket", self.base / "missing", direction="pull")
        for path in ("../escape", "/absolute", "C:/drive", "a\\b", "a/../b", "CON", "a/file."):
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_path(self.root, path)

    def test_training_depth_overrides_initialize_but_cannot_reinterpret_cloud(self):
        manifest = sync_dataset(self.client, "bucket", self.root, depth_scale=256, max_depth=80)
        self.assertEqual(manifest["settings"], {"depth_scale": 256, "max_depth": 80})
        with self.assertRaisesRegex(ValueError, "differs from cloud"):
            sync_dataset(self.client, "bucket", self.root, depth_scale=1000)

    def test_incomplete_remote_manifest_is_rejected_before_downloading(self):
        manifest = sync_dataset(self.client, "bucket", self.root)
        del manifest["files"]["depth/train/session/frame.png"]
        self.client.objects["datasets/curated/manifest.json"] = json.dumps(manifest).encode()
        destination = self.base / "restore"
        with self.assertRaisesRegex(ValueError, "Incomplete cloud pair"):
            sync_dataset(self.client, "bucket", destination, direction="pull")
        self.assertFalse(destination.exists())

    def test_configuration_reports_missing_names_without_secrets(self):
        from helper.cloud_config import cloud_settings
        with patch.dict(os.environ, {"ACCESS_KEY": "", "SECRET_KEY": "", "BUCKET_NAME": ""}):
            with self.assertRaisesRegex(ValueError, "ACCESS_KEY, SECRET_KEY, BUCKET_NAME"):
                cloud_settings()

    def test_immutable_runs_round_trip(self):
        run = self.base / "run"
        (run / "weights").mkdir(parents=True)
        (run / "weights/best.pt").write_bytes(b"weights")
        (run / "results.csv").write_text("epoch,loss\n1,0.1\n")
        (run / ".env").write_text("SECRET_KEY=excluded")
        upload_run(self.client, "bucket", run, "experiment-1", {"dataset": "curated"})
        with self.assertRaisesRegex(ValueError, "already exists"):
            upload_run(self.client, "bucket", run, "experiment-1")
        destination = self.base / "restored run"
        download_run(self.client, "bucket", destination, "experiment-1")
        self.assertEqual((destination / "weights/best.pt").read_bytes(), b"weights")
        self.assertTrue((destination / "results.csv").is_file())
        self.assertFalse((destination / ".env").exists())

    def test_real_sdk_accepts_conditional_manifest_write(self):
        client = boto3.client("s3", aws_access_key_id="test", aws_secret_access_key="test", region_name="us-east-1")
        self.addCleanup(client.close)
        with Stubber(client) as stubber:
            stubber.add_client_error("get_object", "NoSuchKey", expected_params={"Bucket": "bucket", "Key": "datasets/curated/manifest.json"})
            stubber.add_response("put_object", {}, {"Bucket": "bucket", "Key": "datasets/curated/manifest.json",
                                                  "Body": ANY, "ContentType": "application/json", "IfNoneMatch": "*"})
            with patch.object(client, "upload_file"):
                sync_dataset(client, "bucket", self.root, direction="push")
            stubber.assert_no_pending_responses()

    def test_training_syncs_and_archives_actual_numbered_run_directory(self):
        import train_model

        run = self.base / "train2"
        (run / "weights").mkdir(parents=True)
        (run / "weights/best.pt").write_bytes(b"best")
        (run / "weights/last.pt").write_bytes(b"last")
        model = Mock(task="depth", trainer=types.SimpleNamespace(save_dir=run))
        fake_ultralytics = types.SimpleNamespace(YOLO=Mock(return_value=model))
        with patch.dict("sys.modules", {"ultralytics": fake_ultralytics}), patch(
            "helper.cloud_config.cloud_settings", return_value=(self.client, "bucket")
        ), patch("sys.argv", ["train_model.py", "--data", str(self.root), "--sync-s3", "--upload-results",
                             "--workers", "4", "--cache", "ram", "--no-deterministic", "--no-plots",
                             "--device", "0,1", "--batch", "32"]):
            train_model.main()
        model.train.assert_called_once()
        settings = model.train.call_args.kwargs
        self.assertEqual((settings["device"], settings["batch"], settings["workers"]), ("0,1", 32, 4))
        self.assertEqual(settings["cache"], "ram")
        self.assertFalse(settings["deterministic"])
        self.assertFalse(settings["plots"])
        key = next(k for k in self.client.objects if k.startswith("runs/") and k.endswith("manifest.json"))
        manifest = json.loads(self.client.objects[key])
        self.assertIn("weights/last.pt", manifest["files"])
        self.assertEqual(manifest["provenance"]["dataset"], "curated")
        self.assertEqual(len(manifest["provenance"]["manifest"]["files"]), 4)

    def test_check_data_with_upload_flag_stays_local(self):
        import train_model

        with patch("helper.cloud_config.cloud_settings") as cloud, patch(
            "sys.argv", ["train_model.py", "--data", str(self.root), "--check-data", "--upload-results"]
        ):
            train_model.main()
        cloud.assert_not_called()

    def test_concurrent_manifest_update_reports_retry(self):
        sync_dataset(self.client, "bucket", self.root)
        self.add_pair(self.root, "train", "new")
        def conflict(**kwargs):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        with patch.object(self.client, "put_object", side_effect=conflict):
            with self.assertRaisesRegex(ValueError, "concurrently"):
                sync_dataset(self.client, "bucket", self.root, direction="push")


if __name__ == "__main__":
    unittest.main()
