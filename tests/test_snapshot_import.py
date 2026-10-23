"""Check S3 pagination, interrupted uploads, and the resulting real image dataset."""

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import boto3
import numpy as np
import yaml
from botocore.response import StreamingBody
from botocore.stub import Stubber
from PIL import Image

from helper.snapshot_dataset import assign_splits, build_dataset, sample_id, split_group
from helper.snapshot_s3 import download_object, find_snapshots


def png_bytes(pixels):
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


class SnapshotImportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "dataset with spaces"
        self.client = boto3.client("s3", aws_access_key_id="testing", aws_secret_access_key="testing", region_name="us-east-1")
        self.addCleanup(self.client.close)
        self.stubber = Stubber(self.client)
        self.stubber.activate()
        self.addCleanup(self.stubber.deactivate)
        self.settings = {
            "bucket": "test-bucket", "prefix": "", "val_fraction": 0.2, "seed": 0,
            "split_by": "snapshot", "depth_scale": 256, "max_depth": 80,
        }
        self.depth = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48) + 100
        self.objects = {}
        for capture in (1700000000, 1700000100):
            for camera in ("camera_a", "camera_b"):
                prefix = f"device/snapshots/snapshot_{capture}/{camera}"
                self.objects[f"{prefix}/rgb.png"] = png_bytes(np.full((32, 48, 3), (180, 60, 20), dtype=np.uint8))
                self.objects[f"{prefix}/depth.png"] = png_bytes(self.depth)

    def object_info(self, key):
        data = self.objects[key]
        return {"Key": key, "ETag": f'"{hashlib.md5(data).hexdigest()}"', "Size": len(data)}

    def queue_listing(self, keys=None):
        keys = sorted(self.objects) if keys is None else keys
        self.stubber.add_response(
            "list_objects_v2", {"Contents": [self.object_info(key) for key in keys], "IsTruncated": False},
            {"Bucket": "test-bucket", "Prefix": ""},
        )

    def queue_download(self, source):
        data = self.objects[source["key"]]
        self.stubber.add_response(
            "get_object", {"Body": StreamingBody(io.BytesIO(data), len(data)), "ContentLength": len(data)},
            {"Bucket": "test-bucket", "Key": source["key"], "IfMatch": source["etag"]},
        )

    def discover_and_queue_downloads(self):
        self.queue_listing()
        samples = find_snapshots(self.client, "test-bucket")
        for sample in samples:
            for source in sample["objects"].values():
                self.queue_download(source)
        return samples

    def test_all_pages_are_read_and_videos_are_ignored(self):
        keys = sorted(self.objects)
        first = [self.object_info(keys[0]), {"Key": "device/events/event_1/camera/rgb.png", "ETag": '"x"', "Size": 1}]
        self.stubber.add_response("list_objects_v2", {
            "Contents": first, "IsTruncated": True, "NextContinuationToken": "next-page",
        }, {"Bucket": "test-bucket", "Prefix": ""})
        self.stubber.add_response("list_objects_v2", {
            "Contents": [self.object_info(key) for key in keys[1:]], "IsTruncated": False,
        }, {"Bucket": "test-bucket", "Prefix": "", "ContinuationToken": "next-page"})
        samples = find_snapshots(self.client, "test-bucket")
        self.assertEqual(len(samples), 4)
        self.assertTrue(all(len(sample["objects"]) == 2 for sample in samples))
        self.stubber.assert_no_pending_responses()

    def test_dataset_preserves_depth_colors_pairing_and_resumes_without_downloads(self):
        samples = self.discover_and_queue_downloads()
        summary = build_dataset(self.client, samples, self.output, self.settings)
        self.assertEqual(summary, {"imported": 4, "incomplete": 0, "failed": 0})
        configuration = yaml.safe_load((self.output / "depth_dataset.generated.yaml").read_text())
        self.assertEqual(configuration["depth_scale"], 256)
        report = [json.loads(line) for line in (self.output / "import_report.jsonl").read_text().splitlines()]
        capture_splits = {}
        for row in report:
            capture_splits.setdefault(row["capture"], set()).add(row["split"])
            depth_path = self.output / row["depth"]
            np.testing.assert_array_equal(np.asarray(Image.open(depth_path)), self.depth)
            self.assertEqual(depth_path.read_bytes(), self.objects[row["source"]["depth.png"]["key"]])
            with Image.open(self.output / row["image"]) as image:
                self.assertEqual(image.format, "JPEG")
                red, green, blue = image.getpixel((10, 10))
                self.assertGreater(red, green)
                self.assertGreater(green, blue)
        self.assertTrue(all(len(splits) == 1 for splits in capture_splits.values()))
        self.assertEqual({row["split"] for row in report}, {"train", "val"})
        # Stubber has no responses left: any unexpected network request fails this repeat import.
        self.assertEqual(build_dataset(self.client, samples, self.output, self.settings), summary)
        self.stubber.assert_no_pending_responses()

    def test_incomplete_pair_is_downloaded_raw_and_reported(self):
        key = "device/snapshots/snapshot_1700000200/camera_c/rgb.png"
        self.objects[key] = png_bytes(np.full((32, 48, 3), 100, dtype=np.uint8))
        samples = self.discover_and_queue_downloads()
        summary = build_dataset(self.client, samples, self.output, self.settings)
        self.assertEqual(summary["imported"], 4)
        self.assertEqual(summary["incomplete"], 1)
        identifier = sample_id(samples[-1], "test-bucket")
        self.assertTrue((self.output / "raw" / identifier / "rgb.png").is_file())
        self.assertEqual(len(list((self.output / "images").rglob("*.jpg"))), 4)

    def test_changed_s3_object_is_rejected_and_other_pairs_continue(self):
        self.queue_listing()
        samples = find_snapshots(self.client, "test-bucket")
        first_source = next(iter(samples[0]["objects"].values()))
        self.stubber.add_client_error("get_object", service_error_code="PreconditionFailed", http_status_code=412,
                                     expected_params={"Bucket": "test-bucket", "Key": first_source["key"],
                                                      "IfMatch": first_source["etag"]})
        for sample in samples[1:]:
            for source in sample["objects"].values():
                self.queue_download(source)
        summary = build_dataset(self.client, samples, self.output, self.settings)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["imported"], 3)
        self.assertFalse((self.output / "depth_dataset.generated.yaml").exists())
        self.assertFalse(list(self.output.rglob("*.part")))
        self.stubber.assert_no_pending_responses()

    def test_corrupt_local_cache_is_downloaded_again(self):
        self.queue_listing()
        sample = find_snapshots(self.client, "test-bucket")[0]
        source = sample["objects"]["rgb.png"]
        destination = self.output / "cache.png"
        self.queue_download(source)
        download_object(self.client, "test-bucket", source, destination)
        destination.write_bytes(b"x" * destination.stat().st_size)
        self.queue_download(source)
        download_object(self.client, "test-bucket", source, destination)
        self.assertEqual(destination.read_bytes(), self.objects[source["key"]])

    def test_new_groups_do_not_move_previous_assignments(self):
        previous = assign_splits({"one", "two"}, {}, 0.2, 0)
        updated = assign_splits({"one", "two", "three"}, previous, 0.2, 0)
        self.assertTrue(all(updated[group] == split for group, split in previous.items()))

    def test_device_day_grouping_and_safe_local_names(self):
        self.queue_listing()
        samples = find_snapshots(self.client, "test-bucket")
        self.assertEqual(split_group(samples[0], "day"), split_group(samples[-1], "day"))
        samples[0]["camera_prefix"] += "/../C:/unsafe"
        self.assertRegex(sample_id(samples[0], "test-bucket"), r"^snapshot_\d+_[a-f0-9]{24}$")

    def test_one_capture_keeps_raw_files_but_does_not_create_a_training_yaml(self):
        self.objects = {key: value for key, value in self.objects.items() if "1700000000" in key}
        samples = self.discover_and_queue_downloads()
        with self.assertRaisesRegex(ValueError, "at least two complete split groups"):
            build_dataset(self.client, samples, self.output, self.settings)
        self.assertEqual(len(list((self.output / "raw").rglob("rgb.png"))), 2)
        self.assertFalse((self.output / "depth_dataset.generated.yaml").exists())

    def test_existing_unrelated_output_is_protected(self):
        self.output.mkdir()
        (self.output / "important.txt").write_text("keep me")
        with self.assertRaisesRegex(ValueError, "Output is not empty"):
            build_dataset(self.client, [], self.output, self.settings)
        self.assertEqual((self.output / "important.txt").read_text(), "keep me")

    def test_truncated_download_is_not_published_or_cached(self):
        source = {"key": "device/snapshots/snapshot_1/camera/rgb.png", "etag": '"test"', "size": 100}
        self.stubber.add_response(
            "get_object", {"Body": StreamingBody(io.BytesIO(b"partial"), 7)},
            {"Bucket": "test-bucket", "Key": source["key"], "IfMatch": source["etag"]},
        )
        destination = self.output / "raw/rgb.png"
        with self.assertRaisesRegex(OSError, "Incomplete download"):
            download_object(self.client, "test-bucket", source, destination)
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_name("rgb.png.s3.json").exists())
        self.assertFalse(list(self.output.rglob("*.part")))

    def test_invalid_depth_is_reported_without_publishing_a_half_pair(self):
        key = next(key for key in sorted(self.objects) if key.endswith("depth.png"))
        self.objects[key] = png_bytes(np.full((32, 48), 10, dtype=np.uint8))
        samples = self.discover_and_queue_downloads()
        summary = build_dataset(self.client, samples, self.output, self.settings)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["imported"], 3)
        identifier = sample_id(samples[0], "test-bucket")
        self.assertFalse(list((self.output / "images").rglob(f"{identifier}.jpg")))
        self.assertFalse(list((self.output / "depth").rglob(f"{identifier}.png")))

    def test_default_depth_scale_uses_orbbec_millimeters(self):
        from download_snapshots import parse_arguments

        with patch("sys.argv", ["download_snapshots.py", "--list-only"]):
            self.assertEqual(parse_arguments().depth_scale, 1000)


if __name__ == "__main__":
    unittest.main()
