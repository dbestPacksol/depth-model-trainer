"""Exercise the data contract with real JPEGs and 16-bit depth PNGs."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from helper.dataset import prepare_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "data with spaces"
        for split in ("train", "val"):
            image_path = self.root / "images" / split / "session_a" / "frame.jpg"
            depth_path = self.root / "depth" / split / "session_a" / "frame.png"
            image_path.parent.mkdir(parents=True)
            depth_path.parent.mkdir(parents=True)
            Image.new("RGB", (48, 32), color=(120, 80, 40)).save(image_path)
            Image.fromarray(np.full((32, 48), 1500, dtype=np.uint16)).save(depth_path)
        self.image_path = self.root / "images/train/session_a/frame.jpg"
        self.depth_path = self.root / "depth/train/session_a/frame.png"

    def prepare(self):
        return prepare_dataset(self.root, depth_scale=1000, max_depth=10)

    def test_yaml_and_pairing_work_with_the_real_ultralytics_loader(self):
        from ultralytics.data.dataset import DepthDataset
        from ultralytics.data.utils import check_det_dataset, load_depth
        from ultralytics.utils import DEFAULT_CFG

        configuration = check_det_dataset(str(self.prepare()), autodownload=False)
        dataset = DepthDataset(
            img_path=configuration["train"],
            data=configuration,
            task="depth",
            imgsz=64,
            batch_size=1,
            augment=False,
            hyp=DEFAULT_CFG,
        )
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertIn("depth", sample)
        self.assertTrue((sample["depth"] > 0).any())
        np.testing.assert_allclose(load_depth(self.depth_path, configuration["depth_scale"]), 1.5)
        self.assertEqual(configuration["max_depth"], 10)

    def test_custom_depth_scale_is_preserved(self):
        path = prepare_dataset(self.root, depth_scale=256, max_depth=80)
        configuration = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(configuration["depth_scale"], 256)
        self.assertEqual(configuration["path"], self.root.resolve().as_posix())

    def test_smaller_registered_depth_is_accepted(self):
        Image.fromarray(np.full((16, 24), 1500, dtype=np.uint16)).save(self.depth_path)
        self.prepare()

    def test_training_reuses_imported_depth_settings_unless_overridden(self):
        path = prepare_dataset(self.root, depth_scale=256, max_depth=80)
        prepare_dataset(self.root)
        configuration = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(configuration["depth_scale"], 256)
        self.assertEqual(configuration["max_depth"], 80)
        prepare_dataset(self.root, depth_scale=1000)
        configuration = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(configuration["depth_scale"], 1000)
        self.assertEqual(configuration["max_depth"], 80)

    def test_bad_depth_images_are_rejected(self):
        cases = (
            (np.ones((32, 48), dtype=np.uint8), "16-bit"),
            (np.ones((32, 48, 3), dtype=np.uint8), "single-channel"),
            (np.zeros((32, 48), dtype=np.uint16), "no valid pixels"),
            (np.ones((20, 48), dtype=np.uint16), "aspect ratio"),
        )
        for pixels, message in cases:
            with self.subTest(message=message):
                Image.fromarray(pixels).save(self.depth_path)
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare()

    def test_missing_depth_is_rejected(self):
        self.depth_path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing depth PNG"):
            self.prepare()

    def test_orphan_depth_is_rejected(self):
        self.image_path.rename(self.image_path.with_name("renamed.jpg"))
        self.depth_path.with_name("renamed.png").write_bytes(self.depth_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "no matching RGB"):
            self.prepare()

    def test_missing_validation_split_is_rejected(self):
        (self.root / "images/val/session_a/frame.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "No .jpg images"):
            self.prepare()

    def test_damaged_rgb_is_rejected(self):
        self.image_path.write_bytes(b"not an image")
        with self.assertRaises(OSError):
            self.prepare()

    def test_check_data_command_writes_configuration_without_training(self):
        result = subprocess.run(
            [sys.executable, "train_model.py", "--data", str(self.root), "--check-data"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Validated val: 1", result.stdout)
        self.assertTrue((self.root / "depth_dataset.generated.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
