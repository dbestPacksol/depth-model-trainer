"""Validate explicit multi-GPU batch configuration before launching DDP."""

import contextlib
import io
import unittest
from unittest.mock import patch

from train_model import parse_arguments


class TrainingArgumentsTests(unittest.TestCase):
    def test_uneven_multi_gpu_batch_is_rejected_before_training(self):
        stderr = io.StringIO()
        with patch("sys.argv", ["train_model.py", "--data", "unused", "--device", "0,1", "--batch", "15"]), \
                contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            parse_arguments()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("divisible", stderr.getvalue())

    def test_odd_batch_remains_valid_for_single_gpu(self):
        with patch("sys.argv", ["train_model.py", "--data", "unused", "--device", "0", "--batch", "15"]):
            self.assertEqual(parse_arguments().batch, 15)
