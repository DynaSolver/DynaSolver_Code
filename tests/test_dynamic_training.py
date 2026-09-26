from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.dynamic_cfd import DynamicCFDOneStepDataset, DynamicStateNormalizer
from dynamic_cfd.training import rollout_trajectory, run_one_step_epoch, weighted_masked_mse
from tests.test_dynamic_cfd_data import write_trajectory


class BiasDeltaModel(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.delta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, query_xyz, static_fx, previous_state, boundary_feat):
        del query_xyz, static_fx, boundary_feat
        return previous_state + self.delta


class DynamicTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        entries = []
        for index, split in enumerate(("train", "validation", "test")):
            path = self.root / f"{split}.h5"
            write_trajectory(path, offset=float(index))
            entries.append(
                {
                    "trajectory_id": f"h1_{split}",
                    "robot": "H1",
                    "split": split,
                    "hdf5_path": path.name,
                    "training_data_eligible": True,
                    "point_bank_id": "h1_bank_v1",
                }
            )
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps({"schema_version": 1, "trajectories": entries}), encoding="utf-8"
        )
        self.normalizer = DynamicStateNormalizer.fit_from_manifest(
            self.manifest, u_ref_m_per_s=2.0, l_ref_m=1.5
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_weighted_masked_mse_ignores_unsupervised_points(self):
        prediction = torch.tensor([[[1.0], [1000.0], [3.0]]])
        target = torch.tensor([[[0.0], [0.0], [1.0]]])
        mask = torch.tensor([[True, False, True]])
        weight = torch.tensor([[1.0, 1.0, 3.0]])
        loss = weighted_masked_mse(prediction, target, mask, weight)
        self.assertAlmostEqual(float(loss), (1.0 + 3.0 * 4.0) / 4.0)

    def test_one_step_training_and_rollout_run_on_hdf5(self):
        train = DynamicCFDOneStepDataset(
            self.manifest, "train", self.normalizer, subset_points=12
        )
        validation = DynamicCFDOneStepDataset(
            self.manifest, "validation", self.normalizer, subset_points=12
        )
        self.addCleanup(train.close)
        self.addCleanup(validation.close)
        model = BiasDeltaModel(6)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        metrics = run_one_step_epoch(
            model, DataLoader(train, batch_size=1), torch.device("cpu"), optimizer
        )
        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertGreater(float(model.delta.grad.abs().sum()), 0.0)
        rollout = rollout_trajectory(
            model, validation, "h1_validation", torch.device("cpu"), max_steps=2
        )
        self.assertEqual(rollout["steps"], 2)
        self.assertEqual(len(rollout["channel_relative_l2"]), 6)
        self.assertTrue(np.isfinite(rollout["relative_l2_final"]))

    def test_training_cli_writes_epoch_history_checkpoint_and_rollout(self):
        output_dir = self.root / "run"
        command = [
            sys.executable,
            "scripts/train_dynamic_cfd_v0.py",
            "--manifest", str(self.manifest),
            "--output-dir", str(output_dir),
            "--initialization", "from_scratch",
            "--allow-unhashed-data",
            "--points", "12",
            "--epochs", "1",
            "--early-stop-patience", "1",
            "--batch-size", "1",
            "--validation-rollout-steps", "2",
            "--l-ref", "1.5",
            "--device", "cpu",
            "--hidden-dim", "32",
            "--layers", "2",
            "--heads", "4",
            "--slices", "4",
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        for name in (
            "normalizer.json",
            "run_manifest.json",
            "history.json",
            "epoch_history.csv",
            "checkpoint_best.pt",
            "summary.json",
        ):
            self.assertTrue((output_dir / name).is_file(), name)
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["epochs_completed"], 1)
        self.assertEqual(summary["test_full_cycle_rollout"]["trajectories"][0]["steps"], 2)
        self.assertIn("persistence_full_cycle_rollout", summary)


if __name__ == "__main__":
    unittest.main()
