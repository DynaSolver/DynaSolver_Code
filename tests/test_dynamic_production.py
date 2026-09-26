from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml
import h5py

from dynamic_cfd.production import validate_production_contract
from tests.test_dynamic_cfd_data import write_trajectory


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DynamicProductionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        source_contract = Path(__file__).resolve().parents[1] / "configs/h1_g1_dynamic_v0.yaml"
        self.contract = yaml.safe_load(source_contract.read_text(encoding="utf-8"))
        self.contract["point_bank"]["full_point_count"] = 12
        self.contract_path = self.root / "contract.yaml"
        self.contract_path.write_text(yaml.safe_dump(self.contract), encoding="utf-8")

        required_interval_gates = self.contract["cfd_gates"]["interval_required_gates"]
        entries = []
        for index, row in enumerate(self.contract["formal_dataset"]["trajectories"]):
            path = self.root / f"trajectory_{index:02d}.h5"
            write_trajectory(path, offset=float(index))
            with h5py.File(path, "a") as handle:
                handle["static_fx"][..., 7:10] = self.contract["physics"][
                    "inflow_direction"
                ]
                handle["static_fx"][..., 10] = (
                    self.contract["physics"]["inflow_speed_m_per_s"]
                    / self.contract["model_contract"]["speed_normalization_m_per_s"]
                )
            entries.append(
                {
                    "trajectory_id": row["id"],
                    "robot": row["robot"],
                    "split": row["split"],
                    "cadence_scale": row["cadence_scale"],
                    "amplitude_scale": row["amplitude_scale"],
                    "solver": self.contract["formal_dataset"]["solver"],
                    "hdf5_path": path.name,
                    "sha256": sha256(path),
                    "training_data_eligible": True,
                    "diagnostic_only": False,
                    "point_bank_id": f"{row['robot'].lower()}_bank_v1",
                    "interval_gates": {gate: True for gate in required_interval_gates},
                }
            )
        self.payload = {
            "schema_version": 1,
            "dataset_metadata": {
                "pressure_kind": "kinematic",
                "pressure_gauge_policy": "farfield_reference_zero",
                "density_kg_per_m3": self.contract["physics"]["density_kg_per_m3"],
                "kinematic_viscosity_m2_per_s": self.contract["physics"][
                    "kinematic_viscosity_m2_per_s"
                ],
                "inflow_velocity_m_per_s": self.contract["physics"][
                    "inflow_velocity_m_per_s"
                ],
                "l_ref_m": 1.0,
            },
            "dataset_gates": {
                "all_formal_trajectories_completed": True,
                "wall_layer_gate_passed": True,
                "stationary_initialization_passed": True,
                "time_step_sensitivity_passed": True,
                "pose_spacing_gate_passed": True,
                "smooth_ramp_excluded": True,
                "selected_pose_spacing_s": 0.001,
            },
            "trajectories": entries,
        }
        self.manifest_path = self.root / "manifest.json"
        self._write_manifest()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.payload), encoding="utf-8")

    def test_complete_twelve_trajectory_contract_passes(self):
        report = validate_production_contract(self.manifest_path, self.contract_path)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["split_counts"], {"train": 8, "validation": 2, "test": 2})
        self.assertEqual(report["per_robot_split_counts"]["H1"], {
            "train": 4,
            "validation": 1,
            "test": 1,
        })

    def test_contract_rejects_motion_or_gate_drift(self):
        self.payload["trajectories"][0]["cadence_scale"] = 9.0
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "cadence_scale"):
            validate_production_contract(self.manifest_path, self.contract_path)

        self.payload["trajectories"][0]["cadence_scale"] = self.contract[
            "formal_dataset"
        ]["trajectories"][0]["cadence_scale"]
        self.payload["dataset_gates"]["time_step_sensitivity_passed"] = False
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "dataset gates"):
            validate_production_contract(self.manifest_path, self.contract_path)

    def test_contract_rejects_multiple_point_banks_for_one_robot(self):
        self.payload["trajectories"][0]["point_bank_id"] = "h1_other_bank"
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "exactly one fixed point bank"):
            validate_production_contract(self.manifest_path, self.contract_path)

    def test_contract_rejects_wrong_inflow_features(self):
        path = self.root / self.payload["trajectories"][0]["hdf5_path"]
        with h5py.File(path, "a") as handle:
            handle["static_fx"][..., 7:10] = [-1.0, 0.0, 0.0]
        self.payload["trajectories"][0]["sha256"] = sha256(path)
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "inflow direction"):
            validate_production_contract(self.manifest_path, self.contract_path)


if __name__ == "__main__":
    unittest.main()
