from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import yaml

from data_provider.dynamic_cfd import load_trajectory_manifest, validate_trajectory_hdf5


REQUIRED_DATASET_GATES = (
    "all_formal_trajectories_completed",
    "wall_layer_gate_passed",
    "stationary_initialization_passed",
    "time_step_sensitivity_passed",
    "pose_spacing_gate_passed",
    "smooth_ramp_excluded",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_true(mapping: Mapping[str, Any], keys: tuple[str, ...] | list[str], name: str) -> None:
    failed = [key for key in keys if mapping.get(key) is not True]
    if failed:
        raise ValueError(f"{name} gates are missing or not true: {failed}")


def _point_bank_digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with h5py.File(path, "r") as handle:
        for name in ("points_world_m", "query_xyz_normalized"):
            values = np.ascontiguousarray(handle[name][...])
            digest.update(name.encode("ascii"))
            digest.update(str(values.dtype).encode("ascii"))
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.tobytes())
    return digest.hexdigest()


def validate_production_contract(
    manifest_path: str | Path,
    contract_path: str | Path,
    *,
    verify_hashes: bool = True,
    validate_values: bool = True,
) -> dict[str, Any]:
    """Enforce the registered 12-trajectory SST contract before formal training."""
    manifest_path = Path(manifest_path).resolve()
    contract_path = Path(contract_path).resolve()
    contract = _mapping(
        yaml.safe_load(contract_path.read_text(encoding="utf-8")), "contract"
    )
    formal = _mapping(contract.get("formal_dataset"), "formal_dataset")
    expected_rows = formal.get("trajectories")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise ValueError("formal_dataset.trajectories must be a non-empty list")
    expected = {row["id"]: row for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("formal_dataset contains duplicate trajectory ids")

    raw_manifest = _mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")), "manifest"
    )
    raw_entries = raw_manifest.get("trajectories")
    if not isinstance(raw_entries, list):
        raise ValueError("manifest.trajectories must be a list")
    actual_by_id = {entry.get("trajectory_id"): entry for entry in raw_entries}
    if set(actual_by_id) != set(expected):
        missing = sorted(set(expected) - set(actual_by_id))
        unexpected = sorted(set(actual_by_id) - set(expected))
        raise ValueError(
            f"production trajectory ids do not match contract; missing={missing}, unexpected={unexpected}"
        )

    metadata = _mapping(raw_manifest.get("dataset_metadata"), "dataset_metadata")
    physics = _mapping(contract.get("physics"), "physics")
    if metadata.get("pressure_kind") != "kinematic":
        raise ValueError("dataset_metadata.pressure_kind must be kinematic")
    if not isinstance(metadata.get("pressure_gauge_policy"), str) or not metadata[
        "pressure_gauge_policy"
    ].strip():
        raise ValueError("dataset_metadata.pressure_gauge_policy must be documented")
    for key in ("density_kg_per_m3", "kinematic_viscosity_m2_per_s"):
        if not np.isclose(float(metadata.get(key, np.nan)), float(physics[key])):
            raise ValueError(f"dataset_metadata.{key} does not match the contract")
    if not np.allclose(
        metadata.get("inflow_velocity_m_per_s", []), physics["inflow_velocity_m_per_s"]
    ):
        raise ValueError("dataset_metadata.inflow_velocity_m_per_s does not match the contract")
    expected_l_ref = float(
        _mapping(_mapping(contract.get("export"), "export").get("normalization"), "normalization")[
            "reference_length_m"
        ]
    )
    if not np.isclose(float(metadata.get("l_ref_m", np.nan)), expected_l_ref):
        raise ValueError("dataset_metadata.l_ref_m does not match the normalization contract")

    dataset_gates = _mapping(raw_manifest.get("dataset_gates"), "dataset_gates")
    _require_true(dataset_gates, REQUIRED_DATASET_GATES, "dataset")
    selected_spacing = float(dataset_gates.get("selected_pose_spacing_s", np.nan))
    spacing = _mapping(
        _mapping(contract.get("cfd_gates"), "cfd_gates").get("pose_spacing"),
        "pose_spacing",
    )
    if not any(
        np.isclose(selected_spacing, float(spacing[key]))
        for key in ("candidate_s", "fallback_s")
    ):
        raise ValueError("dataset_gates.selected_pose_spacing_s must be accepted 1 ms or 2 ms")

    manifest = load_trajectory_manifest(
        manifest_path,
        require_all_splits=True,
        require_production_eligible=True,
        require_hashes=True,
        verify_hashes=verify_hashes,
    )
    expected_count = int(formal["required_trajectory_count"])
    if len(manifest.records) != expected_count:
        raise ValueError(f"production contract requires {expected_count} trajectories")

    split_counts = {split: 0 for split in ("train", "validation", "test")}
    robot_split_counts: dict[str, dict[str, int]] = {}
    expected_points = int(_mapping(contract.get("point_bank"), "point_bank")["full_point_count"])
    model_contract = _mapping(contract.get("model_contract"), "model_contract")
    expected_direction = np.asarray(physics["inflow_direction"], dtype=np.float64)
    expected_normalized_speed = float(physics["inflow_speed_m_per_s"]) / float(
        model_contract["speed_normalization_m_per_s"]
    )
    required_interval_gates = list(
        _mapping(contract.get("cfd_gates"), "cfd_gates")["interval_required_gates"]
    )
    point_bank_digests: dict[tuple[str, str], str] = {}
    point_bank_ids_by_robot: dict[str, set[str]] = {}

    for record in manifest.records:
        expected_row = expected[record.trajectory_id]
        entry = _mapping(actual_by_id[record.trajectory_id], record.trajectory_id)
        for key, actual in (("robot", record.robot), ("split", record.split)):
            if actual != expected_row[key]:
                raise ValueError(f"{record.trajectory_id}.{key} does not match the contract")
        for key in ("cadence_scale", "amplitude_scale"):
            if not np.isclose(float(entry.get(key, np.nan)), float(expected_row[key])):
                raise ValueError(f"{record.trajectory_id}.{key} does not match the contract")
        if entry.get("solver") != formal["solver"]:
            raise ValueError(f"{record.trajectory_id}.solver must be {formal['solver']}")
        if entry.get("diagnostic_only") is not False:
            raise ValueError(f"{record.trajectory_id}.diagnostic_only must be false")
        _require_true(
            _mapping(entry.get("interval_gates"), f"{record.trajectory_id}.interval_gates"),
            required_interval_gates,
            record.trajectory_id,
        )

        shape = validate_trajectory_hdf5(
            record.hdf5_path,
            expected_state_dim=6,
            validate_values=validate_values,
        )
        if shape.points != expected_points:
            raise ValueError(
                f"{record.trajectory_id} has {shape.points} points; contract requires {expected_points}"
            )
        with h5py.File(record.hdf5_path, "r") as handle:
            for frame in range(shape.frames):
                static_fx = np.asarray(handle["static_fx"][frame])
                if not np.allclose(
                    static_fx[:, 7:10], expected_direction, atol=1.0e-6, rtol=1.0e-6
                ):
                    raise ValueError(
                        f"{record.trajectory_id} static_fx inflow direction does not match the contract"
                    )
                if not np.allclose(
                    static_fx[:, 10], expected_normalized_speed, atol=1.0e-6, rtol=1.0e-6
                ):
                    raise ValueError(
                        f"{record.trajectory_id} static_fx normalized speed does not match the contract"
                    )
        bank_key = (record.robot, record.point_bank_id)
        point_bank_ids_by_robot.setdefault(record.robot, set()).add(record.point_bank_id)
        digest = _point_bank_digest(record.hdf5_path)
        previous = point_bank_digests.setdefault(bank_key, digest)
        if previous != digest:
            raise ValueError(f"point bank {bank_key} is not fixed across trajectories")
        split_counts[record.split] += 1
        robot_split_counts.setdefault(record.robot, {split: 0 for split in split_counts})[
            record.split
        ] += 1

    expected_splits = {key: int(value) for key, value in formal["split_counts"].items()}
    if split_counts != expected_splits:
        raise ValueError(f"split counts do not match contract: {split_counts}")
    expected_per_robot = {
        key: int(value) for key, value in formal["per_robot_split_counts"].items()
    }
    bad_robots = {
        robot: counts
        for robot, counts in robot_split_counts.items()
        if counts != expected_per_robot
    }
    if bad_robots:
        raise ValueError(f"per-robot split counts do not match contract: {bad_robots}")
    changed_banks = {
        robot: sorted(bank_ids)
        for robot, bank_ids in point_bank_ids_by_robot.items()
        if len(bank_ids) != 1
    }
    if changed_banks:
        raise ValueError(f"each robot must use exactly one fixed point bank: {changed_banks}")

    return {
        "status": "passed",
        "manifest": str(manifest_path),
        "contract": str(contract_path),
        "trajectory_count": len(manifest.records),
        "split_counts": split_counts,
        "per_robot_split_counts": robot_split_counts,
        "state_dim": 6,
        "point_count": expected_points,
        "selected_pose_spacing_s": selected_spacing,
        "l_ref_m": float(metadata["l_ref_m"]),
        "hashes_verified": verify_hashes,
        "values_validated": validate_values,
    }
