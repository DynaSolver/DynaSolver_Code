from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SPLITS = ("train", "validation", "test")
STATE_DIMS = (4, 6)


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    robot: str
    split: str
    hdf5_path: Path
    training_data_eligible: bool
    point_bank_id: str
    sha256: str | None = None
    accepted_transition_indices: tuple[int, ...] | None = None
    valid_frame_ids: tuple[int, ...] | None = None
    transition_quality_audit_sha256: str | None = None
    speed_m_per_s: float | None = None


@dataclass(frozen=True)
class TrajectoryManifest:
    path: Path
    records: tuple[TrajectoryRecord, ...]

    def records_for_split(
        self,
        split: str,
        *,
        require_production_eligible: bool = True,
    ) -> tuple[TrajectoryRecord, ...]:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        records = tuple(record for record in self.records if record.split == split)
        if not records:
            raise ValueError(f"manifest has no {split!r} trajectories")
        if require_production_eligible:
            rejected = [record.trajectory_id for record in records if not record.training_data_eligible]
            if rejected:
                raise ValueError(
                    f"split {split!r} contains trajectories that are not production eligible: {rejected}"
                )
        return records


@dataclass(frozen=True)
class TrajectoryShape:
    frames: int
    points: int
    state_dim: int
    links: int


def _required_string(entry: Mapping[str, Any], key: str, trajectory_index: int) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"trajectories[{trajectory_index}].{key} must be a non-empty string")
    return value


def load_trajectory_manifest(
    path: str | Path,
    *,
    require_all_splits: bool = True,
    require_production_eligible: bool = False,
    require_hashes: bool = False,
    verify_hashes: bool = False,
) -> TrajectoryManifest:
    """Load a trajectory-level split manifest and reject leakage-prone entries."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("trajectory manifest must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("trajectory manifest schema_version must be 1")
    entries = payload.get("trajectories")
    if not isinstance(entries, list) or not entries:
        raise ValueError("trajectory manifest must contain a non-empty trajectories list")

    records: list[TrajectoryRecord] = []
    ids: set[str] = set()
    paths: set[Path] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"trajectories[{index}] must be a JSON object")
        trajectory_id = _required_string(entry, "trajectory_id", index)
        robot = _required_string(entry, "robot", index)
        split = _required_string(entry, "split", index)
        point_bank_id = _required_string(entry, "point_bank_id", index)
        hdf5_value = _required_string(entry, "hdf5_path", index)
        eligible = entry.get("training_data_eligible")
        if type(eligible) is not bool:
            raise ValueError(
                f"trajectories[{index}].training_data_eligible must be a JSON boolean"
            )
        if split not in SPLITS:
            raise ValueError(
                f"trajectories[{index}].split must be one of {SPLITS}, got {split!r}"
            )

        hdf5_path = Path(hdf5_value)
        if not hdf5_path.is_absolute():
            hdf5_path = path.parent / hdf5_path
        hdf5_path = hdf5_path.resolve()
        if not hdf5_path.is_file():
            raise ValueError(f"trajectory HDF5 does not exist: {hdf5_path}")
        if trajectory_id in ids:
            raise ValueError(f"duplicate trajectory_id in manifest: {trajectory_id}")
        if hdf5_path in paths:
            raise ValueError(
                f"the same HDF5 trajectory appears more than once (split leakage): {hdf5_path}"
            )

        sha256 = entry.get("sha256")
        if require_hashes and sha256 is None:
            raise ValueError(f"trajectories[{index}].sha256 is required")
        if sha256 is not None:
            if not isinstance(sha256, str) or len(sha256) != 64:
                raise ValueError(f"trajectories[{index}].sha256 must be 64 hexadecimal characters")
            try:
                int(sha256, 16)
            except ValueError as error:
                raise ValueError(
                    f"trajectories[{index}].sha256 must be 64 hexadecimal characters"
                ) from error
            if verify_hashes:
                digest = hashlib.sha256()
                with hdf5_path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                actual_sha256 = digest.hexdigest()
                if actual_sha256 != sha256:
                    raise ValueError(
                        f"trajectory HDF5 SHA256 mismatch for {trajectory_id}: "
                        f"{actual_sha256} != {sha256}"
                    )

        accepted_transitions: tuple[int, ...] | None = None
        valid_frame_ids: tuple[int, ...] | None = None
        audit_sha256: str | None = None
        accepted_raw = entry.get("accepted_transition_indices")
        if accepted_raw is not None:
            if not isinstance(accepted_raw, list) or not accepted_raw:
                raise ValueError(
                    f"trajectories[{index}].accepted_transition_indices must be a non-empty list"
                )
            if any(type(frame) is not int or frame < 0 for frame in accepted_raw):
                raise ValueError(
                    "accepted_transition_indices must contain non-negative integers"
                )
            if len(set(accepted_raw)) != len(accepted_raw):
                raise ValueError("accepted_transition_indices must be unique")
            accepted_transitions = tuple(sorted(accepted_raw))
        valid_raw = entry.get("valid_frame_ids")
        if valid_raw is not None:
            if not isinstance(valid_raw, list) or len(valid_raw) < 2:
                raise ValueError(
                    f"trajectories[{index}].valid_frame_ids must be a list of >= 2 ints"
                )
            if any(type(frame) is not int or frame < 0 for frame in valid_raw):
                raise ValueError("valid_frame_ids must contain non-negative integers")
            if len(set(valid_raw)) != len(valid_raw):
                raise ValueError("valid_frame_ids must be unique")
            valid_frame_ids = tuple(sorted(valid_raw))

        quality = entry.get("transition_quality")
        if quality is not None:
            if not isinstance(quality, Mapping):
                raise ValueError(f"trajectories[{index}].transition_quality must be a JSON object")
            accepted = quality.get("accepted_transition_indices")
            audit_sha256 = quality.get("audit_sha256")
            if not isinstance(accepted, list) or not accepted:
                raise ValueError("transition_quality.accepted_transition_indices must be a non-empty list")
            if any(type(frame) is not int or frame < 0 for frame in accepted):
                raise ValueError("accepted_transition_indices must contain non-negative integers")
            if len(set(accepted)) != len(accepted):
                raise ValueError("accepted_transition_indices must be unique")
            if not isinstance(audit_sha256, str) or len(audit_sha256) != 64:
                raise ValueError("transition_quality.audit_sha256 must be a SHA256 hex digest")
            try:
                int(audit_sha256, 16)
            except ValueError as error:
                raise ValueError("transition_quality.audit_sha256 must be a SHA256 hex digest") from error
            audit_path_value = quality.get("audit_path")
            if not isinstance(audit_path_value, str) or not audit_path_value:
                raise ValueError("transition_quality.audit_path must be a non-empty string")
            audit_path = Path(audit_path_value)
            if not audit_path.is_absolute():
                audit_path = path.parent / audit_path
            audit_path = audit_path.resolve()
            if not audit_path.is_file():
                raise ValueError(f"transition quality audit does not exist: {audit_path}")
            digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            if digest != audit_sha256:
                raise ValueError(
                    f"transition quality audit SHA256 mismatch for {trajectory_id}: {digest} != {audit_sha256}"
                )
            quality_accepted = tuple(sorted(accepted))
            if accepted_transitions is not None and accepted_transitions != quality_accepted:
                raise ValueError(
                    f"trajectories[{index}] accepted_transition_indices disagrees with transition_quality"
                )
            accepted_transitions = quality_accepted

        ids.add(trajectory_id)
        paths.add(hdf5_path)
        speed_raw = entry.get("speed_m_per_s")
        speed_m_per_s: float | None
        if speed_raw is None:
            speed_m_per_s = None
        else:
            try:
                speed_m_per_s = float(speed_raw)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"trajectories[{index}].speed_m_per_s must be numeric, got {speed_raw!r}"
                ) from error
        records.append(
            TrajectoryRecord(
                trajectory_id=trajectory_id,
                robot=robot,
                split=split,
                hdf5_path=hdf5_path,
                training_data_eligible=eligible,
                point_bank_id=point_bank_id,
                sha256=sha256,
                accepted_transition_indices=accepted_transitions,
                valid_frame_ids=valid_frame_ids,
                transition_quality_audit_sha256=audit_sha256,
                speed_m_per_s=speed_m_per_s,
            )
        )

    present_splits = {record.split for record in records}
    if require_all_splits and present_splits != set(SPLITS):
        missing = sorted(set(SPLITS) - present_splits)
        raise ValueError(f"trajectory manifest is missing required splits: {missing}")
    if require_production_eligible:
        rejected = [record.trajectory_id for record in records if not record.training_data_eligible]
        if rejected:
            raise ValueError(f"manifest contains non-production trajectories: {rejected}")
    return TrajectoryManifest(path=path, records=tuple(records))


def _require_numeric(dataset: h5py.Dataset, name: str) -> None:
    if not np.issubdtype(dataset.dtype, np.number) and dataset.dtype != np.dtype(bool):
        raise ValueError(f"{name} must have a numeric or boolean dtype, got {dataset.dtype}")


def _require_finite(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")


def _as_binary_mask(dataset: h5py.Dataset, name: str, frame: int) -> np.ndarray:
    values = np.asarray(dataset[frame])
    if not np.logical_or(values == 0, values == 1).all():
        raise ValueError(f"{name}[{frame}] must contain only 0/1 values")
    return values.astype(bool, copy=False)


def validate_trajectory_hdf5(
    path: str | Path,
    *,
    expected_state_dim: int | None = None,
    validate_values: bool = True,
) -> TrajectoryShape:
    """Validate the V0 trajectory schema without loading the whole file at once."""
    path = Path(path)
    required = {
        "points_world_m",
        "query_xyz_normalized",
        "times",
        "state_raw",
        "valid_mask",
        "fluid_mask",
        "sdf",
        "nearest_normal",
        "wall_velocity",
        "boundary_feat",
        "static_fx",
        "loss_weight",
        "link_transform",
        "link_velocity",
    }
    with h5py.File(path, "r") as handle:
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise ValueError(f"{path} is missing required HDF5 datasets: {missing}")
        for name in required:
            if not isinstance(handle[name], h5py.Dataset):
                raise ValueError(f"{name} must be an HDF5 dataset")
            _require_numeric(handle[name], name)

        state_shape = handle["state_raw"].shape
        if len(state_shape) != 3:
            raise ValueError(f"state_raw must have shape [T,N,C], got {state_shape}")
        frames, points, state_dim = map(int, state_shape)
        if frames < 2 or points < 1:
            raise ValueError("state_raw requires at least two frames and one point")
        if state_dim not in STATE_DIMS:
            raise ValueError(f"state_raw channel count must be one of {STATE_DIMS}, got {state_dim}")
        if expected_state_dim is not None and state_dim != expected_state_dim:
            raise ValueError(f"state_raw channel count {state_dim} != expected {expected_state_dim}")

        link_shape = handle["link_transform"].shape
        if len(link_shape) != 4 or link_shape[0] != frames or link_shape[2:] != (4, 4):
            raise ValueError(
                f"link_transform must have shape [T,J,4,4] with T={frames}, got {link_shape}"
            )
        links = int(link_shape[1])
        if links < 1:
            raise ValueError("link_transform requires at least one link")

        expected_shapes = {
            "points_world_m": (points, 3),
            "query_xyz_normalized": (points, 3),
            "times": (frames,),
            "valid_mask": (frames, points),
            "fluid_mask": (frames, points),
            "sdf": (frames, points),
            "nearest_normal": (frames, points, 3),
            "wall_velocity": (frames - 1, points, 3),
            "boundary_feat": (frames - 1, points, 8),
            "static_fx": (frames, points, 11),
            "loss_weight": (points,),
            "link_velocity": (frames - 1, links, 6),
        }
        for name, expected in expected_shapes.items():
            actual = handle[name].shape
            if actual != expected:
                raise ValueError(f"{name} must have shape {expected}, got {actual}")

        times = np.asarray(handle["times"])
        _require_finite(times, "times")
        if not np.all(np.diff(times) > 0):
            raise ValueError("times must be strictly increasing")

        if validate_values:
            for name in ("points_world_m", "query_xyz_normalized", "loss_weight"):
                _require_finite(np.asarray(handle[name]), name)
            weights = np.asarray(handle["loss_weight"])
            if np.any(weights < 0) or not np.any(weights > 0):
                raise ValueError("loss_weight must be non-negative and contain a positive value")

            for frame in range(frames):
                valid = _as_binary_mask(handle["valid_mask"], "valid_mask", frame)
                fluid = _as_binary_mask(handle["fluid_mask"], "fluid_mask", frame)
                sdf = np.asarray(handle["sdf"][frame])
                _require_finite(sdf, f"sdf[{frame}]")
                if not np.array_equal(fluid, sdf > 0):
                    raise ValueError(f"fluid_mask[{frame}] must equal sdf[{frame}] > 0")
                normals = np.asarray(handle["nearest_normal"][frame])
                _require_finite(
                    normals,
                    f"nearest_normal[{frame}]",
                )
                normal_norm = np.linalg.norm(normals, axis=-1)
                if not np.allclose(normal_norm, 1.0, atol=1.0e-4, rtol=1.0e-4):
                    raise ValueError(f"nearest_normal[{frame}] must contain unit normals")
                static_fx = np.asarray(handle["static_fx"][frame])
                _require_finite(static_fx, f"static_fx[{frame}]")
                query_xyz = np.asarray(handle["query_xyz_normalized"])
                if not np.allclose(static_fx[:, :3], query_xyz, atol=1.0e-6, rtol=1.0e-6):
                    raise ValueError(f"static_fx[{frame}] normalized_xyz does not match query_xyz_normalized")
                if not np.allclose(static_fx[:, 3], sdf, atol=1.0e-6, rtol=1.0e-6):
                    raise ValueError(f"static_fx[{frame}] sdf does not match sdf[{frame}]")
                if not np.allclose(static_fx[:, 4:7], normals, atol=1.0e-6, rtol=1.0e-6):
                    raise ValueError(f"static_fx[{frame}] normal does not match nearest_normal[{frame}]")
                inflow_norm = np.linalg.norm(static_fx[:, 7:10], axis=-1)
                if not np.allclose(inflow_norm, 1.0, atol=1.0e-6, rtol=1.0e-6):
                    raise ValueError(f"static_fx[{frame}] inflow direction must be unit length")
                if np.any(static_fx[:, 10] < 0):
                    raise ValueError(f"static_fx[{frame}] normalized speed must be non-negative")
                state = np.asarray(handle["state_raw"][frame])
                active = valid & fluid
                _require_finite(state[active], f"state_raw[{frame}] on valid fluid points")
                if state_dim == 6 and np.any(state[active, 4:6] <= 0):
                    raise ValueError(
                        f"state_raw[{frame}] k and omega must be positive on valid fluid points"
                    )

                transforms = np.asarray(handle["link_transform"][frame])
                _require_finite(transforms, f"link_transform[{frame}]")
                expected_last_row = np.broadcast_to(
                    np.array([0.0, 0.0, 0.0, 1.0]),
                    transforms[:, 3, :].shape,
                )
                if not np.allclose(transforms[:, 3, :], expected_last_row, atol=1.0e-6):
                    raise ValueError(f"link_transform[{frame}] has invalid homogeneous rows")

            for transition in range(frames - 1):
                for name in ("wall_velocity", "boundary_feat", "link_velocity"):
                    _require_finite(
                        np.asarray(handle[name][transition]),
                        f"{name}[{transition}]",
                    )
                boundary = np.asarray(handle["boundary_feat"][transition])
                sdf_current = np.asarray(handle["sdf"][transition])
                sdf_next = np.asarray(handle["sdf"][transition + 1])
                wall_velocity = np.asarray(handle["wall_velocity"][transition])
                fluid_current = np.asarray(handle["fluid_mask"][transition]).astype(bool)
                fluid_next = np.asarray(handle["fluid_mask"][transition + 1]).astype(bool)
                if not np.allclose(boundary[:, 0], sdf_current, atol=1.0e-6, rtol=1.0e-6):
                    raise ValueError(f"boundary_feat[{transition}] sdf_current does not match sdf")
                if not np.allclose(
                    boundary[:, 1], sdf_next - sdf_current, atol=1.0e-6, rtol=1.0e-6
                ):
                    raise ValueError(f"boundary_feat[{transition}] sdf_delta does not match sdf change")
                if not np.allclose(boundary[:, 2:5], wall_velocity, atol=1.0e-6, rtol=1.0e-6):
                    raise ValueError(f"boundary_feat[{transition}] wall velocity does not match wall_velocity")
                if not np.array_equal(boundary[:, 5].astype(bool), fluid_current):
                    raise ValueError(f"boundary_feat[{transition}] current fluid mask does not match")
                if not np.array_equal(boundary[:, 6].astype(bool), fluid_next):
                    raise ValueError(f"boundary_feat[{transition}] next fluid mask does not match")
                if np.any(boundary[:, 7] <= 0) or not np.allclose(
                    boundary[:, 7], boundary[0, 7], atol=1.0e-8, rtol=1.0e-6
                ):
                    raise ValueError(f"boundary_feat[{transition}] normalized_dt must be positive and constant")

    return TrajectoryShape(frames=frames, points=points, state_dim=state_dim, links=links)


def nested_point_indices(
    total_points: int,
    subset_points: int | None,
    *,
    seed: int = 0,
    point_bank_id: str = "default",
) -> np.ndarray:
    """Return a stable permutation prefix, so larger subsets contain smaller ones."""
    total_points = int(total_points)
    subset_points = total_points if subset_points is None else int(subset_points)
    if total_points <= 0:
        raise ValueError("total_points must be positive")
    if subset_points <= 0 or subset_points > total_points:
        raise ValueError(
            f"subset_points must be in [1, {total_points}], got {subset_points}"
        )
    material = f"{int(seed)}:{point_bank_id}:{total_points}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(stable_seed).permutation(total_points)[:subset_points]


def resolve_point_indices(
    total_points: int,
    subset_points: int | None,
    *,
    seed: int = 0,
    point_bank_id: str = "default",
) -> np.ndarray:
    """Alias used by training/eval scripts for nested subset selection."""
    return nested_point_indices(
        total_points,
        subset_points,
        seed=seed,
        point_bank_id=point_bank_id,
    )


class DynamicStateNormalizer:
    """Train-only z-score normalizer over nondimensional dynamic CFD states."""

    def __init__(
        self,
        *,
        u_ref_m_per_s: float,
        l_ref_m: float,
        p_farfield_kinematic: float,
        mean: Sequence[float],
        std: Sequence[float],
        positive_floor: float = 1.0e-12,
    ) -> None:
        self.u_ref_m_per_s = float(u_ref_m_per_s)
        self.l_ref_m = float(l_ref_m)
        self.p_farfield_kinematic = float(p_farfield_kinematic)
        self.positive_floor = float(positive_floor)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        if self.u_ref_m_per_s <= 0 or self.l_ref_m <= 0 or self.positive_floor <= 0:
            raise ValueError("u_ref_m_per_s, l_ref_m, and positive_floor must be positive")
        if self.mean.ndim != 1 or self.mean.size not in STATE_DIMS:
            raise ValueError(f"mean must have {STATE_DIMS} channels")
        if self.std.shape != self.mean.shape or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("std must be finite, positive, and have the same shape as mean")
        if not np.isfinite(self.mean).all():
            raise ValueError("mean must be finite")

    @property
    def state_dim(self) -> int:
        return int(self.mean.size)

    def _resolve_u_ref(self, u_ref_m_per_s: float | None) -> float:
        u_ref = self.u_ref_m_per_s if u_ref_m_per_s is None else float(u_ref_m_per_s)
        if u_ref <= 0:
            raise ValueError("u_ref_m_per_s must be positive")
        return u_ref

    def _nondimensionalize(
        self, state_raw: np.ndarray, *, u_ref_m_per_s: float | None = None
    ) -> np.ndarray:
        state = np.asarray(state_raw, dtype=np.float64)
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"state has {state.shape[-1]} channels, expected {self.state_dim}")
        result = state.copy()
        u_ref = self._resolve_u_ref(u_ref_m_per_s)
        result[..., :3] /= u_ref
        result[..., 3] = (state[..., 3] - self.p_farfield_kinematic) / (u_ref**2)
        if self.state_dim == 6:
            k_nd = state[..., 4] / (u_ref**2)
            omega_nd = state[..., 5] * self.l_ref_m / u_ref
            result[..., 4] = np.log(np.maximum(k_nd, self.positive_floor))
            result[..., 5] = np.log(np.maximum(omega_nd, self.positive_floor))
        return result

    def _dimensionalize(
        self, state_nondimensional: np.ndarray, *, u_ref_m_per_s: float | None = None
    ) -> np.ndarray:
        state = np.asarray(state_nondimensional, dtype=np.float64)
        result = state.copy()
        u_ref = self._resolve_u_ref(u_ref_m_per_s)
        result[..., :3] *= u_ref
        result[..., 3] = state[..., 3] * (u_ref**2) + self.p_farfield_kinematic
        if self.state_dim == 6:
            result[..., 4] = np.exp(state[..., 4]) * (u_ref**2)
            result[..., 5] = np.exp(state[..., 5]) * u_ref / self.l_ref_m
        return result

    def encode(
        self, state_raw: np.ndarray, *, u_ref_m_per_s: float | None = None
    ) -> np.ndarray:
        """Z-score after nondim. Pass per-traj U_inf when μ/σ were fit that way (V0 cohort)."""
        encoded = (
            self._nondimensionalize(state_raw, u_ref_m_per_s=u_ref_m_per_s) - self.mean
        ) / self.std
        return encoded.astype(np.float32)

    def decode(
        self, state_normalized: np.ndarray, *, u_ref_m_per_s: float | None = None
    ) -> np.ndarray:
        normalized = np.asarray(state_normalized, dtype=np.float64)
        if normalized.shape[-1] != self.state_dim:
            raise ValueError(
                f"state has {normalized.shape[-1]} channels, expected {self.state_dim}"
            )
        nondimensional = normalized * self.std + self.mean
        return self._dimensionalize(
            nondimensional, u_ref_m_per_s=u_ref_m_per_s
        ).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "u_ref_m_per_s": self.u_ref_m_per_s,
            "l_ref_m": self.l_ref_m,
            "p_farfield_kinematic": self.p_farfield_kinematic,
            "positive_floor": self.positive_floor,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DynamicStateNormalizer":
        normalizer = cls(
            u_ref_m_per_s=payload["u_ref_m_per_s"],
            l_ref_m=payload["l_ref_m"],
            p_farfield_kinematic=payload["p_farfield_kinematic"],
            positive_floor=payload.get("positive_floor", 1.0e-12),
            mean=payload["mean"],
            std=payload["std"],
        )
        if "state_dim" in payload and int(payload["state_dim"]) != normalizer.state_dim:
            raise ValueError("normalizer state_dim does not match mean/std")
        return normalizer

    @classmethod
    def fit_from_manifest(
        cls,
        manifest: TrajectoryManifest | str | Path,
        *,
        u_ref_m_per_s: float,
        l_ref_m: float,
        p_farfield_kinematic: float = 0.0,
        positive_floor: float = 1.0e-12,
        subset_points: int | None = None,
        subset_seed: int = 0,
        min_std: float = 1.0e-8,
    ) -> "DynamicStateNormalizer":
        if not isinstance(manifest, TrajectoryManifest):
            manifest = load_trajectory_manifest(manifest)
        records = manifest.records_for_split("train", require_production_eligible=True)

        total = 0
        channel_sum: np.ndarray | None = None
        channel_square_sum: np.ndarray | None = None
        expected_state_dim: int | None = None
        probe: DynamicStateNormalizer | None = None
        bank_sizes: dict[str, int] = {}
        for record in records:
            shape = validate_trajectory_hdf5(
                record.hdf5_path,
                expected_state_dim=expected_state_dim,
                validate_values=False,
            )
            expected_state_dim = shape.state_dim
            if record.point_bank_id in bank_sizes and bank_sizes[record.point_bank_id] != shape.points:
                raise ValueError(f"point bank {record.point_bank_id!r} has inconsistent point counts")
            bank_sizes[record.point_bank_id] = shape.points
            indices = np.sort(
                nested_point_indices(
                    shape.points,
                    subset_points,
                    seed=subset_seed,
                    point_bank_id=record.point_bank_id,
                )
            )
            if probe is None:
                probe = cls(
                    u_ref_m_per_s=u_ref_m_per_s,
                    l_ref_m=l_ref_m,
                    p_farfield_kinematic=p_farfield_kinematic,
                    positive_floor=positive_floor,
                    mean=np.zeros(shape.state_dim),
                    std=np.ones(shape.state_dim),
                )
                channel_sum = np.zeros(shape.state_dim, dtype=np.float64)
                channel_square_sum = np.zeros(shape.state_dim, dtype=np.float64)

            # Match V0/Temporal/MGN: nondimensionalize each traj with its own U_inf when known.
            traj_u_ref = (
                float(record.speed_m_per_s)
                if record.speed_m_per_s is not None
                else float(u_ref_m_per_s)
            )
            accepted = record.accepted_transition_indices
            if accepted is None and record.valid_frame_ids is None:
                frame_indices = range(shape.frames)
            else:
                timeline = resolve_keyframe_timeline(
                    n_frames=shape.frames,
                    valid_frame_ids=record.valid_frame_ids,
                    accepted_transition_indices=accepted,
                )
                frame_indices = timeline
            with h5py.File(record.hdf5_path, "r") as handle:
                full_subset = indices.size == shape.points and np.array_equal(
                    indices, np.arange(shape.points)
                )
                for frame in frame_indices:
                    if full_subset:
                        valid = np.asarray(handle["valid_mask"][frame]).astype(bool)
                        fluid = np.asarray(handle["fluid_mask"][frame]).astype(bool)
                        raw_state = np.asarray(handle["state_raw"][frame])
                    else:
                        # Shared-filesystem HDF5 fancy indexing issues tens of thousands
                        # of noncontiguous reads. Read one compact source frame instead.
                        valid = np.asarray(handle["valid_mask"][frame])[indices].astype(bool)
                        fluid = np.asarray(handle["fluid_mask"][frame])[indices].astype(bool)
                        raw_state = np.asarray(handle["state_raw"][frame])[indices]
                    active = valid & fluid
                    if not np.any(active):
                        continue
                    state = raw_state[active]
                    transformed = probe._nondimensionalize(state, u_ref_m_per_s=traj_u_ref)
                    if not np.isfinite(transformed).all():
                        raise ValueError(
                            f"non-finite transformed state in training trajectory {record.trajectory_id}"
                        )
                    channel_sum += transformed.sum(axis=0)
                    channel_square_sum += np.square(transformed).sum(axis=0)
                    total += int(transformed.shape[0])

        if total == 0 or probe is None or channel_sum is None or channel_square_sum is None:
            raise ValueError("training trajectories contain no valid fluid state samples")
        mean = channel_sum / total
        variance = np.maximum(channel_square_sum / total - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(variance), float(min_std))
        return cls(
            u_ref_m_per_s=u_ref_m_per_s,
            l_ref_m=l_ref_m,
            p_farfield_kinematic=p_farfield_kinematic,
            positive_floor=positive_floor,
            mean=mean,
            std=std,
        )


def resolve_keyframe_timeline(
    *,
    n_frames: int,
    valid_frame_ids: Sequence[int] | None = None,
    accepted_transition_indices: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Ordered input frames. Supervision is always input ``t → t+1`` on this list.

    Prefer ``valid_frame_ids``. Gaps in HDF5 indices are fine — we read real states.
    Dense accepted ``0..n-2`` → ``0..n-1``. Sparse starts without ``valid_frame_ids``
    append the last source frame as the final input.
    """
    if valid_frame_ids is not None:
        ids = tuple(int(x) for x in valid_frame_ids)
        if len(ids) < 2:
            raise ValueError("valid_frame_ids must contain at least 2 frames")
        return ids
    if accepted_transition_indices is None:
        if n_frames < 2:
            raise ValueError(f"n_frames must be >= 2, got {n_frames}")
        return tuple(range(n_frames))
    acc = tuple(int(x) for x in accepted_transition_indices)
    if not acc:
        raise ValueError("accepted_transition_indices must be non-empty")
    if len(acc) == n_frames - 1 and acc == tuple(range(n_frames - 1)):
        return tuple(range(n_frames))
    if acc[-1] != n_frames - 1:
        return acc + (n_frames - 1,)
    return acc


def _take_points(
    dataset: h5py.Dataset,
    point_indices: np.ndarray,
    *,
    frame: int | None = None,
) -> np.ndarray:
    if point_indices.size == dataset.shape[0 if frame is None else 1] and np.array_equal(
        point_indices, np.arange(point_indices.size)
    ):
        return np.asarray(dataset[:] if frame is None else dataset[frame])
    order = np.argsort(point_indices)
    sorted_indices = point_indices[order]
    if frame is None:
        sorted_values = np.asarray(dataset[sorted_indices])
    else:
        sorted_values = np.asarray(dataset[frame, sorted_indices])
    return sorted_values[np.argsort(order)]


class DynamicCFDOneStepDataset(Dataset):
    """One-step samples: input ``t`` → supervise input ``t+1`` on the input timeline.

    Timeline = ``valid_frame_ids`` when present (else dense / inferred from accepted).
    Sparse CFD gaps (10/20/50) are fine — we read the real state at each input index.
    """

    def __init__(
        self,
        manifest: TrajectoryManifest | str | Path,
        split: str,
        normalizer: DynamicStateNormalizer,
        *,
        subset_points: int | None = None,
        subset_seed: int = 0,
        history: int = 1,
        include_surface: bool = False,
    ) -> None:
        if not isinstance(manifest, TrajectoryManifest):
            manifest = load_trajectory_manifest(manifest)
        if int(history) < 1:
            raise ValueError(f"history must be >= 1, got {history}")
        self.manifest = manifest
        self.split = split
        self.normalizer = normalizer
        self.history = int(history)
        self.include_surface = bool(include_surface)
        self.records = manifest.records_for_split(split, require_production_eligible=True)
        self._shapes: list[TrajectoryShape] = []
        self._point_indices: list[np.ndarray] = []
        # Per-record ordered input timeline (HDF5 frame ids).
        self._input_timelines: list[tuple[int, ...]] = []
        # Samples are (record_index, seq_pos) with target at seq_pos+1.
        self._samples: list[tuple[int, int]] = []
        self._handles: dict[Path, h5py.File] = {}

        bank_sizes: dict[str, int] = {}
        for record_index, record in enumerate(self.records):
            shape = validate_trajectory_hdf5(
                record.hdf5_path,
                expected_state_dim=normalizer.state_dim,
                validate_values=False,
            )
            if record.point_bank_id in bank_sizes and bank_sizes[record.point_bank_id] != shape.points:
                raise ValueError(f"point bank {record.point_bank_id!r} has inconsistent point counts")
            bank_sizes[record.point_bank_id] = shape.points
            timeline = resolve_keyframe_timeline(
                n_frames=shape.frames,
                valid_frame_ids=record.valid_frame_ids,
                accepted_transition_indices=record.accepted_transition_indices,
            )
            if timeline[-1] >= shape.frames:
                raise ValueError(
                    f"trajectory {record.trajectory_id} input frame {timeline[-1]} "
                    f"out of range for {shape.frames} frames"
                )
            if len(timeline) < 2:
                raise ValueError(
                    f"trajectory {record.trajectory_id} needs >= 2 input frames"
                )
            self._shapes.append(shape)
            self._point_indices.append(
                nested_point_indices(
                    shape.points,
                    subset_points,
                    seed=subset_seed,
                    point_bank_id=record.point_bank_id,
                )
            )
            self._input_timelines.append(timeline)
            self._samples.extend(
                (record_index, seq_pos) for seq_pos in range(len(timeline) - 1)
            )

    def __len__(self) -> int:
        return len(self._samples)

    def point_indices(self, trajectory_id: str) -> np.ndarray:
        for record, indices in zip(self.records, self._point_indices):
            if record.trajectory_id == trajectory_id:
                return indices.copy()
        raise KeyError(f"unknown trajectory_id in {self.split!r} split: {trajectory_id}")

    def sample_indices_for_trajectory(self, trajectory_id: str) -> tuple[int, ...]:
        record_index = next(
            (
                index
                for index, record in enumerate(self.records)
                if record.trajectory_id == trajectory_id
            ),
            None,
        )
        if record_index is None:
            raise KeyError(f"unknown trajectory_id in {self.split!r} split: {trajectory_id}")
        return tuple(
            sample_index
            for sample_index, (candidate_record, _) in enumerate(self._samples)
            if candidate_record == record_index
        )

    def _get_handle(self, path: Path) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, seq_pos = self._samples[index]
        record = self.records[record_index]
        indices = self._point_indices[record_index]
        handle = self._get_handle(record.hdf5_path)
        timeline = self._input_timelines[record_index]
        frame = int(timeline[seq_pos])
        target_frame = int(timeline[seq_pos + 1])

        query_xyz = _take_points(handle["query_xyz_normalized"], indices)
        static_fx = _take_points(handle["static_fx"], indices, frame=target_frame)
        boundary_feat = _take_points(handle["boundary_feat"], indices, frame=frame)
        raw_target = _take_points(handle["state_raw"], indices, frame=target_frame)
        valid_target = _take_points(handle["valid_mask"], indices, frame=target_frame).astype(bool)
        fluid_target = _take_points(handle["fluid_mask"], indices, frame=target_frame).astype(bool)
        base_weight = _take_points(handle["loss_weight"], indices)

        # Per-traj U_inf when available — required for frozen V0 μ/σ (fit with same protocol).
        traj_u_ref = record.speed_m_per_s
        history_states: list[np.ndarray] = []
        history_masks: list[np.ndarray] = []
        for offset in range(self.history - 1, -1, -1):
            # History on the input timeline (not HDF5-adjacent).
            hist_seq = max(0, seq_pos - offset)
            hist_frame = int(timeline[hist_seq])
            raw_prev = _take_points(handle["state_raw"], indices, frame=hist_frame)
            valid_prev = _take_points(handle["valid_mask"], indices, frame=hist_frame).astype(bool)
            fluid_prev = _take_points(handle["fluid_mask"], indices, frame=hist_frame).astype(bool)
            mask = valid_prev & fluid_prev
            encoded = np.zeros((indices.size, self.normalizer.state_dim), dtype=np.float32)
            if np.any(mask):
                encoded[mask] = self.normalizer.encode(
                    raw_prev[mask], u_ref_m_per_s=traj_u_ref
                )
            history_states.append(encoded)
            history_masks.append(mask)

        previous_mask = history_masks[-1]
        loss_mask = valid_target & fluid_target
        target_state = np.zeros((indices.size, self.normalizer.state_dim), dtype=np.float32)
        if np.any(loss_mask):
            target_state[loss_mask] = self.normalizer.encode(
                raw_target[loss_mask], u_ref_m_per_s=traj_u_ref
            )
        loss_weight = np.where(loss_mask, base_weight, 0.0).astype(np.float32)

        if self.history == 1:
            previous_state = history_states[0]
        else:
            previous_state = np.stack(history_states, axis=0)

        times = handle["times"]
        result = {
            "query_xyz": torch.from_numpy(np.ascontiguousarray(query_xyz, dtype=np.float32)),
            "static_fx": torch.from_numpy(np.ascontiguousarray(static_fx, dtype=np.float32)),
            "previous_state": torch.from_numpy(np.ascontiguousarray(previous_state)),
            "boundary_feat": torch.from_numpy(
                np.ascontiguousarray(boundary_feat, dtype=np.float32)
            ),
            "target_state": torch.from_numpy(target_state),
            "previous_mask": torch.from_numpy(
                np.ascontiguousarray(previous_mask, dtype=bool)
            ),
            "loss_mask": torch.from_numpy(np.ascontiguousarray(loss_mask, dtype=bool)),
            "loss_weight": torch.from_numpy(loss_weight),
            "trajectory_id": record.trajectory_id,
            "u_ref_m_per_s": float(
                traj_u_ref if traj_u_ref is not None else self.normalizer.u_ref_m_per_s
            ),
            "sample_index": torch.tensor(index, dtype=torch.int64),
            "time_index": torch.tensor(frame, dtype=torch.int64),
            "time_next_index": torch.tensor(target_frame, dtype=torch.int64),
            "time_current": torch.tensor(float(times[frame]), dtype=torch.float64),
            "time_next": torch.tensor(float(times[target_frame]), dtype=torch.float64),
        }
        if self.include_surface:
            required = ("surface_xyz_normalized", "surface_normal_world", "surface_state_raw", "surface_valid_mask")
            missing = [name for name in required if name not in handle]
            if missing:
                raise ValueError(f"surface-enabled dataset is missing fields: {missing}")
            surface_xyz = np.asarray(handle["surface_xyz_normalized"][frame], dtype=np.float32)
            surface_normal = np.asarray(handle["surface_normal_world"][frame], dtype=np.float32)
            surface_raw = np.asarray(handle["surface_state_raw"][frame], dtype=np.float32)
            surface_valid = np.asarray(handle["surface_valid_mask"][frame], dtype=bool)
            if surface_xyz.ndim != 2 or surface_xyz.shape[-1] != 3:
                raise ValueError(f"surface_xyz_normalized must be [Ns,3], got {surface_xyz.shape}")
            if surface_normal.shape != surface_xyz.shape or surface_raw.shape != (surface_xyz.shape[0], self.normalizer.state_dim):
                raise ValueError(f"surface fields have inconsistent shapes: xyz={surface_xyz.shape}, normal={surface_normal.shape}, state={surface_raw.shape}")
            if surface_valid.shape != (surface_xyz.shape[0],):
                raise ValueError(f"surface_valid_mask must be [Ns], got {surface_valid.shape}")
            if not np.any(surface_valid):
                raise ValueError(f"surface frame {frame} has no valid points")
            if not (np.isfinite(surface_xyz[surface_valid]).all() and np.isfinite(surface_normal[surface_valid]).all() and np.isfinite(surface_raw[surface_valid]).all()):
                raise ValueError(f"non-finite values in valid surface points for frame {frame}")
            surface_state = np.zeros_like(surface_raw, dtype=np.float32)
            surface_state[surface_valid] = self.normalizer.encode(surface_raw[surface_valid], u_ref_m_per_s=traj_u_ref)
            result.update({
                "surface_xyz_normalized": torch.from_numpy(np.ascontiguousarray(surface_xyz)),
                "surface_normal_current": torch.from_numpy(np.ascontiguousarray(surface_normal)),
                "surface_state_current": torch.from_numpy(np.ascontiguousarray(surface_state)),
                "surface_valid_mask": torch.from_numpy(np.ascontiguousarray(surface_valid)),
            })
        return result

    def close(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            handle.close()
        if hasattr(self, "_handles"):
            self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        self.close()
