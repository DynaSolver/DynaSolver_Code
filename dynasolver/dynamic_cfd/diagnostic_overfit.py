from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import torch

from data_provider.dynamic_cfd import DynamicStateNormalizer, validate_trajectory_hdf5


def select_stratified_point_indices(
    cell_centres: np.ndarray,
    distance_to_robot: np.ndarray,
    robot_bounds: np.ndarray,
    point_count: int,
    *,
    seed: int,
) -> np.ndarray:
    """Select a deterministic near-wall/wake/domain point bank from CFD cells."""
    centres = np.asarray(cell_centres, dtype=np.float64)
    distances = np.asarray(distance_to_robot, dtype=np.float64)
    bounds = np.asarray(robot_bounds, dtype=np.float64)
    if centres.ndim != 2 or centres.shape[1] != 3:
        raise ValueError("cell_centres must have shape [M,3]")
    if distances.shape != (centres.shape[0],) or not np.isfinite(distances).all():
        raise ValueError("distance_to_robot must be finite with shape [M]")
    if bounds.shape != (3, 2) or not np.all(bounds[:, 1] > bounds[:, 0]):
        raise ValueError("robot_bounds must have shape [3,2] with positive extents")
    point_count = int(point_count)
    if point_count <= 0 or point_count > centres.shape[0]:
        raise ValueError("point_count must be between 1 and the number of CFD cells")

    rng = np.random.default_rng(seed)
    near_count = min(point_count // 2, centres.shape[0])
    selected = list(np.argsort(distances, kind="stable")[:near_count])
    chosen = np.zeros(centres.shape[0], dtype=bool)
    chosen[selected] = True

    extent = bounds[:, 1] - bounds[:, 0]
    wake_candidates = np.flatnonzero(
        (~chosen)
        & (centres[:, 0] >= bounds[0, 1])
        & (centres[:, 0] <= bounds[0, 1] + max(3.0, 2.0 * extent[2]))
        & (np.abs(centres[:, 1] - bounds[1].mean()) <= max(1.0, 2.0 * extent[1]))
        & (centres[:, 2] >= bounds[2, 0] - 0.5 * extent[2])
        & (centres[:, 2] <= bounds[2, 1] + 0.5 * extent[2])
    )
    wake_count = min(point_count * 3 // 10, wake_candidates.size)
    if wake_count:
        wake = rng.choice(wake_candidates, size=wake_count, replace=False)
        selected.extend(wake.tolist())
        chosen[wake] = True

    remaining_count = point_count - len(selected)
    if remaining_count:
        remaining = np.flatnonzero(~chosen)
        fill = rng.choice(remaining, size=remaining_count, replace=False)
        selected.extend(fill.tolist())

    result = np.asarray(selected, dtype=np.int64)
    if result.size != point_count or np.unique(result).size != point_count:
        raise RuntimeError("stratified point selection did not produce unique requested points")
    return result


def select_nested_stratified_point_indices(
    cell_centres: np.ndarray,
    distance_to_robot: np.ndarray,
    robot_bounds: np.ndarray,
    master_count: int,
    subset_count: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 50/30/20 master bank and a composition-matched nested subset."""
    centres = np.asarray(cell_centres, dtype=np.float64)
    distances = np.asarray(distance_to_robot, dtype=np.float64)
    bounds = np.asarray(robot_bounds, dtype=np.float64)
    master_count, subset_count = int(master_count), int(subset_count)
    if not 0 < subset_count <= master_count <= centres.shape[0]:
        raise ValueError("require 0 < subset_count <= master_count <= cell count")
    rng = np.random.default_rng(seed)
    extent = bounds[:, 1] - bounds[:, 0]
    near_order = np.argsort(distances, kind="stable")
    near_master = master_count // 2
    near = near_order[:near_master]
    used = np.zeros(centres.shape[0], dtype=bool)
    used[near] = True
    wake_pool = np.flatnonzero(
        (~used)
        & (centres[:, 0] >= bounds[0, 1])
        & (centres[:, 0] <= bounds[0, 1] + max(3.0, 2.0 * extent[2]))
        & (np.abs(centres[:, 1] - bounds[1].mean()) <= max(1.0, 2.0 * extent[1]))
        & (centres[:, 2] >= bounds[2, 0] - 0.5 * extent[2])
        & (centres[:, 2] <= bounds[2, 1] + 0.5 * extent[2])
    )
    rng.shuffle(wake_pool)
    wake_master = min(master_count * 3 // 10, wake_pool.size)
    wake = wake_pool[:wake_master]
    used[wake] = True
    far_pool = np.flatnonzero(~used)
    rng.shuffle(far_pool)
    far_count = master_count - near.size - wake.size
    if far_pool.size < far_count:
        raise ValueError("insufficient remaining cells for master point bank")
    far = far_pool[:far_count]
    master = np.concatenate((near, wake, far)).astype(np.int64)

    subset_near = min(subset_count // 2, near.size)
    subset_wake = min(subset_count * 3 // 10, wake.size)
    subset_far = subset_count - subset_near - subset_wake
    if subset_far > far.size:
        deficit = subset_far - far.size
        extra_wake = min(deficit, wake.size - subset_wake)
        subset_wake += extra_wake
        subset_far -= extra_wake
    if subset_far > far.size:
        deficit = subset_far - far.size
        subset_near += deficit
        subset_far -= deficit
    subset_positions = np.concatenate(
        (
            np.arange(subset_near),
            near.size + np.arange(subset_wake),
            near.size + wake.size + np.arange(subset_far),
        )
    ).astype(np.int64)
    if master.size != master_count or subset_positions.size != subset_count:
        raise RuntimeError("nested point bank size mismatch")
    if np.unique(master).size != master_count or np.unique(subset_positions).size != subset_count:
        raise RuntimeError("nested point bank contains duplicate indices")
    return master, subset_positions


def finite_difference_link_velocity(
    transforms: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Return world-frame linear/angular velocities as [T-1,J,6]."""
    from scipy.spatial.transform import Rotation

    transforms = np.asarray(transforms, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if transforms.ndim != 4 or transforms.shape[2:] != (4, 4):
        raise ValueError("transforms must have shape [T,J,4,4]")
    if times.shape != (transforms.shape[0],) or not np.all(np.diff(times) > 0):
        raise ValueError("times must be strictly increasing with shape [T]")
    dt = np.diff(times)
    linear = np.diff(transforms[:, :, :3, 3], axis=0) / dt[:, None, None]
    angular = np.empty_like(linear)
    for frame in range(transforms.shape[0] - 1):
        relative = transforms[frame + 1, :, :3, :3] @ np.swapaxes(
            transforms[frame, :, :3, :3], -1, -2
        )
        angular[frame] = Rotation.from_matrix(relative).as_rotvec() / dt[frame]
    return np.concatenate((linear, angular), axis=-1).astype(np.float32)


def fit_diagnostic_normalizer(
    hdf5_path: str,
    *,
    u_ref_m_per_s: float,
    l_ref_m: float,
    p_farfield_kinematic: float = 0.0,
    min_std: float = 1.0e-6,
    max_speed_m_per_s: float | None = None,
) -> DynamicStateNormalizer:
    """Fit normalization to a diagnostic trajectory without bypassing production APIs."""
    shape = validate_trajectory_hdf5(hdf5_path)
    probe = DynamicStateNormalizer(
        u_ref_m_per_s=u_ref_m_per_s,
        l_ref_m=l_ref_m,
        p_farfield_kinematic=p_farfield_kinematic,
        mean=np.zeros(shape.state_dim),
        std=np.ones(shape.state_dim),
    )
    total = 0
    channel_sum = np.zeros(shape.state_dim, dtype=np.float64)
    channel_square_sum = np.zeros(shape.state_dim, dtype=np.float64)
    with h5py.File(hdf5_path, "r") as handle:
        for frame in range(shape.frames):
            active = np.asarray(handle["valid_mask"][frame]).astype(bool)
            active &= np.asarray(handle["fluid_mask"][frame]).astype(bool)
            raw = np.asarray(handle["state_raw"][frame])[active]
            if max_speed_m_per_s is not None:
                raw = raw[np.linalg.norm(raw[:, :3], axis=1) <= max_speed_m_per_s]
            if raw.size:
                transformed = probe._nondimensionalize(raw)
                channel_sum += transformed.sum(axis=0)
                channel_square_sum += np.square(transformed).sum(axis=0)
                total += transformed.shape[0]
    if total == 0:
        raise ValueError("diagnostic trajectory contains no valid fluid states")
    mean = channel_sum / total
    variance = np.maximum(channel_square_sum / total - np.square(mean), 0.0)
    return DynamicStateNormalizer(
        u_ref_m_per_s=u_ref_m_per_s,
        l_ref_m=l_ref_m,
        p_farfield_kinematic=p_farfield_kinematic,
        mean=mean,
        std=np.maximum(np.sqrt(variance), min_std),
    )


def load_diagnostic_transition(
    hdf5_path: str,
    transition_index: int,
    normalizer: DynamicStateNormalizer,
    *,
    point_count: int | None = None,
    subset_seed: int = 0,
) -> dict[str, Any]:
    """Load one real transition while requiring explicit diagnostic provenance."""
    shape = validate_trajectory_hdf5(hdf5_path)
    transition_index = int(transition_index)
    if transition_index < 0:
        transition_index += shape.frames - 1
    if transition_index < 0 or transition_index >= shape.frames - 1:
        raise ValueError(f"transition_index must be in [0,{shape.frames - 2}]")
    point_count = shape.points if point_count is None else int(point_count)
    if point_count <= 0 or point_count > shape.points:
        raise ValueError(f"point_count must be in [1,{shape.points}]")
    rng = np.random.default_rng(subset_seed)
    indices = np.sort(rng.choice(shape.points, size=point_count, replace=False))

    with h5py.File(hdf5_path, "r") as handle:
        if not bool(handle.attrs.get("diagnostic_only", False)):
            raise ValueError("one-batch diagnostic runner requires diagnostic_only=true")
        if bool(handle.attrs.get("training_data_eligible", True)):
            raise ValueError("diagnostic HDF5 must not be marked training_data_eligible")
        current = transition_index
        target = current + 1
        valid_previous = np.asarray(handle["valid_mask"][current, indices]).astype(bool)
        fluid_previous = np.asarray(handle["fluid_mask"][current, indices]).astype(bool)
        valid_target = np.asarray(handle["valid_mask"][target, indices]).astype(bool)
        fluid_target = np.asarray(handle["fluid_mask"][target, indices]).astype(bool)
        previous_mask = valid_previous & fluid_previous
        loss_mask = valid_target & fluid_target
        if not np.any(loss_mask):
            raise ValueError("selected transition has no supervised fluid points")

        previous = np.zeros((point_count, shape.state_dim), dtype=np.float32)
        target_state = np.zeros_like(previous)
        raw_previous = np.asarray(handle["state_raw"][current, indices])
        raw_target = np.asarray(handle["state_raw"][target, indices])
        previous[previous_mask] = normalizer.encode(raw_previous[previous_mask])
        target_state[loss_mask] = normalizer.encode(raw_target[loss_mask])
        base_weight = np.asarray(handle["loss_weight"][indices], dtype=np.float32)
        weight = np.where(loss_mask, base_weight, 0.0).astype(np.float32)

        tensor = lambda value, dtype=np.float32: torch.from_numpy(
            np.ascontiguousarray(value, dtype=dtype)
        ).unsqueeze(0)
        return {
            "query_xyz": tensor(handle["query_xyz_normalized"][indices]),
            "static_fx": tensor(handle["static_fx"][target, indices]),
            "previous_state": tensor(previous),
            "boundary_feat": tensor(handle["boundary_feat"][current, indices]),
            "target_state": tensor(target_state),
            "previous_mask": tensor(previous_mask, bool),
            "loss_mask": tensor(loss_mask, bool),
            "loss_weight": tensor(weight),
            "raw_target": raw_target.astype(np.float32),
            "indices": indices,
            "time_current": float(handle["times"][current]),
            "time_next": float(handle["times"][target]),
        }


def load_diagnostic_sequence(
    hdf5_path: str,
    normalizer: DynamicStateNormalizer,
    *,
    point_count: int,
    max_speed_m_per_s: float | None = None,
) -> dict[str, Any]:
    """Load an entire diagnostic trajectory using an explicit nested point set."""
    shape = validate_trajectory_hdf5(hdf5_path)
    point_count = int(point_count)
    with h5py.File(hdf5_path, "r") as handle:
        if not bool(handle.attrs.get("diagnostic_only", False)):
            raise ValueError("sequence overfit requires diagnostic_only=true")
        if bool(handle.attrs.get("training_data_eligible", True)):
            raise ValueError("diagnostic HDF5 must not be marked training_data_eligible")
        if point_count == shape.points:
            indices = np.arange(shape.points, dtype=np.int64)
        else:
            name = f"nested_subset_indices_{point_count}"
            if name not in handle:
                raise ValueError(f"HDF5 has no explicit nested point subset {name}")
            indices = np.asarray(handle[name], dtype=np.int64)
            if indices.shape != (point_count,) or np.unique(indices).size != point_count:
                raise ValueError(f"invalid explicit nested point subset {name}")
            if np.any(indices < 0) or np.any(indices >= shape.points):
                raise ValueError(f"nested point subset {name} is out of bounds")
        indices = np.sort(indices)

        def take_time(name: str) -> np.ndarray:
            dataset = handle[name]
            if point_count == shape.points:
                return np.asarray(dataset)
            return np.stack(
                [np.asarray(dataset[frame])[indices] for frame in range(dataset.shape[0])],
                axis=0,
            )

        valid = take_time("valid_mask").astype(bool)
        fluid = take_time("fluid_mask").astype(bool)
        active = valid & fluid
        raw = take_time("state_raw")
        if max_speed_m_per_s is not None:
            active &= np.linalg.norm(raw[:, :, :3], axis=2) <= max_speed_m_per_s
        state = np.zeros_like(raw, dtype=np.float32)
        for frame in range(shape.frames):
            state[frame, active[frame]] = normalizer.encode(raw[frame, active[frame]])
        return {
            "query_xyz": np.asarray(handle["query_xyz_normalized"][indices], dtype=np.float32),
            "static_fx": np.asarray(take_time("static_fx"), dtype=np.float32),
            "boundary_feat": np.asarray(take_time("boundary_feat"), dtype=np.float32),
            "state": state,
            "active_mask": active,
            "loss_weight": np.asarray(handle["loss_weight"][indices], dtype=np.float32),
            "times": np.asarray(handle["times"], dtype=np.float64),
            "indices": indices,
            "hdf5_path": str(hdf5_path),
        }
