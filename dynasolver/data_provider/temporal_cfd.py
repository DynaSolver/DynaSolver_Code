from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data_provider.dynamic_cfd import (
    DynamicStateNormalizer,
    TrajectoryManifest,
    TrajectoryShape,
    _take_points,
    load_trajectory_manifest,
    nested_point_indices,
    resolve_keyframe_timeline,
    validate_trajectory_hdf5,
)
from data_provider.frame_cache import FrameLRUCache


def causal_window_frame_ids(current_t: int, history_frames: int) -> list[int]:
    """Growing-prefix then slide — no left-pad.

    length = min(current_t + 1, history_frames + 1):

      t=0 → [0]
      t=1 → [0, 1]
      t=2 → [0, 1, 2]
      t=H → [0, 1, ..., H]
      t>H → [t-H, ..., t]   (slide)

    Indices are positions on the **input timeline**. Map to HDF5 ids via
    ``causal_window_on_sequence`` when the timeline is sparse.
    """
    if int(current_t) < 0:
        raise ValueError(f"current_t must be >= 0, got {current_t}")
    if int(history_frames) < 0:
        raise ValueError(f"history_frames must be >= 0, got {history_frames}")
    window = int(history_frames) + 1
    start = max(0, int(current_t) - int(history_frames))
    ids = list(range(start, int(current_t) + 1))
    if len(ids) > window:
        raise RuntimeError(f"window longer than max: {ids} vs max {window}")
    return ids


def causal_window_on_sequence(
    seq_pos: int,
    history_frames: int,
    sequence_frames: Sequence[int],
) -> list[int]:
    """Causal window in input-sequence space, returned as dataset frame ids.

    Example (uniform_20 inputs ``[0,10,21,31,...]``, ``history_frames=3``):

      seq_pos=3 → local [0,1,2,3] → frames [0,10,21,31]
    """
    if not sequence_frames:
        raise ValueError("sequence_frames must be non-empty")
    n = len(sequence_frames)
    if not (0 <= int(seq_pos) < n):
        raise IndexError(f"seq_pos={seq_pos} out of range for sequence length {n}")
    local = causal_window_frame_ids(int(seq_pos), history_frames)
    return [int(sequence_frames[i]) for i in local]


def advance_causal_window_tensors(
    state: torch.Tensor,
    static: torch.Tensor,
    boundary: torch.Tensor,
    next_state: torch.Tensor,
    next_static: torch.Tensor,
    next_boundary: torch.Tensor,
    *,
    max_window: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append next frame; slide if length would exceed ``max_window``."""
    state = torch.cat([state, next_state.unsqueeze(1)], dim=1)
    static = torch.cat([static, next_static.unsqueeze(1)], dim=1)
    boundary = torch.cat([boundary, next_boundary.unsqueeze(1)], dim=1)
    if state.shape[1] > max_window:
        state = state[:, -max_window:]
        static = static[:, -max_window:]
        boundary = boundary[:, -max_window:]
    return state, static, boundary


class SameWindowLengthBatchSampler(Sampler[list[int]]):
    """Yield batches whose samples share the same causal window length."""

    def __init__(
        self,
        dataset: "TemporalCFDWindowDataset",
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0
        by_len: dict[int, list[int]] = {}
        for index in range(len(dataset)):
            _record_index, seq_pos = dataset._samples[index]
            length = len(causal_window_frame_ids(seq_pos, dataset.history_frames))
            by_len.setdefault(length, []).append(index)
        self._by_len = by_len
        self.num_batches = 0
        for idxs in by_len.values():
            n = len(idxs) // self.batch_size
            if not self.drop_last and (len(idxs) % self.batch_size):
                n += 1
            self.num_batches += n

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        batches: list[list[int]] = []
        for length in sorted(self._by_len):
            idxs = list(self._by_len[length])
            if self.shuffle:
                rng.shuffle(idxs)
            for start in range(0, len(idxs), self.batch_size):
                chunk = idxs[start : start + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                if chunk:
                    batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


class TemporalCFDWindowDataset(Dataset):
    """
    Sliding short windows for Temporal Transolver on the **keyframe input axis**.

    Timeline = ``valid_frame_ids`` when present (else dense / inferred from accepted).
    At input index ``i`` (HDF5 frame ``F[i]``):
      growing-prefix then slide ending at ``i`` (``i=0 → [0]``, never left-pad);
      predict input ``t+1`` = ``F[i+1]`` (e.g. @10 → @21).
    """

    def __init__(
        self,
        manifest: TrajectoryManifest | str | Path,
        split: str,
        normalizer: DynamicStateNormalizer,
        *,
        history_frames: int = 4,
        subset_points: int | None = None,
        subset_seed: int = 0,
        unroll_steps: int = 1,
        sample_stride: int = 1,
        frame_cache_max_bytes: int = 0,
        fixed_window_frames: int | None = None,
        start_singleton: bool = False,
    ) -> None:
        if history_frames < 0:
            raise ValueError("history_frames must be >= 0")
        if int(unroll_steps) < 1:
            raise ValueError("unroll_steps must be >= 1")
        if int(sample_stride) < 1:
            raise ValueError("sample_stride must be >= 1")
        if int(frame_cache_max_bytes) < 0:
            raise ValueError("frame_cache_max_bytes must be >= 0")
        if fixed_window_frames is not None and int(fixed_window_frames) < 1:
            raise ValueError("fixed_window_frames must be >= 1")
        if bool(start_singleton) and fixed_window_frames is not None:
            raise ValueError("start_singleton is incompatible with fixed_window_frames")
        if not isinstance(manifest, TrajectoryManifest):
            manifest = load_trajectory_manifest(manifest)
        self.manifest = manifest
        self.split = split
        self.normalizer = normalizer
        self.history_frames = int(history_frames)
        self.fixed_window_frames = (
            int(fixed_window_frames) if fixed_window_frames is not None else None
        )
        self.start_singleton = bool(start_singleton)
        self.window_frames = (
            self.fixed_window_frames
            if self.fixed_window_frames is not None
            else self.history_frames + 1
        )
        self.unroll_steps = int(unroll_steps)
        if self.fixed_window_frames is not None:
            # Parallel TF: one target per input position in the fixed clip.
            self.unroll_steps = self.fixed_window_frames
        self.sample_stride = int(sample_stride)
        self.frame_cache_max_bytes = int(frame_cache_max_bytes)
        self.records = manifest.records_for_split(split, require_production_eligible=True)
        self._shapes: list[TrajectoryShape] = []
        self._point_indices: list[np.ndarray] = []
        # Per-record ordered keyframe timeline (HDF5 frame ids).
        self._input_timelines: list[tuple[int, ...]] = []
        # Samples are (record_index, seq_pos) on that timeline (need next keyframe).
        self._samples: list[tuple[int, int]] = []
        self._handles: dict[Path, h5py.File] = {}
        self._frame_cache: FrameLRUCache | None = None

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
            self._shapes.append(shape)
            self._point_indices.append(
                nested_point_indices(
                    shape.points,
                    subset_points,
                    seed=subset_seed,
                    point_bank_id=record.point_bank_id,
                )
            )
            timeline = resolve_keyframe_timeline(
                n_frames=shape.frames,
                valid_frame_ids=record.valid_frame_ids,
                accepted_transition_indices=record.accepted_transition_indices,
            )
            if timeline[-1] >= shape.frames:
                raise ValueError(
                    f"trajectory {record.trajectory_id} keyframe {timeline[-1]} "
                    f"out of range for {shape.frames} frames"
                )
            if self.fixed_window_frames is not None:
                # Need W input KFs + one KF after the last input for target_W.
                last_seq = len(timeline) - self.fixed_window_frames - 1
            else:
                last_seq = len(timeline) - 1 - self.unroll_steps
            if last_seq < 0:
                raise ValueError(
                    f"trajectory {record.trajectory_id} has too few keyframes "
                    f"({len(timeline)}) for window/unroll"
                )
            self._input_timelines.append(timeline)
            self._samples.extend(
                (record_index, seq_pos)
                for seq_pos in range(0, last_seq + 1, self.sample_stride)
            )

    def __len__(self) -> int:
        return len(self._samples)

    def window_length_at(self, index: int) -> int:
        if self.fixed_window_frames is not None:
            return int(self.fixed_window_frames)
        # start_singleton is a train-time PF init flag (see train_temporal_cfd);
        # samples still expose the full causal GT window so L_clean matches val.
        _record_index, seq_pos = self._samples[index]
        return len(causal_window_frame_ids(seq_pos, self.history_frames))

    def _get_frame_cache(self) -> FrameLRUCache | None:
        if self.frame_cache_max_bytes <= 0:
            return None
        if self._frame_cache is None:
            self._frame_cache = FrameLRUCache(self.frame_cache_max_bytes)
        return self._frame_cache

    def _get_handle(self, path: Path) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        if self._frame_cache is not None:
            self._frame_cache.clear()

    def _encode_state(
        self,
        raw: np.ndarray,
        mask: np.ndarray,
        speed: float | None,
    ) -> np.ndarray:
        out = np.zeros((raw.shape[0], self.normalizer.state_dim), dtype=np.float32)
        if np.any(mask):
            out[mask] = self.normalizer.encode(raw[mask], u_ref_m_per_s=speed)
        return out

    def _cached_take(
        self,
        *,
        record_index: int,
        frame: int | None,
        field: str,
        loader,
    ) -> np.ndarray:
        cache = self._get_frame_cache()
        key = (record_index, frame, field)
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                return hit
        value = loader()
        if cache is not None:
            cache.put(key, value)
        return value

    def _load_static(
        self, handle: h5py.File, indices: np.ndarray, record_index: int, frame: int
    ) -> np.ndarray:
        return self._cached_take(
            record_index=record_index,
            frame=frame,
            field="static",
            loader=lambda: np.asarray(
                _take_points(handle["static_fx"], indices, frame=frame), dtype=np.float32
            ),
        )

    def _load_boundary(
        self, handle: h5py.File, indices: np.ndarray, record_index: int, frame: int
    ) -> np.ndarray:
        # Prefer precomputed KF-hop boundary (uniform_20 → boundary_feat_kf20).
        dset_name = "boundary_feat_kf20" if "boundary_feat_kf20" in handle else "boundary_feat"
        return self._cached_take(
            record_index=record_index,
            frame=frame,
            field=f"boundary:{dset_name}",
            loader=lambda: np.asarray(
                _take_points(handle[dset_name], indices, frame=frame), dtype=np.float32
            ),
        )

    def _load_encoded_state(
        self,
        handle: h5py.File,
        indices: np.ndarray,
        record_index: int,
        frame: int,
        speed: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        def loader() -> tuple[np.ndarray, np.ndarray]:
            raw = _take_points(handle["state_raw"], indices, frame=frame)
            valid = _take_points(handle["valid_mask"], indices, frame=frame).astype(bool)
            fluid = _take_points(handle["fluid_mask"], indices, frame=frame).astype(bool)
            loss_mask = valid & fluid
            encoded = self._encode_state(raw, loss_mask, speed)
            return encoded, loss_mask

        return self._cached_take(
            record_index=record_index,
            frame=frame,
            field="state_enc",
            loader=loader,
        )

    def _load_query_weight(
        self, handle: h5py.File, indices: np.ndarray, record_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        def loader() -> tuple[np.ndarray, np.ndarray]:
            query = np.asarray(
                _take_points(handle["query_xyz_normalized"], indices), dtype=np.float32
            )
            weight = np.asarray(_take_points(handle["loss_weight"], indices), dtype=np.float32)
            return query, weight

        return self._cached_take(
            record_index=record_index,
            frame=None,
            field="query_weight",
            loader=loader,
        )

    def _load_surface_mask(
        self, handle: h5py.File, indices: np.ndarray, record_index: int, frame: int
    ) -> np.ndarray:
        """Per-frame surface cells (True=surface branch); volume = ~surface."""
        if "surface_mask" not in handle:
            raise KeyError(
                f"{handle.filename} missing surface_mask (required for surface/volume branch)"
            )
        return self._cached_take(
            record_index=record_index,
            frame=frame,
            field="surface_mask",
            loader=lambda: np.asarray(
                _take_points(handle["surface_mask"], indices, frame=frame), dtype=bool
            ),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, seq_pos = self._samples[index]
        record = self.records[record_index]
        indices = self._point_indices[record_index]
        handle = self._get_handle(record.hdf5_path)
        speed = record.speed_m_per_s
        timeline = self._input_timelines[record_index]
        current_t = int(timeline[seq_pos])

        if self.fixed_window_frames is not None:
            w = self.fixed_window_frames
            local_ids = list(range(seq_pos, seq_pos + w))
        else:
            # Always return the full causal GT window. When training.mixed
            # start_singleton=True, the PF branch slices to the last frame
            # after L_clean so grow/slide starts at T=1 without poisoning clean.
            local_ids = causal_window_frame_ids(seq_pos, self.history_frames)
        frame_ids = [int(timeline[i]) for i in local_ids]
        window_len = len(frame_ids)
        if window_len < 1 or window_len > self.window_frames:
            raise RuntimeError(
                f"window length {window_len} out of range [1, {self.window_frames}]"
            )

        query_xyz, base_weight = self._load_query_weight(handle, indices, record_index)

        static_list = []
        state_list = []
        boundary_list = []
        surface_mask_list = []
        for local_i, frame in zip(local_ids, frame_ids):
            next_kf = int(timeline[local_i + 1])
            # Next-keyframe pose for static; boundary on current keyframe.
            static_list.append(self._load_static(handle, indices, record_index, next_kf))
            boundary_list.append(self._load_boundary(handle, indices, record_index, frame))
            encoded, _ = self._load_encoded_state(
                handle, indices, record_index, frame, speed
            )
            state_list.append(encoded)
            if "surface_mask" in handle:
                surface_mask_list.append(
                    self._load_surface_mask(handle, indices, record_index, frame)
                )

        target_list = []
        future_static = []
        future_boundary = []
        future_weight = []
        for step in range(self.unroll_steps):
            target_t = int(timeline[seq_pos + 1 + step])
            encoded, loss_mask = self._load_encoded_state(
                handle, indices, record_index, target_t, speed
            )
            target_list.append(encoded)
            future_weight.append(np.where(loss_mask, base_weight, 0.0).astype(np.float32))
            if step + 1 < self.unroll_steps:
                next_after_target = int(timeline[seq_pos + 2 + step])
                future_static.append(
                    self._load_static(handle, indices, record_index, next_after_target)
                )
                future_boundary.append(
                    self._load_boundary(handle, indices, record_index, target_t)
                )
            else:
                future_static.append(np.zeros_like(static_list[0], dtype=np.float32))
                future_boundary.append(np.zeros_like(boundary_list[0], dtype=np.float32))

        target_states = np.stack(target_list, axis=0).astype(np.float32)
        out = {
            "trajectory_id": record.trajectory_id,
            "robot": record.robot,
            "split": record.split,
            "current_frame": current_t,
            "seq_pos": seq_pos,
            "window_length": window_len,
            "query_xyz": np.broadcast_to(
                query_xyz[None], (window_len, *query_xyz.shape)
            ).copy().astype(np.float32),
            "static_fx": np.stack(static_list, axis=0).astype(np.float32),
            "state": np.stack(state_list, axis=0).astype(np.float32),
            "boundary_feat": np.stack(boundary_list, axis=0).astype(np.float32),
            "target_state": target_states[0].astype(np.float32),
            "loss_mask": (future_weight[0] > 0).astype(bool),
            "loss_weight": future_weight[0],
            "target_states": target_states,
            "future_static_fx": np.stack(future_static, axis=0).astype(np.float32),
            "future_boundary_feat": np.stack(future_boundary, axis=0).astype(np.float32),
            "future_loss_weight": np.stack(future_weight, axis=0).astype(np.float32),
        }
        if surface_mask_list:
            out["surface_mask"] = np.stack(surface_mask_list, axis=0)
        return out
