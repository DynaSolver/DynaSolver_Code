from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from layers.Temporal_Physics_Block import HistSpatialCache
from models.TemporalTransolver import Model as TemporalModel


@dataclass
class _TimingRecord:
    name: str
    start: torch.cuda.Event | None = None
    end: torch.cuda.Event | None = None
    cpu_start: float | None = None
    cpu_end: float | None = None


def _latency_stats(milliseconds: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(milliseconds.mean()),
        "median": float(np.median(milliseconds)),
        "p95": float(np.percentile(milliseconds, 95)),
        "min": float(milliseconds.min()),
        "max": float(milliseconds.max()),
        "std": float(milliseconds.std()),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class InferenceProfiler:
    """CUDA-event profiler for one forward pass with component breakdown."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.use_cuda = device.type == "cuda"
        self.records: list[_TimingRecord] = []

    def mark_start(self, name: str) -> None:
        if self.use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self.records.append(_TimingRecord(name=name, start=start))
        else:
            self.records.append(_TimingRecord(name=name, cpu_start=time.perf_counter()))

    def mark_end(self, name: str) -> None:
        if self.use_cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            for record in reversed(self.records):
                if record.name == name and record.end is None:
                    record.end = end
                    return
            raise KeyError(f"missing profiler start for {name!r}")
        ended = time.perf_counter()
        for record in reversed(self.records):
            if record.name == name and record.cpu_end is None:
                record.cpu_end = ended
                return
        raise KeyError(f"missing profiler start for {name!r}")

    def elapsed_ms(self) -> dict[str, float]:
        if self.use_cuda:
            synchronize(self.device)
        out: dict[str, float] = {}
        for record in self.records:
            if self.use_cuda:
                assert record.start is not None and record.end is not None
                out[record.name] = float(record.start.elapsed_time(record.end))
            else:
                assert record.cpu_start is not None and record.cpu_end is not None
                out[record.name] = (record.cpu_end - record.cpu_start) * 1000.0
        return out


def profile_temporal_forward_window(
    model: TemporalModel,
    query_xyz: torch.Tensor,
    static_fx: torch.Tensor,
    state: torch.Tensor,
    boundary_feat: torch.Tensor,
    memory_prev: torch.Tensor,
) -> dict[str, float]:
    """Run one instrumented Temporal ``forward_window`` and return component ms."""
    profiler = InferenceProfiler(query_xyz.device)
    batch, frames, _, _ = query_xyz.shape
    memory = memory_prev

    profiler.mark_start("total_forward")

    profiler.mark_start("embed_frames")
    hidden_frames = [
        model.embed_frame(
            query_xyz[:, frame],
            static_fx[:, frame],
            state[:, frame],
            boundary_feat[:, frame],
        )
        for frame in range(frames)
    ]
    hidden_win = torch.stack(hidden_frames, dim=1)
    profiler.mark_end("embed_frames")

    spatial_total = 0.0
    temporal_total = 0.0
    long_memory_total = 0.0
    deslice_mlp_total = 0.0
    state_update_total = 0.0
    point_hidden = None
    hist_cache = HistSpatialCache(window_frames=frames, n_layers=len(model.blocks))

    for layer_index, block in enumerate(model.blocks):
        prefix = f"block_{layer_index:02d}"

        # Integrated path: time one full block forward (figure-aligned Physics-Attn).
        profiler.mark_start(f"{prefix}_spatial")
        out = block(
            hidden_win,
            memory if block.use_long_memory else None,
            layer_idx=layer_index,
            hist_cache=hist_cache,
        )
        point_out = out["point_out"]
        if out["memory_new"] is not None:
            memory = out["memory_new"]
        profiler.mark_end(f"{prefix}_spatial")
        profiler.mark_start(f"{prefix}_temporal")
        profiler.mark_end(f"{prefix}_temporal")
        profiler.mark_start(f"{prefix}_long_memory")
        profiler.mark_end(f"{prefix}_long_memory")
        profiler.mark_start(f"{prefix}_deslice_mlp")
        profiler.mark_end(f"{prefix}_deslice_mlp")

        profiler.mark_start(f"{prefix}_state_update")
        hidden_win = hidden_win.clone()
        hidden_win[:, -1] = point_out
        point_hidden = point_out
        profiler.mark_end(f"{prefix}_state_update")

        partial = profiler.elapsed_ms()
        spatial_total += partial[f"{prefix}_spatial"]
        temporal_total += partial[f"{prefix}_temporal"]
        long_memory_total += partial[f"{prefix}_long_memory"]
        deslice_mlp_total += partial[f"{prefix}_deslice_mlp"]
        state_update_total += partial[f"{prefix}_state_update"]

    assert point_hidden is not None
    profiler.mark_start("output_head")
    delta = model.output_proj(model.output_norm(point_hidden))
    _ = state[:, -1] + delta
    profiler.mark_end("output_head")

    profiler.mark_end("total_forward")
    components = profiler.elapsed_ms()
    components["blocks_spatial_total"] = spatial_total
    components["blocks_temporal_total"] = temporal_total
    components["blocks_long_memory_total"] = long_memory_total
    components["blocks_deslice_mlp_total"] = deslice_mlp_total
    components["blocks_state_update_total"] = state_update_total
    return components


def profile_v0_forward(
    model: torch.nn.Module,
    query_xyz: torch.Tensor,
    static_fx: torch.Tensor,
    previous_state: torch.Tensor,
    boundary_feat: torch.Tensor,
) -> dict[str, float]:
    """Run one instrumented V0 forward and return component ms."""
    profiler = InferenceProfiler(query_xyz.device)
    profiler.mark_start("total_forward")

    profiler.mark_start("embed")
    base = torch.cat((query_xyz, static_fx), dim=-1)
    base_linear = model.preprocess.linear_pre[0]
    activation = model.preprocess.linear_pre[1]
    preactivation = base_linear(base)
    preactivation = preactivation + model.state_proj(previous_state) + model.boundary_proj(boundary_feat)
    hidden = model.preprocess.linear_post(activation(preactivation))
    hidden = hidden + model.placeholder[None, None, :]
    profiler.mark_end("embed")

    blocks_total = 0.0
    for layer_index, block in enumerate(model.blocks):
        name = f"block_{layer_index:02d}"
        profiler.mark_start(name)
        hidden = block(hidden)
        profiler.mark_end(name)
        blocks_total += profiler.elapsed_ms()[name]

    profiler.mark_start("output_head")
    delta = model.output_proj(model.output_norm(hidden))
    _ = previous_state + delta
    profiler.mark_end("output_head")

    profiler.mark_end("total_forward")
    components = profiler.elapsed_ms()
    components["blocks_total"] = blocks_total
    return components


def benchmark_temporal_inference(
    model: TemporalModel,
    tensors: dict[str, torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    device = tensors["query_xyz"].device
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            memory = model.init_memory(tensors["state"].shape[0], device, tensors["state"].dtype)
            profile_temporal_forward_window(
                model,
                tensors["query_xyz"],
                tensors["static_fx"],
                tensors["state"],
                tensors["boundary_feat"],
                memory,
            )
        synchronize(device)

        aggregate: dict[str, list[float]] = {}
        for _ in range(repeats):
            memory = model.init_memory(tensors["state"].shape[0], device, tensors["state"].dtype)
            components = profile_temporal_forward_window(
                model,
                tensors["query_xyz"],
                tensors["static_fx"],
                tensors["state"],
                tensors["boundary_feat"],
                memory,
            )
            for key, value in components.items():
                aggregate.setdefault(key, []).append(value)
        synchronize(device)

    summary = {key: _latency_stats(np.asarray(values, dtype=np.float64)) for key, values in aggregate.items()}
    total_mean = summary["total_forward"]["mean"]
    rolled = {
        "embed_frames": summary["embed_frames"]["mean"],
        "blocks_spatial_total": summary["blocks_spatial_total"]["mean"],
        "blocks_temporal_total": summary["blocks_temporal_total"]["mean"],
        "blocks_long_memory_total": summary["blocks_long_memory_total"]["mean"],
        "blocks_deslice_mlp_total": summary["blocks_deslice_mlp_total"]["mean"],
        "blocks_state_update_total": summary["blocks_state_update_total"]["mean"],
        "output_head": summary["output_head"]["mean"],
        "total_forward": total_mean,
    }
    fractions = {
        key: (value / total_mean if total_mean > 0 else 0.0) for key, value in rolled.items() if key != "total_forward"
    }
    return {
        "scope": "temporal_forward_window_gpu_only_excludes_hdf5_and_host_to_device",
        "warmup_iterations": warmup,
        "measured_iterations": repeats,
        "batch_size": int(tensors["query_xyz"].shape[0]),
        "window_frames": int(tensors["query_xyz"].shape[1]),
        "points": int(tensors["query_xyz"].shape[2]),
        "components_ms": summary,
        "rolled_up_ms_mean": rolled,
        "fraction_of_total_mean": fractions,
    }


def benchmark_v0_inference(
    model: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    device = tensors["query_xyz"].device
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            profile_v0_forward(
                model,
                tensors["query_xyz"],
                tensors["static_fx"],
                tensors["previous_state"],
                tensors["boundary_feat"],
            )
        synchronize(device)

        aggregate: dict[str, list[float]] = {}
        for _ in range(repeats):
            components = profile_v0_forward(
                model,
                tensors["query_xyz"],
                tensors["static_fx"],
                tensors["previous_state"],
                tensors["boundary_feat"],
            )
            for key, value in components.items():
                aggregate.setdefault(key, []).append(value)
        synchronize(device)

    summary = {key: _latency_stats(np.asarray(values, dtype=np.float64)) for key, values in aggregate.items()}
    total_mean = summary["total_forward"]["mean"]
    rolled = {
        "embed": summary["embed"]["mean"],
        "blocks_total": summary["blocks_total"]["mean"],
        "output_head": summary["output_head"]["mean"],
        "total_forward": total_mean,
    }
    fractions = {
        key: (value / total_mean if total_mean > 0 else 0.0) for key, value in rolled.items() if key != "total_forward"
    }
    return {
        "scope": "v0_single_step_gpu_only_excludes_hdf5_and_host_to_device",
        "warmup_iterations": warmup,
        "measured_iterations": repeats,
        "batch_size": int(tensors["query_xyz"].shape[1]),
        "points": int(tensors["query_xyz"].shape[1]),
        "components_ms": summary,
        "rolled_up_ms_mean": rolled,
        "fraction_of_total_mean": fractions,
    }
