import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class IPCFrameDataset(Dataset):
    def __init__(self, data_path, frame_ids, num_points, full_mesh=False, seed=0):
        self.data_path = Path(data_path)
        self.frame_ids = list(frame_ids)
        self.num_points = int(num_points)
        self.full_mesh = bool(full_mesh)
        self.seed = int(seed)

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, index):
        frame_id = self.frame_ids[index]
        x = np.load(self.data_path / f"x_{frame_id:06d}.npy")
        y = np.load(self.data_path / f"y_{frame_id:06d}.npy")
        point_idx = np.arange(x.shape[0], dtype=np.int64)

        if x.ndim != 2 or x.shape[-1] != 11:
            raise RuntimeError(
                f"x_{frame_id:06d}.npy must have shape (N, 11), got {x.shape}."
            )
        if y.ndim != 2 or y.shape[-1] != 3:
            raise RuntimeError(
                f"y_{frame_id:06d}.npy must have shape (N, 3), got {y.shape}."
            )
        if x.shape[0] != y.shape[0]:
            raise RuntimeError(
                f"x/y point counts differ for frame {frame_id:06d}: "
                f"{x.shape[0]} vs {y.shape[0]}."
            )

        if not self.full_mesh:
            if self.num_points <= 0:
                raise RuntimeError("--num_points must be positive for sampled IPC loading.")
            if x.shape[0] < self.num_points:
                raise RuntimeError(
                    f"x_{frame_id:06d}.npy has {x.shape[0]} points, "
                    f"but --num_points={self.num_points}."
                )
            rng = np.random.default_rng(self.seed + frame_id)
            point_idx = rng.choice(x.shape[0], size=self.num_points, replace=False)
            x = x[point_idx]
            y = y[point_idx]

        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        pos = x[:, :3]
        fx = x
        cond = np.zeros((1, 1), dtype=np.float32)
        return (
            torch.from_numpy(pos),
            torch.from_numpy(fx),
            torch.from_numpy(cond),
            torch.from_numpy(y),
            torch.from_numpy(np.ascontiguousarray(point_idx, dtype=np.int64)),
        )


class IPC(object):
    def __init__(self, args):
        self.data_path = Path(args.data_path)
        self.batch_size = int(args.batch_size)
        self.full_mesh_batch_size = int(getattr(args, "full_mesh_batch_size", 1))
        self.ntrain = int(args.ntrain)
        self.ntest = int(args.ntest)
        self.num_points = int(args.num_points)
        self.normalize = bool(args.normalize)
        self._cloth_target_stats = None

        if not self.data_path.is_dir():
            raise RuntimeError(f"IPC data_path does not exist: {self.data_path}")

        self.frame_ids = self._discover_frame_ids()
        required = self.ntrain + self.ntest
        if len(self.frame_ids) < required:
            raise RuntimeError(
                f"Not enough IPC frames under {self.data_path}: available={len(self.frame_ids)}, "
                f"but ntrain+ntest={required}."
            )
        self.train_ids = self.frame_ids[: self.ntrain]
        self.test_ids = self.frame_ids[self.ntrain : required]

    def _discover_frame_ids(self):
        x_ids = self._ids_for_prefix("x")
        y_ids = self._ids_for_prefix("y")
        if not x_ids:
            raise RuntimeError(f"No x_*.npy files found under {self.data_path}")
        if x_ids != y_ids:
            missing_y = sorted(set(x_ids) - set(y_ids))
            missing_x = sorted(set(y_ids) - set(x_ids))
            raise RuntimeError(
                "IPC x/y frame ids do not match. "
                f"Missing y for {missing_y[:10]}, missing x for {missing_x[:10]}."
            )

        expected = list(range(x_ids[0], x_ids[-1] + 1))
        if x_ids != expected:
            missing = sorted(set(expected) - set(x_ids))
            raise RuntimeError(
                "IPC frame ids must be continuous. "
                f"Found {x_ids[0]}..{x_ids[-1]}, missing {missing[:20]}."
            )
        return x_ids

    def _ids_for_prefix(self, prefix):
        pattern = re.compile(rf"{prefix}_(\d{{6}})\.npy$")
        ids = []
        for path in self.data_path.iterdir():
            match = pattern.match(path.name)
            if match:
                ids.append(int(match.group(1)))
        return sorted(ids)

    def get_loader(self, full_mesh=False):
        train_dataset = IPCFrameDataset(
            self.data_path,
            self.train_ids,
            self.num_points,
            full_mesh=full_mesh,
            seed=0,
        )
        test_dataset = IPCFrameDataset(
            self.data_path,
            self.test_ids,
            self.num_points,
            full_mesh=full_mesh,
            seed=10_000,
        )
        batch_size = self.full_mesh_batch_size if full_mesh else self.batch_size
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=not full_mesh,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        point_count = self._point_count(full_mesh)
        print(
            f"IPC dataloading is over. train={len(train_dataset)}, test={len(test_dataset)}, "
            f"points={'full' if full_mesh else self.num_points}, batch_size={batch_size}"
        )
        return train_loader, test_loader, [point_count]

    def _point_count(self, full_mesh):
        if not full_mesh:
            return self.num_points
        first_id = self.frame_ids[0]
        x = np.load(self.data_path / f"x_{first_id:06d}.npy", mmap_mode="r")
        return int(x.shape[0])

    def get_cloth_target_stats(self):
        if self._cloth_target_stats is not None:
            return self._cloth_target_stats

        target_sum = np.zeros(3, dtype=np.float64)
        target_sq_sum = np.zeros(3, dtype=np.float64)
        cloth_points = 0

        for frame_id in self.train_ids:
            x = np.load(self.data_path / f"x_{frame_id:06d}.npy", mmap_mode="r")
            y = np.load(self.data_path / f"y_{frame_id:06d}.npy", mmap_mode="r")
            cloth_mask = x[:, 9] > 0.5
            cloth_y = np.asarray(y[cloth_mask], dtype=np.float64)
            if cloth_y.size == 0:
                raise RuntimeError(f"Frame {frame_id:06d} has no cloth targets for normalization.")

            target_sum += cloth_y.sum(axis=0)
            target_sq_sum += np.square(cloth_y).sum(axis=0)
            cloth_points += int(cloth_y.shape[0])

        if cloth_points <= 0:
            raise RuntimeError("IPC normalization requires at least one cloth target point.")

        mean = target_sum / cloth_points
        variance = np.maximum(target_sq_sum / cloth_points - np.square(mean), 0.0)
        std = np.sqrt(variance)

        self._cloth_target_stats = {
            "mean": mean.astype(np.float32),
            "std": np.maximum(std, 1.0e-8).astype(np.float32),
            "count": cloth_points,
        }
        return self._cloth_target_stats
