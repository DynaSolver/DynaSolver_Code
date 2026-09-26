from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.export_h1_legacy_overfit import interval_dir

import h5py
import numpy as np

from data_provider.dynamic_cfd import (
    DynamicCFDOneStepDataset,
    DynamicStateNormalizer,
    load_trajectory_manifest,
    nested_point_indices,
    validate_trajectory_hdf5,
)
from dynamic_cfd.diagnostic_overfit import (
    finite_difference_link_velocity,
    fit_diagnostic_normalizer,
    load_diagnostic_transition,
    load_diagnostic_sequence,
    select_nested_stratified_point_indices,
    select_stratified_point_indices,
)


def write_trajectory(
    path: Path,
    offset: float = 0.0,
    *,
    state_dim: int = 6,
    inflow_direction=(1.0, 0.0, 0.0),
    normalized_speed: float = 1.0,
    times=None,
) -> dict[str, np.ndarray]:
    frames, points, links = 3, 12, 2
    rng = np.random.default_rng(17)
    query = rng.normal(size=(points, 3)).astype(np.float32)
    state = np.empty((frames, points, state_dim), dtype=np.float32)
    for frame in range(frames):
        state[frame, :, :3] = offset + frame + rng.normal(scale=0.1, size=(points, 3))
        state[frame, :, 3] = offset + 2.0 * frame + rng.normal(scale=0.1, size=points)
        if state_dim == 6:
            state[frame, :, 4] = 0.2 + 0.01 * frame + rng.random(points)
            state[frame, :, 5] = 1.0 + 0.1 * frame + rng.random(points)
    valid = np.ones((frames, points), dtype=np.uint8)
    fluid = np.ones((frames, points), dtype=np.uint8)
    valid[0, 0] = 0
    fluid[0, 0] = 0
    valid[2, 1] = 0
    fluid[2, 1] = 0
    state[0, 0] = np.nan
    state[2, 1] = np.nan
    sdf = np.ones((frames, points), dtype=np.float32)
    sdf[~fluid.astype(bool)] = -1.0
    normals = np.zeros((frames, points, 3), dtype=np.float32)
    normals[..., 0] = 1.0
    wall_velocity = rng.normal(size=(frames - 1, points, 3)).astype(np.float32)
    boundary = np.empty((frames - 1, points, 8), dtype=np.float32)
    boundary[..., 0] = sdf[:-1]
    boundary[..., 1] = sdf[1:] - sdf[:-1]
    boundary[..., 2:5] = wall_velocity
    boundary[..., 5] = fluid[:-1]
    boundary[..., 6] = fluid[1:]
    boundary[..., 7] = 0.1
    static_fx = np.empty((frames, points, 11), dtype=np.float32)
    static_fx[..., :3] = query[None, ...]
    static_fx[..., 3] = sdf
    static_fx[..., 4:7] = normals
    static_fx[..., 7:10] = np.asarray(inflow_direction, dtype=np.float32)
    static_fx[..., 10] = normalized_speed
    transforms = np.broadcast_to(np.eye(4), (frames, links, 4, 4)).copy()
    link_velocity = rng.normal(size=(frames - 1, links, 6)).astype(np.float32)
    loss_weight = np.linspace(0.5, 1.5, points, dtype=np.float32)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("points_world_m", data=query + 1.0)
        handle.create_dataset("query_xyz_normalized", data=query)
        handle.create_dataset(
            "times",
            data=np.asarray([0.0, 0.001, 0.002] if times is None else times),
        )
        handle.create_dataset("state_raw", data=state)
        handle.create_dataset("valid_mask", data=valid)
        handle.create_dataset("fluid_mask", data=fluid)
        handle.create_dataset("sdf", data=sdf)
        handle.create_dataset("nearest_normal", data=normals)
        handle.create_dataset("wall_velocity", data=wall_velocity)
        handle.create_dataset("boundary_feat", data=boundary)
        handle.create_dataset("static_fx", data=static_fx)
        handle.create_dataset("loss_weight", data=loss_weight)
        handle.create_dataset("link_transform", data=transforms)
        handle.create_dataset("link_velocity", data=link_velocity)
    return {
        "query": query,
        "state": state,
        "valid": valid,
        "fluid": fluid,
        "boundary": boundary,
        "static_fx": static_fx,
        "loss_weight": loss_weight,
    }


class DynamicCFDDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.arrays = {}
        entries = []
        for index, (trajectory_id, split, offset) in enumerate(
            (
                ("h1_train", "train", 0.0),
                ("h1_validation", "validation", 100.0),
                ("h1_test", "test", 200.0),
            )
        ):
            path = self.root / f"trajectory_{index}.h5"
            self.arrays[trajectory_id] = write_trajectory(path, offset)
            entries.append(
                {
                    "trajectory_id": trajectory_id,
                    "robot": "H1",
                    "split": split,
                    "hdf5_path": path.name,
                    "training_data_eligible": True,
                    "point_bank_id": "h1_bank_v1",
                }
            )
        self.manifest_path = self.root / "manifest.json"
        self._write_manifest(entries)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_manifest(self, entries) -> None:
        self.manifest_path.write_text(
            json.dumps({"schema_version": 1, "trajectories": entries}),
            encoding="utf-8",
        )

    def _manifest_entries(self):
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))["trajectories"]

    def _normalizer(self):
        return DynamicStateNormalizer.fit_from_manifest(
            self.manifest_path,
            u_ref_m_per_s=2.0,
            l_ref_m=1.5,
            p_farfield_kinematic=0.25,
        )

    def test_manifest_enforces_trajectory_level_splits_and_eligibility(self):
        manifest = load_trajectory_manifest(self.manifest_path)
        self.assertEqual([record.split for record in manifest.records], list(("train", "validation", "test")))

        entries = self._manifest_entries()
        entries[2]["hdf5_path"] = entries[0]["hdf5_path"]
        self._write_manifest(entries)
        with self.assertRaisesRegex(ValueError, "split leakage"):
            load_trajectory_manifest(self.manifest_path)

        entries[2]["hdf5_path"] = "trajectory_2.h5"
        entries[2]["training_data_eligible"] = False
        self._write_manifest(entries)
        with self.assertRaisesRegex(ValueError, "non-production"):
            load_trajectory_manifest(
                self.manifest_path,
                require_production_eligible=True,
            )

    def test_manifest_can_require_and_verify_asset_hashes(self):
        entries = self._manifest_entries()
        for entry in entries:
            payload = (self.root / entry["hdf5_path"]).read_bytes()
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
        self._write_manifest(entries)
        load_trajectory_manifest(
            self.manifest_path,
            require_hashes=True,
            verify_hashes=True,
        )

        entries[0]["sha256"] = "0" * 64
        self._write_manifest(entries)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            load_trajectory_manifest(self.manifest_path, verify_hashes=True)

    def test_hdf5_schema_validation_accepts_valid_file_and_rejects_shape_drift(self):
        path = self.root / "trajectory_0.h5"
        shape = validate_trajectory_hdf5(path)
        self.assertEqual((shape.frames, shape.points, shape.state_dim, shape.links), (3, 12, 6, 2))

        with h5py.File(path, "a") as handle:
            del handle["boundary_feat"]
            handle.create_dataset("boundary_feat", data=np.zeros((2, 12, 7)))
        with self.assertRaisesRegex(ValueError, "boundary_feat must have shape"):
            validate_trajectory_hdf5(path, validate_values=False)

    def test_hdf5_validation_rejects_feature_semantics_and_nonpositive_sst_state(self):
        path = self.root / "trajectory_0.h5"
        with h5py.File(path, "a") as handle:
            handle["static_fx"][1, 2, 3] += 0.5
        with self.assertRaisesRegex(ValueError, "static_fx.*sdf"):
            validate_trajectory_hdf5(path)

        write_trajectory(path)
        with h5py.File(path, "a") as handle:
            handle["state_raw"][1, 2, 4] = 0.0
        with self.assertRaisesRegex(ValueError, "k and omega must be positive"):
            validate_trajectory_hdf5(path)

    def test_normalizer_uses_only_train_split_and_round_trips(self):
        normalizer = self._normalizer()
        train = self.arrays["h1_train"]
        active = train["valid"].astype(bool) & train["fluid"].astype(bool)
        raw_train = train["state"][active]
        expected_probe = DynamicStateNormalizer(
            u_ref_m_per_s=2.0,
            l_ref_m=1.5,
            p_farfield_kinematic=0.25,
            mean=np.zeros(6),
            std=np.ones(6),
        )
        transformed = expected_probe._nondimensionalize(raw_train)
        np.testing.assert_allclose(normalizer.mean, transformed.mean(axis=0), rtol=1e-6, atol=1e-7)

        raw = raw_train[:5]
        decoded = normalizer.decode(normalizer.encode(raw))
        np.testing.assert_allclose(decoded, raw, rtol=2e-6, atol=2e-6)
        restored = DynamicStateNormalizer.from_dict(normalizer.to_dict())
        np.testing.assert_allclose(restored.encode(raw), normalizer.encode(raw))

    def test_normalizer_applies_physical_nondimensionalization(self):
        normalizer = DynamicStateNormalizer(
            u_ref_m_per_s=2.0,
            l_ref_m=1.5,
            p_farfield_kinematic=0.25,
            mean=np.zeros(6),
            std=np.ones(6),
        )
        raw = np.array([[2.0, 4.0, 6.0, 9.25, 8.0, 4.0]], dtype=np.float32)
        expected = np.array(
            [[1.0, 2.0, 3.0, 2.25, np.log(2.0), np.log(3.0)]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(normalizer.encode(raw), expected, rtol=1e-6, atol=1e-6)

    def test_one_step_dataset_masks_previous_and_target_states(self):
        normalizer = self._normalizer()
        dataset = DynamicCFDOneStepDataset(
            self.manifest_path,
            "train",
            normalizer,
            subset_points=12,
            subset_seed=9,
        )
        self.addCleanup(dataset.close)
        self.assertEqual(len(dataset), 2)
        sample = dataset[0]
        indices = dataset.point_indices("h1_train")
        point_zero = int(np.flatnonzero(indices == 0)[0])

        self.assertEqual(tuple(sample["query_xyz"].shape), (12, 3))
        self.assertEqual(tuple(sample["static_fx"].shape), (12, 11))
        self.assertEqual(tuple(sample["previous_state"].shape), (12, 6))
        self.assertEqual(tuple(sample["boundary_feat"].shape), (12, 8))
        self.assertFalse(bool(sample["previous_mask"][point_zero]))
        self.assertTrue(np.all(sample["previous_state"][point_zero].numpy() == 0))
        self.assertTrue(bool(sample["loss_mask"][point_zero]))
        self.assertGreater(float(sample["loss_weight"][point_zero]), 0.0)

        train = self.arrays["h1_train"]
        np.testing.assert_allclose(sample["query_xyz"].numpy(), train["query"][indices])
        np.testing.assert_allclose(sample["static_fx"].numpy(), train["static_fx"][1, indices])
        np.testing.assert_allclose(sample["boundary_feat"].numpy(), train["boundary"][0, indices])

        second = dataset[1]
        point_one = int(np.flatnonzero(indices == 1)[0])
        self.assertFalse(bool(second["loss_mask"][point_one]))
        self.assertEqual(float(second["loss_weight"][point_one]), 0.0)
        self.assertTrue(np.all(second["target_state"][point_one].numpy() == 0))
        self.assertEqual(dataset.sample_indices_for_trajectory("h1_train"), (0, 1))

    def test_point_subsets_are_deterministic_and_strictly_nested(self):
        small = nested_point_indices(20, 5, seed=3, point_bank_id="H1")
        medium = nested_point_indices(20, 11, seed=3, point_bank_id="H1")
        repeat = nested_point_indices(20, 5, seed=3, point_bank_id="H1")
        np.testing.assert_array_equal(small, medium[:5])
        np.testing.assert_array_equal(small, repeat)
        self.assertEqual(len(np.unique(medium)), 11)

    def test_diagnostic_point_bank_is_deterministic_and_unique(self):
        rng = np.random.default_rng(4)
        centres = rng.uniform((-2.0, -1.0, -1.0), (4.0, 1.0, 2.0), size=(200, 3))
        distances = np.linalg.norm(centres, axis=1)
        bounds = np.array([[-0.2, 0.2], [-0.3, 0.3], [-0.8, 0.8]])
        first = select_stratified_point_indices(centres, distances, bounds, 80, seed=7)
        second = select_stratified_point_indices(centres, distances, bounds, 80, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.size, np.unique(first).size)
        np.testing.assert_array_equal(first[:40], np.argsort(distances)[:40])

        master, subset = select_nested_stratified_point_indices(
            centres, distances, bounds, 120, 40, seed=7
        )
        self.assertEqual(np.unique(master).size, 120)
        self.assertEqual(np.unique(subset).size, 40)
        self.assertTrue(np.all((subset >= 0) & (subset < master.size)))

    def test_legacy_t580_interval_uses_authoritative_bridge(self):
        bridge = self.root / "posewise_interval_legacy_offset_bridge_t000575ms_to_t000580ms_v1"
        bridge.mkdir()
        resolved = interval_dir(self.root, 575, 580)
        self.assertEqual(resolved, bridge)

    def test_link_velocity_uses_world_frame_finite_differences(self):
        transforms = np.broadcast_to(np.eye(4), (2, 1, 4, 4)).copy()
        transforms[1, 0, 0, 3] = 0.2
        angle = 0.1
        transforms[1, 0, :3, :3] = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        velocity = finite_difference_link_velocity(transforms, np.array([0.0, 0.1]))
        np.testing.assert_allclose(velocity[0, 0, :3], [2.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(velocity[0, 0, 3:], [0.0, 0.0, 1.0], atol=1e-6)

    def test_diagnostic_loader_requires_explicit_nonproduction_attrs(self):
        path = self.root / "diagnostic.h5"
        write_trajectory(path, state_dim=4)
        normalizer = DynamicStateNormalizer(
            u_ref_m_per_s=2.0,
            l_ref_m=1.0,
            p_farfield_kinematic=0.0,
            mean=np.zeros(4),
            std=np.ones(4),
        )
        with self.assertRaisesRegex(ValueError, "diagnostic_only"):
            load_diagnostic_transition(str(path), 0, normalizer)

        with h5py.File(path, "a") as handle:
            handle.attrs["diagnostic_only"] = True
            handle.attrs["training_data_eligible"] = False
            handle.create_dataset("nested_subset_indices_8", data=np.arange(8))
            handle["state_raw"][1, 0, :3] = [100.0, 0.0, 0.0]
        fitted = fit_diagnostic_normalizer(
            str(path), u_ref_m_per_s=2.0, l_ref_m=1.0, max_speed_m_per_s=5.0
        )
        batch = load_diagnostic_transition(str(path), 0, fitted, point_count=8, subset_seed=5)
        self.assertEqual(tuple(batch["previous_state"].shape), (1, 8, 4))
        self.assertEqual(tuple(batch["boundary_feat"].shape), (1, 8, 8))
        self.assertTrue(bool(batch["loss_mask"].any()))
        sequence = load_diagnostic_sequence(
            str(path), fitted, point_count=8, max_speed_m_per_s=5.0
        )
        self.assertEqual(sequence["state"].shape, (3, 8, 4))
        self.assertEqual(sequence["boundary_feat"].shape, (2, 8, 8))
        self.assertFalse(sequence["active_mask"][1, 0])

    def test_dataset_rejects_ineligible_selected_split(self):
        entries = self._manifest_entries()
        entries[0]["training_data_eligible"] = False
        self._write_manifest(entries)
        normalizer = DynamicStateNormalizer(
            u_ref_m_per_s=1.0,
            l_ref_m=1.0,
            p_farfield_kinematic=0.0,
            mean=np.zeros(6),
            std=np.ones(6),
        )
        with self.assertRaisesRegex(ValueError, "not production eligible"):
            DynamicCFDOneStepDataset(self.manifest_path, "train", normalizer)


if __name__ == "__main__":
    unittest.main()
