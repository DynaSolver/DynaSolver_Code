# Data layout

DynaSolver does not ship HDF5 trajectories or checkpoints. Set:

| Variable | Purpose |
|----------|---------|
| `DYNASOLVER_DATA` | Root for manifests, runs, optional checkpoints |
| `GEOPT_CHECKPOINT` | Optional GeoPT backbone weights (`.pt`) |
| `DYNASOLVER_BASELINE_ROOT` | Optional external baseline workspace for `eval_waterlily20_baseline_rollout_vp.py` |
| `PYTHON_BIN` | Interpreter override (defaults to `python`) |

Suggested tree:

```
$DYNASOLVER_DATA/
  manifests/
    waterlily20_grid18x18x36_n11664_f30_kinematic_temporal_manifest.json
  runs/
  checkpoints/
    GeoPT_8layers.pt          # or set GEOPT_CHECKPOINT elsewhere
```

Manifest JSON entries should point at trajectory HDF5 files on your machine.
Example config paths use placeholders under `$DYNASOLVER_DATA`; override any
path with CLI flags (`--manifest`, `--output-dir`, …).

Normalizer JSON (kinematic claim) is typically stored next to the packed
dataset; pass it via config `paths.normalizer` or the training CLI.
