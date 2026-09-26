"""Environment-driven path defaults for the anonymous release.

No cluster hosts or personal home directories are hard-coded. Set
``DYNASOLVER_DATA`` (and optionally ``GEOPT_CHECKPOINT``) before training.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

_data_env = os.environ.get("DYNASOLVER_DATA") or os.environ.get(
    "DYNAMIC_TRANSOLVER_DATA"
)
if _data_env:
    DATA_ROOT = Path(_data_env).expanduser().resolve()
else:
    DATA_ROOT = (REPO_ROOT / "data").resolve()

MANIFESTS_DIR = DATA_ROOT / "manifests"
RUNS_DIR = DATA_ROOT / "runs"
LOGS_DIR = DATA_ROOT / "logs"
CHECKPOINTS_DIR = DATA_ROOT / "checkpoints"

# Legacy 16-wind roots (optional; only used if present under DATA_ROOT).
H1_16WIND_ROOT = DATA_ROOT / "h1_16wind"
G1_16WIND_ROOT = DATA_ROOT / "g1_16wind"
H1_16WIND_DATASETS = H1_16WIND_ROOT / "datasets"
G1_16WIND_DATASETS = G1_16WIND_ROOT / "datasets"

H1_16WIND_MANIFEST = MANIFESTS_DIR / "h1_16wind_n250000_manifest.json"
G1_16WIND_MANIFEST = MANIFESTS_DIR / "g1_16wind_n250000_manifest.json"

H1_16WIND_CONTRACT = REPO_ROOT / "configs" / "h1_16wind_legacy_laminar_v0.yaml"
G1_16WIND_CONTRACT = REPO_ROOT / "configs" / "g1_16wind_legacy_laminar_v0.yaml"
LEGACY_H1_G1_CONTRACT = REPO_ROOT / "configs" / "h1_g1_dynamic_v0.yaml"

GEOPT_CHECKPOINT = Path(
    os.environ.get(
        "GEOPT_CHECKPOINT",
        str(CHECKPOINTS_DIR / "GeoPT_8layers.pt"),
    )
).expanduser().resolve()

DEFAULT_PYTHON = os.environ.get("PYTHON_BIN", "python")
DEFAULT_SMOKE_PYTHON = os.environ.get("SMOKE_PYTHON_BIN", DEFAULT_PYTHON)
