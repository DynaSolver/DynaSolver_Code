# DynaSolver — Anonymous Conference Submission (Limited Code)

**Anonymous code release for conference submission. Author identities withheld.**

This is a **reviewer-facing, limited** package. It is intentionally incomplete to
protect unpublished implementation details (防洗稿). The full source will be
released upon acceptance.

## What is included

| Component | Status |
|-----------|--------|
| Data layout + kinematic claim docs | Full |
| Flagship training configs (hyperparameters) | Full |
| Dataset loaders (`data_provider/`) | Full |
| Train / AR-eval orchestration scripts | Full (call into model API) |
| `TemporalTransolver` / DynaSolver **model body** | **Stub only** |
| Proprietary temporal / physics layers | **Stub only** |
| Third-party baselines (e.g. MSPT) | **Not included** — use upstream |

Stubs raise `NotImplementedError` with a clear message. Scripts and configs are
provided so reviewers can inspect the **protocol** (data splits, losses, AR
eval, hyperparameters). Algorithms are described in the paper.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# or: pip install -r requirements.txt && export PYTHONPATH="$PWD:$PWD/dynasolver"

export DYNASOLVER_DATA=/path/to/your/data_root
export GEOPT_CHECKPOINT=/path/to/GeoPT_8layers.pt   # optional; init withheld here
```

See [docs/DATA.md](docs/DATA.md) and [docs/kinematic_claim_dataset.md](docs/kinematic_claim_dataset.md).

## Inspect training / eval entrypoints

```bash
# Training CLI + config (model forward is stubbed in this package)
python -u scripts/train_temporal_cfd.py --help
python -u scripts/train_temporal_cfd.py \
  --config configs/casual_dual_grad_latent_dec_s16_rope_waterlily20_kinematic_grid18.yaml \
  --manifest "$DYNASOLVER_DATA/manifests/waterlily20_grid18x18x36_n11664_f30_kinematic_temporal_manifest.json" \
  --normalizer-path "$DYNASOLVER_DATA/manifests/normalizer.json" \
  --output-dir /tmp/dynasolver_submit_dryrun \
  --initialization from_scratch
# → builds dataloaders / optim; model.forward_window raises NotImplementedError

# AR eval CLI (requires a real checkpoint from the full code; not runnable here)
python -u scripts/eval_waterlily20_rollout_vp.py --help
```

## License

MIT — Copyright (c) 2026 Anonymous Authors.
