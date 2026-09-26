from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def artifact_settings(cfg: dict) -> dict[str, Any]:
    defaults = {
        "logs_dir": None,
        "run_dir_template": "{stamp}_{initialization}_n{points}",
        "write_history_json": True,
        "write_experiment_index": True,
        "link_train_log": True,
    }
    user = cfg.get("artifacts", {})
    if not isinstance(user, dict):
        return defaults
    return {**defaults, **user}


def format_run_dir_name(template: str, *, stamp: str, initialization: str, points: int, batch_size: int) -> str:
    return template.format(
        stamp=stamp,
        initialization=initialization,
        points=points,
        batch_size=batch_size,
    )


def build_run_manifest(
    *,
    run_id: str,
    config_path: Path,
    manifest_path: Path,
    output_dir: Path,
    runs_root: Path,
    seed: int,
    points: int,
    initialization: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    pin_memory: bool,
    epochs: int,
    lr: float,
    history_frames: int,
    subset_seed: int,
    resume_checkpoint: Path | None,
    log_file: Path | None,
    model: dict[str, Any],
    split_counts: dict[str, int],
    checkpoint_load: dict[str, Any] | None,
    command: list[str],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_id": None,
        "started_at_utc": utc_now_iso(),
        "status": "running",
        "config": str(config_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "runs_root": str(runs_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "log_file": str(log_file.resolve()) if log_file is not None else None,
        "seed": seed,
        "points": points,
        "initialization": initialization,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
        "epochs_requested": epochs,
        "lr": lr,
        "history_frames": history_frames,
        "subset_seed": subset_seed,
        "resume_checkpoint": str(resume_checkpoint.resolve()) if resume_checkpoint is not None else None,
        "checkpoint_load": checkpoint_load,
        "model": model,
        "split_counts": split_counts,
        "command": command,
        "git_commit": resolve_git_commit(repo_root),
        "python": sys.executable,
    }


def write_run_status(
    output_dir: Path,
    *,
    status: str,
    run_id: str,
    current_epoch: int,
    epochs_requested: int,
    best_epoch: int,
    best_validation_loss: float,
    last_train_loss: float | None = None,
    last_validation_loss: float | None = None,
    started_at_utc: str,
    message: str | None = None,
) -> None:
    payload = {
        "run_id": run_id,
        "status": status,
        "current_epoch": current_epoch,
        "epochs_requested": epochs_requested,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "last_train_loss": last_train_loss,
        "last_validation_loss": last_validation_loss,
        "started_at_utc": started_at_utc,
        "updated_at_utc": utc_now_iso(),
        "resumable": (output_dir / "checkpoint_last.pt").exists()
        or (output_dir / "checkpoint_best.pt").exists(),
        "checkpoint_last": str((output_dir / "checkpoint_last.pt").resolve())
        if (output_dir / "checkpoint_last.pt").exists()
        else None,
        "checkpoint_best": str((output_dir / "checkpoint_best.pt").resolve())
        if (output_dir / "checkpoint_best.pt").exists()
        else None,
    }
    if message is not None:
        payload["message"] = message
    write_json_atomic(output_dir / "run_status.json", payload)


def write_checkpoint_index(
    output_dir: Path,
    *,
    last_epoch: int | None,
    best_epoch: int | None,
    best_validation_loss: float | None,
) -> None:
    payload = {
        "updated_at_utc": utc_now_iso(),
        "checkpoint_last": {
            "path": str((output_dir / "checkpoint_last.pt").resolve()),
            "epoch": last_epoch,
        }
        if (output_dir / "checkpoint_last.pt").exists()
        else None,
        "checkpoint_best": {
            "path": str((output_dir / "checkpoint_best.pt").resolve()),
            "epoch": best_epoch,
            "validation_loss": best_validation_loss,
        }
        if (output_dir / "checkpoint_best.pt").exists()
        else None,
    }
    write_json_atomic(output_dir / "checkpoint_index.json", payload)


def write_summary(
    output_dir: Path,
    *,
    status: str,
    best_epoch: int,
    best_validation_loss: float,
    epochs_completed: int,
    test_loss: float | None = None,
    message: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": epochs_completed,
        "finished_at_utc": utc_now_iso(),
    }
    if test_loss is not None:
        payload["test_loss"] = test_loss
    if message is not None:
        payload["message"] = message
    write_json_atomic(output_dir / "summary.json", payload)


def update_experiment_index(
    runs_root: Path,
    *,
    run_id: str,
    output_dir: Path,
    initialization: str,
    seed: int,
    points: int,
    batch_size: int,
    status: str,
    best_epoch: int,
    best_validation_loss: float,
    epochs_completed: int,
    log_file: Path | None,
    started_at_utc: str,
) -> None:
    index_path = runs_root / "index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        runs = payload.get("runs", [])
    else:
        payload = {"schema_version": 1, "runs_root": str(runs_root.resolve()), "runs": []}
        runs = []

    entry = {
        "run_id": run_id,
        "output_dir": str(output_dir.resolve()),
        "initialization": initialization,
        "seed": seed,
        "points": points,
        "batch_size": batch_size,
        "status": status,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": epochs_completed,
        "log_file": str(log_file.resolve()) if log_file is not None else None,
        "started_at_utc": started_at_utc,
        "updated_at_utc": utc_now_iso(),
    }

    replaced = False
    for index, existing in enumerate(runs):
        if existing.get("run_id") == run_id or existing.get("output_dir") == entry["output_dir"]:
            runs[index] = entry
            replaced = True
            break
    if not replaced:
        runs.append(entry)

    payload["runs"] = sorted(runs, key=lambda item: item.get("updated_at_utc", ""), reverse=True)
    payload["updated_at_utc"] = utc_now_iso()
    write_json_atomic(index_path, payload)


def link_train_log(output_dir: Path, log_file: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    link_path = output_dir / "train.log"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(log_file.resolve())


def log_banner(message: str) -> None:
    line = f"[{utc_now_iso()}] {message}"
    print(line, flush=True)


def current_command() -> list[str]:
    return list(sys.argv)


class RunArtifactWriter:
    def __init__(
        self,
        *,
        output_dir: Path,
        runs_root: Path,
        run_id: str,
        epochs_requested: int,
        initialization: str,
        seed: int,
        points: int,
        batch_size: int,
        log_file: Path | None,
        write_history_json: bool,
        write_experiment_index: bool,
        link_train_log_enabled: bool,
    ) -> None:
        self.output_dir = output_dir
        self.runs_root = runs_root
        self.run_id = run_id
        self.epochs_requested = epochs_requested
        self.initialization = initialization
        self.seed = seed
        self.points = points
        self.batch_size = batch_size
        self.log_file = log_file
        self.write_history_json = write_history_json
        self.write_experiment_index = write_experiment_index
        self.started_at_utc = utc_now_iso()
        self.status = "running"
        self._finalize_state: dict[str, Any] | None = None

        if link_train_log_enabled and log_file is not None:
            link_train_log(output_dir, log_file)

    def persist_history(self, history: list[dict]) -> None:
        if not history:
            return
        from scripts.train_temporal_cfd import write_history_csv

        write_history_csv(self.output_dir / "epoch_history.csv", history)
        if self.write_history_json:
            write_json_atomic(self.output_dir / "history.json", history)

    def after_epoch(
        self,
        *,
        history: list[dict],
        best_epoch: int,
        best_validation_loss: float,
        last_epoch: int,
    ) -> None:
        row = history[-1]
        self.persist_history(history)
        write_run_status(
            self.output_dir,
            status="running",
            run_id=self.run_id,
            current_epoch=last_epoch,
            epochs_requested=self.epochs_requested,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            last_train_loss=float(row["train_loss"]),
            last_validation_loss=float(row["validation_loss"]),
            started_at_utc=self.started_at_utc,
        )
        write_checkpoint_index(
            self.output_dir,
            last_epoch=last_epoch,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
        )

    def finalize(
        self,
        *,
        status: str,
        best_epoch: int,
        best_validation_loss: float,
        epochs_completed: int,
        test_loss: float | None = None,
        message: str | None = None,
    ) -> None:
        self.status = status
        last_row = self._finalize_state or {}
        write_run_status(
            self.output_dir,
            status=status,
            run_id=self.run_id,
            current_epoch=epochs_completed,
            epochs_requested=self.epochs_requested,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            last_train_loss=last_row.get("last_train_loss"),
            last_validation_loss=last_row.get("last_validation_loss"),
            started_at_utc=self.started_at_utc,
            message=message,
        )
        write_summary(
            self.output_dir,
            status=status,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            epochs_completed=epochs_completed,
            test_loss=test_loss,
            message=message,
        )
        if self.write_experiment_index:
            update_experiment_index(
                self.runs_root,
                run_id=self.run_id,
                output_dir=self.output_dir,
                initialization=self.initialization,
                seed=self.seed,
                points=self.points,
                batch_size=self.batch_size,
                status=status,
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
                epochs_completed=epochs_completed,
                log_file=self.log_file,
                started_at_utc=self.started_at_utc,
            )
        log_banner(
            f"RUN {status.upper()} epochs={epochs_completed} best_epoch={best_epoch} "
            f"best_val={best_validation_loss:.6f}"
        )

    def remember_last_losses(self, train_loss: float, validation_loss: float) -> None:
        self._finalize_state = {
            "last_train_loss": train_loss,
            "last_validation_loss": validation_loss,
        }
