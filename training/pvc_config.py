"""Configuration loading and validation for the dedicated PVC experiment."""

from pathlib import Path

from training.pc_config import load_pc_config


def load_pvc_config(path: str | Path, overrides: list[str]) -> dict:
    """Load a standard PC config and enforce PVC's fixed experiment contract."""
    config = load_pc_config(path, overrides)
    validate_pvc_config(config)
    return config


def validate_pvc_config(config: dict) -> None:
    """Reject any configuration that is not the dedicated PVC history-DDPM run."""
    data = config.get("data", {})
    model = config.get("model", {})
    objective = config.get("objective", {})
    if model.get("conditioning") != "history":
        raise ValueError("PVC requires model.conditioning 'history'")
    if model.get("history_frames") != 4:
        raise ValueError("PVC requires model.history_frames 4")
    if model.get("utonia_enabled") is not True:
        raise ValueError("PVC requires model.utonia_enabled true")
    if model.get("point_view_gate_mode", "shared") not in {"shared", "separate"}:
        raise ValueError("PVC model.point_view_gate_mode must be 'shared' or 'separate'")
    if objective.get("type") != "ddpm":
        raise ValueError("PVC requires objective.type 'ddpm'")
    for field in ("object_id", "utonia_cache_root", "point_view_utonia_cache_root"):
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"PVC requires data.{field} to be a non-empty string")
