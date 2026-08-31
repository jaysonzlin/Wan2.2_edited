"""PC-only training primitives for native SimGen history trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

import torch

from training.pc_ddpm import make_pc_ddpm_batch
from training.pc_visualization import save_pointcloud_comparison_mp4


@dataclass(frozen=True)
class SimGenPCBatch:
    """Independent object trajectories extracted from one SimGen sample."""

    history: torch.Tensor
    future: torch.Tensor
    utonia_features: torch.Tensor


def flatten_simgen_pc_batch(batch: dict[str, object], device: torch.device | str) -> SimGenPCBatch:
    """Flatten one variable-object sample into independently modeled trajectories."""
    point_clouds = batch["point_clouds"]
    utonia_features = batch["utonia_features"]
    if not isinstance(point_clouds, torch.Tensor) or not isinstance(utonia_features, torch.Tensor):
        raise ValueError("SimGen PC pretraining requires point clouds and Utonia features")
    if point_clouds.ndim != 6 or point_clouds.shape[0] != 1 or point_clouds.shape[2:4] != (49, 1):
        raise ValueError("expected point_clouds with shape [1, K, 49, 1, N, 3]")
    if point_clouds.shape[-1] != 3:
        raise ValueError("point_clouds must store XYZ coordinates")
    if (
        utonia_features.ndim != 4
        or utonia_features.shape[:2] != point_clouds.shape[:2]
        or utonia_features.shape[2] != point_clouds.shape[-2]
    ):
        raise ValueError("utonia_features must align with every SimGen object and point")
    point_clouds = point_clouds.to(device, non_blocking=True)
    utonia_features = utonia_features.to(device, non_blocking=True)
    return SimGenPCBatch(
        history=point_clouds[0, :, :4],
        future=point_clouds[0, :, 4:],
        utonia_features=utonia_features[0],
    )


def pc_pretraining_prediction(
    batch: SimGenPCBatch,
    model,
    noise_scheduler,
    generator: torch.Generator,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict clean future trajectories from noised futures and known history."""
    ddpm_batch = make_pc_ddpm_batch(
        batch.future, noise_scheduler, generator, known_frames=4
    )
    prediction = model(
        ddpm_batch.model_input,
        ddpm_batch.frame_times,
        batch.history,
        None,
        None,
        utonia_features=batch.utonia_features,
    )
    return prediction, ddpm_batch.target


def object_mse_totals(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return the sum of independent-object MSEs and its object count."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    per_object = (prediction - target).square().flatten(1).mean(dim=1)
    return per_object.sum(), per_object.numel()


def reduce_object_mean(accelerator, local_sum: torch.Tensor, local_count: int) -> torch.Tensor:
    """Reduce a validation MSE by objects rather than by rank or sample."""
    global_sum = accelerator.reduce(local_sum, reduction="sum")
    global_count = accelerator.reduce(local_sum.new_tensor(local_count), reduction="sum")
    return global_sum / global_count


def load_best_export_metadata(output_dir: str | Path) -> tuple[float, int]:
    """Return the best validation score saved by this pretraining run."""
    path = Path(output_dir) / "best_pc_model.json"
    if not path.exists():
        return float("inf"), 0
    try:
        payload = json.loads(path.read_text())
        validation_loss = float(payload["validation_loss"])
        step = payload["step"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid best PC export metadata: {path}") from error
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"Invalid best PC export metadata: {path}")
    return validation_loss, step


def _atomic_torch_save(value, path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as output:
        temporary_path = Path(output.name)
    try:
        torch.save(value, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json_save(value: dict[str, float | int], path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", mode="w", encoding="utf-8", delete=False
    ) as output:
        json.dump(value, output)
        temporary_path = Path(output.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_best_pc_export(
    model: torch.nn.Module,
    output_dir: str | Path,
    *,
    validation_loss: float,
    step: int,
    best_loss: float,
) -> bool:
    """Atomically replace the transfer artifact only when validation improves."""
    if validation_loss >= best_loss:
        return False
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(model.state_dict(), root / "best_pc_model.pt")
    _atomic_json_save(
        {"validation_loss": float(validation_loss), "step": int(step)},
        root / "best_pc_model.json",
    )
    return True


@torch.no_grad()
def save_sample_zero_visualizations(
    pipeline,
    batch: SimGenPCBatch,
    output_dir: str | Path,
    *,
    step: int,
    fps: int,
    device: torch.device | str,
    num_inference_steps: int,
    generator: torch.Generator,
) -> None:
    """Render one history-conditioned trajectory comparison per sample-0 object."""
    predicted_future = pipeline(
        batch.history,
        device,
        num_inference_steps,
        generator,
        utonia_features=batch.utonia_features,
    )
    prediction = torch.cat((batch.history, predicted_future), dim=1)
    ground_truth = torch.cat((batch.history, batch.future), dim=1)
    sample_dir = Path(output_dir) / "visualizations" / f"step_{step:07d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for object_index in range(prediction.shape[0]):
        save_pointcloud_comparison_mp4(
            prediction[object_index, :, 0].detach().cpu().numpy()[:, None],
            ground_truth[object_index, :, 0].detach().cpu().numpy()[:, None],
            sample_dir / f"object_{object_index:03d}_trajectory_comparison.mp4",
            fps,
        )
