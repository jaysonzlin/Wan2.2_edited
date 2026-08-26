"""Dense Utonia feature transfer between canonical and SimGen object clouds."""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch


CANONICAL_CLASSES = frozenset({"panda", "ball", "can"})
CACHE_RECORD_VERSION = 1
NORMALIZATION_VERSION = "centroid-rms-v1"


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SimGenUtoniaCache:
    """Read class-keyed canonical dense feature records without mutating them."""

    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root)
        self._records: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._feature_width: int | None = None

    def features_for(self, object_name: str, target_points: torch.Tensor) -> torch.Tensor:
        reference_points, features = self._record_for(object_name)
        return transfer_dense_features(reference_points, features, target_points)

    def _record_for(self, object_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if object_name in self._records:
            return self._records[object_name]
        if object_name not in CANONICAL_CLASSES:
            raise ValueError(f"Unknown canonical Utonia class: {object_name}")
        path = self.cache_root / f"{object_name}.pt"
        if not path.is_file():
            raise ValueError(f"Missing canonical Utonia cache for class {object_name}: {path}")
        try:
            record = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(f"Invalid canonical Utonia cache: {path}") from error
        metadata = record.get("metadata") if isinstance(record, dict) else None
        points = record.get("reference_points") if isinstance(record, dict) else None
        features = record.get("features") if isinstance(record, dict) else None
        if not isinstance(metadata, dict) or metadata.get("class_name") != object_name:
            raise ValueError(f"{path}: cache class metadata does not match {object_name}")
        if metadata.get("record_version") != CACHE_RECORD_VERSION:
            raise ValueError(f"{path}: incompatible cache record version")
        if metadata.get("normalization_version") != NORMALIZATION_VERSION:
            raise ValueError(f"{path}: incompatible cache normalization version")
        if not all(
            isinstance(metadata.get(key), str) and metadata[key]
            for key in ("source_fingerprint", "checkpoint_fingerprint", "preprocess_version")
        ):
            raise ValueError(f"{path}: cache fingerprint metadata is missing")
        if not isinstance(points, torch.Tensor) or not isinstance(features, torch.Tensor):
            raise ValueError(f"{path}: cache must contain reference_points and features tensors")
        if points.ndim != 2 or points.shape[1] != 3 or features.ndim != 2 or features.shape[0] != points.shape[0]:
            raise ValueError(f"{path}: invalid canonical Utonia tensor shapes")
        if metadata.get("point_count") != points.shape[0] or metadata.get("feature_dim") != features.shape[1]:
            raise ValueError(f"{path}: cache metadata does not match tensor shapes")
        if not torch.isfinite(points).all() or not torch.isfinite(features).all():
            raise ValueError(f"{path}: cache tensors must be finite")
        result = (points.float().contiguous(), features.float().contiguous())
        if self._feature_width is None:
            self._feature_width = result[1].shape[1]
        elif result[1].shape[1] != self._feature_width:
            raise ValueError("canonical Utonia cache feature widths must agree")
        self._records[object_name] = result
        return result


def prepare_simgen_utonia_cache(canonical_sources, cache_root: str | Path, extractor) -> int:
    """Extract and atomically save one fingerprinted dense record per canonical class."""
    root = Path(cache_root)
    widths = set()
    for class_name, raw_path in canonical_sources.items():
        if class_name not in CANONICAL_CLASSES:
            raise ValueError(f"Unknown canonical Utonia class: {class_name}")
        path = Path(raw_path)
        try:
            source_fingerprint = _file_fingerprint(path)
        except OSError as error:
            raise ValueError(f"Unable to read canonical Utonia source: {path}") from error
        expected_metadata = {
            "class_name": class_name,
            "record_version": CACHE_RECORD_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "source_fingerprint": source_fingerprint,
            "checkpoint_fingerprint": extractor.checkpoint_fingerprint,
            "preprocess_version": extractor.preprocess_version,
        }
        record_path = root / f"{class_name}.pt"
        if record_path.is_file():
            try:
                existing = torch.load(record_path, map_location="cpu", weights_only=True)
                metadata = existing.get("metadata") if isinstance(existing, dict) else None
                features = existing.get("features") if isinstance(existing, dict) else None
            except (OSError, RuntimeError, ValueError):
                metadata, features = None, None
            if (
                isinstance(metadata, dict)
                and all(metadata.get(key) == value for key, value in expected_metadata.items())
                and isinstance(features, torch.Tensor)
                and features.ndim == 2
            ):
                widths.add(features.shape[1])
                continue
        try:
            with h5py.File(path, "r") as source:
                points = np.asarray(source["point_cloud"][0, 0], dtype=np.float32)
                rgb = np.asarray(source["rgb"][:])
        except (OSError, KeyError, IndexError) as error:
            raise ValueError(f"Unable to read canonical Utonia source: {path}") from error
        features = extractor(points, rgb).detach().to(device="cpu", dtype=torch.float32).contiguous()
        if points.ndim != 2 or points.shape[1] != 3 or features.ndim != 2 or features.shape[0] != points.shape[0]:
            raise ValueError(f"{class_name}: extractor returned incompatible dense features")
        record = {
            "metadata": {
                **expected_metadata,
                "point_count": points.shape[0],
                "feature_dim": features.shape[1],
            },
            "reference_points": torch.from_numpy(points).contiguous(),
            "features": features,
        }
        root.mkdir(parents=True, exist_ok=True)
        torch.save(record, root / f"{class_name}.pt")
        widths.add(features.shape[1])
    if not widths or len(widths) != 1:
        raise ValueError("canonical Utonia cache feature widths must agree")
    return widths.pop()


def normalize_object_points(points: torch.Tensor) -> torch.Tensor:
    """Remove world translation and normalize RMS object size without rotation inference."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if not torch.isfinite(points).all():
        raise ValueError("points must be finite")
    centered = points - points.mean(dim=0, keepdim=True)
    radius = centered.square().sum(dim=-1).mean().sqrt().clamp_min(1e-8)
    return centered / radius


def transfer_dense_features(
    reference_points: torch.Tensor,
    reference_features: torch.Tensor,
    target_points: torch.Tensor,
) -> torch.Tensor:
    """Map every target point to its normalized nearest canonical feature."""
    if reference_features.ndim != 2 or reference_features.shape[0] != reference_points.shape[0]:
        raise ValueError("reference_features must have shape [reference_points, D]")
    if not torch.isfinite(reference_features).all():
        raise ValueError("reference_features must be finite")
    nearest = torch.cdist(
        normalize_object_points(target_points), normalize_object_points(reference_points)
    ).argmin(dim=1)
    return reference_features.index_select(0, nearest)
