"""Persistent frozen-Utonia cache records for variable-size point views."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from typing import Mapping, Protocol

import torch

from training.pvc_dataset import POINT_VIEW_FRAMES, POINT_VIEW_POINTS, read_point_view


CACHE_VERSION = "pvc-utonia-features-v1"


class PointViewFeatureExtractor(Protocol):
    checkpoint_fingerprint: str
    preprocess_version: str

    def __call__(self, coordinates, rgb) -> torch.Tensor: ...


def _cache_path(cache_root: str | Path, sample_id: str) -> Path:
    parts = Path(sample_id).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid PVC cache sample id: {sample_id!r}")
    return Path(cache_root).joinpath(*parts, "point_view_utonia_features.pt")


def _source_fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _valid_record(record, *, sample_id, fingerprint, extractor, feature_dim):
    if not isinstance(record, dict):
        return None
    metadata, features, mask = record.get("metadata"), record.get("features"), record.get("mask")
    if not isinstance(metadata, dict) or not isinstance(features, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return None
    if metadata != {
        "cache_version": CACHE_VERSION,
        "sample_id": sample_id,
        "source_fingerprint": fingerprint,
        "checkpoint_fingerprint": extractor.checkpoint_fingerprint,
        "preprocess_version": extractor.preprocess_version,
        "feature_dim": feature_dim,
    }:
        return None
    if features.shape != (POINT_VIEW_FRAMES, POINT_VIEW_POINTS, feature_dim) or features.dtype != torch.float32:
        return None
    if mask.shape != (POINT_VIEW_FRAMES, POINT_VIEW_POINTS) or mask.dtype != torch.bool:
        return None
    if not torch.isfinite(features).all():
        return None
    return features.contiguous(), mask.contiguous()


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(record, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def prepare_point_view_utonia_feature_cache(
    sources: Mapping[str, tuple[Path, ...]], cache_root: str | Path, extractor: PointViewFeatureExtractor, *, feature_dim: int
) -> None:
    """Create or reuse one validated feature/mask record for each clip."""
    if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim <= 0:
        raise ValueError("feature_dim must be a positive integer")
    for sample_id, raw_paths in sorted(sources.items()):
        paths = tuple(Path(path) for path in raw_paths)
        if len(paths) != POINT_VIEW_FRAMES:
            raise ValueError(f"{sample_id}: point views must contain exactly 49 frames")
        fingerprint = _source_fingerprint(paths)
        cache_path = _cache_path(cache_root, sample_id)
        try:
            matching = _valid_record(_load(cache_path), sample_id=sample_id, fingerprint=fingerprint, extractor=extractor, feature_dim=feature_dim)
        except (OSError, RuntimeError, ValueError):
            matching = None
        if matching is not None:
            continue
        features = torch.zeros((POINT_VIEW_FRAMES, POINT_VIEW_POINTS, feature_dim), dtype=torch.float32)
        mask = torch.zeros((POINT_VIEW_FRAMES, POINT_VIEW_POINTS), dtype=torch.bool)
        for frame, path in enumerate(paths):
            xyz, rgb = read_point_view(path)
            count = xyz.shape[0]
            if not count:
                continue
            encoded = extractor(xyz, rgb)
            if not isinstance(encoded, torch.Tensor) or encoded.shape != (count, feature_dim):
                raise ValueError(f"{path}: Utonia features must have shape ({count}, {feature_dim})")
            if not torch.isfinite(encoded).all():
                raise ValueError(f"{path}: Utonia features must be finite")
            features[frame, :count] = encoded.detach().to(device="cpu", dtype=torch.float32)
            mask[frame, :count] = True
        metadata = {
            "cache_version": CACHE_VERSION, "sample_id": sample_id,
            "source_fingerprint": fingerprint,
            "checkpoint_fingerprint": extractor.checkpoint_fingerprint,
            "preprocess_version": extractor.preprocess_version, "feature_dim": feature_dim,
        }
        _write(cache_path, {"metadata": metadata, "features": features, "mask": mask})


def load_cached_point_view_utonia_features(cache_root: str | Path, sample_id: str, *, feature_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a structurally valid PVC cache record without source I/O."""
    path = _cache_path(cache_root, sample_id)
    try:
        record = _load(path)
        features, mask = record["features"], record["mask"]
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        raise ValueError(f"Invalid or missing PVC Utonia cache entry: {path}") from error
    if features.shape != (POINT_VIEW_FRAMES, POINT_VIEW_POINTS, feature_dim) or features.dtype != torch.float32:
        raise ValueError("PVC Utonia features have an invalid shape or dtype")
    if mask.shape != (POINT_VIEW_FRAMES, POINT_VIEW_POINTS) or mask.dtype != torch.bool:
        raise ValueError("PVC Utonia mask has an invalid shape or dtype")
    if not torch.isfinite(features).all():
        raise ValueError("PVC Utonia features must be finite")
    return features.contiguous(), mask.contiguous()
