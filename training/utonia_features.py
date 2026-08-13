"""Persistent dense Utonia feature caching for point-cloud trajectory training."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile
from typing import Callable, Mapping, Protocol

import h5py
import numpy as np
import torch


CACHE_VERSION = "utonia-pc-features-v1"


class FeatureExtractor(Protocol):
    checkpoint_fingerprint: str
    preprocess_version: str

    def __call__(self, coordinates: np.ndarray, rgb: np.ndarray) -> torch.Tensor:
        """Return dense features aligned with the input point ordering."""


def _cache_path(cache_root: Path, sample_id: str) -> Path:
    parts = Path(sample_id).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid cache sample id: {sample_id!r}")
    return cache_root.joinpath(*parts, "utonia_features.pt")


def _source_fingerprint(coordinates: np.ndarray, rgb: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (coordinates, rgb):
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def validate_utonia_rgb(rgb: np.ndarray, *, point_count: int = 2048) -> None:
    """Validate RGB stored by a supported Utonia point-cloud source."""
    if rgb.shape != (point_count, 3):
        raise ValueError(f"rgb must have shape ({point_count}, 3)")
    if rgb.dtype == np.uint8:
        return
    if not np.issubdtype(rgb.dtype, np.floating):
        raise ValueError("rgb must have dtype uint8 or a floating-point dtype")
    if not np.isfinite(rgb).all():
        raise ValueError("rgb floating-point values must be finite")
    if rgb.min() < 0.0 or rgb.max() > 1.0:
        raise ValueError("rgb floating-point values must be in [0, 1]")


def prepare_utonia_color(rgb: np.ndarray) -> np.ndarray:
    """Return stored RGB on Utonia's expected pre-NormalizeColor scale."""
    validate_utonia_rgb(rgb, point_count=rgb.shape[0])
    color = np.asarray(rgb, dtype=np.float32)
    return color if rgb.dtype == np.uint8 else color * 255.0


def _read_source(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with h5py.File(path, "r") as source:
            coordinates = np.asarray(source["point_cloud"][0, 0], dtype=np.float32)
            rgb = np.asarray(source["rgb"][:])
    except (KeyError, OSError, IndexError) as error:
        raise ValueError(f"Unable to read Utonia source data from {path}") from error
    if coordinates.shape != (2048, 3):
        raise ValueError(f"{path}: frame-zero coordinates must have shape (2048, 3)")
    try:
        validate_utonia_rgb(rgb)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    return coordinates, rgb


def _torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility for older PyTorch installs
        return torch.load(path, map_location="cpu")


def _validate_record(
    record: object,
    *,
    sample_id: str | None = None,
    source_fingerprint: str | None = None,
    checkpoint_fingerprint: str | None = None,
    preprocess_version: str | None = None,
) -> torch.Tensor:
    if not isinstance(record, dict):
        raise ValueError("Utonia cache entry must be a dictionary")
    metadata = record.get("metadata")
    features = record.get("features")
    if not isinstance(metadata, dict) or not isinstance(features, torch.Tensor):
        raise ValueError("Utonia cache entry is missing metadata or features")
    if metadata.get("cache_version") != CACHE_VERSION:
        raise ValueError("Utonia cache version is stale")
    expected_fields = {
        "sample_id": sample_id,
        "source_fingerprint": source_fingerprint,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "preprocess_version": preprocess_version,
    }
    for field, expected in expected_fields.items():
        if expected is not None and metadata.get(field) != expected:
            raise ValueError(f"Utonia cache {field} is stale")
    if (
        features.ndim != 2
        or features.shape[0] != 2048
        or features.shape[1] <= 0
        or features.dtype != torch.float32
        or not torch.isfinite(features).all()
    ):
        raise ValueError("Utonia features must have shape (2048, D), dtype float32, and be finite")
    if metadata.get("point_count") != 2048 or metadata.get("feature_dim") != features.shape[1]:
        raise ValueError("Utonia cache metadata does not match feature tensor")
    return features.contiguous()


def load_cached_utonia_features(cache_root: str | Path, sample_id: str) -> torch.Tensor:
    """Load a locally cached dense feature tensor after structural validation."""
    path = _cache_path(Path(cache_root), sample_id)
    if not path.is_file():
        raise ValueError(f"Missing Utonia cache entry: {path}")
    try:
        return _validate_record(_torch_load(path))
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid Utonia cache entry: {path}") from error


def _matching_features(
    cache_path: Path,
    *,
    sample_id: str,
    source_fingerprint: str,
    extractor: FeatureExtractor,
) -> torch.Tensor | None:
    if not cache_path.is_file():
        return None
    try:
        return _validate_record(
            _torch_load(cache_path),
            sample_id=sample_id,
            source_fingerprint=source_fingerprint,
            checkpoint_fingerprint=extractor.checkpoint_fingerprint,
            preprocess_version=extractor.preprocess_version,
        )
    except (OSError, RuntimeError, ValueError):
        return None


def _write_record(path: Path, metadata: dict, features: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save({"metadata": metadata, "features": features}, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def prepare_utonia_feature_cache(
    sources: Mapping[str, str | Path], cache_root: str | Path, extractor: FeatureExtractor
) -> int:
    """Build/reuse validated dense feature entries and return their common width."""
    if not sources:
        raise ValueError("At least one object source is required for Utonia caching")
    widths: set[int] = set()
    root = Path(cache_root)
    for sample_id, raw_path in sorted(sources.items()):
        coordinates, rgb = _read_source(Path(raw_path))
        fingerprint = _source_fingerprint(coordinates, rgb)
        path = _cache_path(root, sample_id)
        features = _matching_features(
            path,
            sample_id=sample_id,
            source_fingerprint=fingerprint,
            extractor=extractor,
        )
        if features is None:
            features = extractor(coordinates, rgb)
            if not isinstance(features, torch.Tensor):
                raise ValueError("Utonia extractor must return a torch.Tensor")
            features = features.detach().to(device="cpu", dtype=torch.float32).contiguous()
            if features.ndim != 2 or features.shape[0] != 2048 or features.shape[1] <= 0:
                raise ValueError("Utonia features must have shape (2048, D)")
            if not torch.isfinite(features).all():
                raise ValueError("Utonia features must be finite")
            metadata = {
                "cache_version": CACHE_VERSION,
                "sample_id": sample_id,
                "source_fingerprint": fingerprint,
                "checkpoint_fingerprint": extractor.checkpoint_fingerprint,
                "preprocess_version": extractor.preprocess_version,
                "point_count": 2048,
                "feature_dim": features.shape[1],
            }
            _write_record(path, metadata, features)
        widths.add(features.shape[1])
    if len(widths) != 1:
        raise ValueError(f"Utonia feature width differs across cache entries: {sorted(widths)}")
    return widths.pop()


@contextmanager
def _numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class UtoniaFeatureExtractor:
    """Lazy CUDA-backed extractor using Utonia's official Hugging Face loader."""

    preprocess_version = "utonia-default-normalize-coord-grid-sample-v1"

    def __init__(self, cache_root: str | Path):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to prepare Utonia feature cache")
        try:
            import utonia
        except ImportError as error:  # pragma: no cover - depends on local CUDA setup
            raise RuntimeError(
                "Utonia must be installed before running the Utonia PC experiment"
            ) from error
        self._utonia = utonia
        checkpoint_root = Path(cache_root) / "_utonia_checkpoint"
        load_kwargs: dict = {}
        try:
            import flash_attn  # noqa: F401
        except ImportError:  # pragma: no cover - CUDA runtime dependent
            load_kwargs["custom_config"] = {
                "enc_patch_size": [1024 for _ in range(5)],
                "enable_flash": False,
            }
        self.model = utonia.model.load(
            "utonia",
            repo_id="Pointcept/Utonia",
            download_root=str(checkpoint_root),
            **load_kwargs,
        ).cuda().eval()
        self.transform = utonia.transform.default(
            scale=1.0, apply_z_positive=True, normalize_coord=True
        )
        checkpoint = checkpoint_root / "utonia.pth"
        self.checkpoint_fingerprint = _file_fingerprint(checkpoint)

    @torch.no_grad()
    def __call__(self, coordinates: np.ndarray, rgb: np.ndarray) -> torch.Tensor:
        seed = int(_source_fingerprint(coordinates, rgb)[:16], 16) % (2**32)
        point = {
            "coord": coordinates.copy(),
            "color": prepare_utonia_color(rgb),
            "normal": np.zeros_like(coordinates),
        }
        with _numpy_seed(seed):
            point = self.transform(point)
        point = {
            key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in point.items()
        }
        point = self.model(point)
        for _ in range(2):
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point:
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        return point.feat[point.inverse]


def _file_fingerprint(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Utonia checkpoint was not downloaded: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
