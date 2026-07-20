# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan package with lazy heavy-model exports.

Point-cloud helpers must remain importable on CPU-only machines.  The video
entry points load T5 at module import time, so defer them until requested.
"""

from importlib import import_module

from . import configs, distributed

__all__ = ["configs", "distributed", "modules", "WanI2V", "WanT2V", "WanTI2V"]

_LAZY_EXPORTS = {
    "modules": (".modules", None),
    "WanI2V": (".image2video", "WanI2V"),
    "WanT2V": (".text2video", "WanT2V"),
    "WanTI2V": (".textimage2video", "WanTI2V"),
}


def __getattr__(name):
    """Load video-model modules only when their public export is requested."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    module = import_module(module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value
