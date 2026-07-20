# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Lazy exports so lightweight Wan modules can run without a CUDA T5 import."""

from importlib import import_module

__all__ = [
    "Wan2_1_VAE", "Wan2_2_VAE", "WanModel", "T5Model", "T5Encoder",
    "T5Decoder", "T5EncoderModel", "HuggingfaceTokenizer", "flash_attention",
    "PCTrajectoryModel",
]

_LAZY_EXPORTS = {
    "flash_attention": (".attention", "flash_attention"),
    "WanModel": (".model", "WanModel"),
    "T5Model": (".t5", "T5Model"),
    "T5Encoder": (".t5", "T5Encoder"),
    "T5Decoder": (".t5", "T5Decoder"),
    "T5EncoderModel": (".t5", "T5EncoderModel"),
    "HuggingfaceTokenizer": (".tokenizers", "HuggingfaceTokenizer"),
    "Wan2_1_VAE": (".vae2_1", "Wan2_1_VAE"),
    "Wan2_2_VAE": (".vae2_2", "Wan2_2_VAE"),
    "PCTrajectoryModel": (".pc_trajectory", "PCTrajectoryModel"),
}


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
