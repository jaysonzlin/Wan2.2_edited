# Kubric float RGB support for Utonia PC training

## Goal

Allow the Utonia-conditioned point-cloud training experiment to consume the
RGB representation used by `kubric/td_832x480_3_soft`: `float32 [2048, 3]`
with finite values in `[0, 1]`.

## Design

The object-mode dataset and cache source reader will accept either:

- `uint8 [2048, 3]` in `[0, 255]`; or
- floating-point `[2048, 3]` in `[0, 1]`.

They preserve the stored values and dtype for source fingerprints. The real
`UtoniaFeatureExtractor` alone converts accepted unit-range floating RGB to
`float32 [0, 255]` immediately before Utonia's existing `NormalizeColor`
transform. This keeps Utonia's resulting normalized colors identical to an
equivalent uint8 source and leaves the HDF5 files untouched.

Invalid shapes, non-finite values, unsupported dtypes, and out-of-range values
continue to fail early. Tests cover float RGB acceptance, rejection, cache
reuse, and the 255-scale extractor boundary with an injectable preprocessing
seam; CUDA/Utonia inference is not required for the tests.
