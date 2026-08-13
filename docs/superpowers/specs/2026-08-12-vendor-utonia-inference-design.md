# Vendor Utonia Inference Design

## Goal

Make the Utonia-conditioned point-cloud trainer self-contained in the Wan
repository. The container must not copy a sibling Utonia checkout or install
Utonia as an editable package.

## Scope

Vendor only the Utonia inference implementation used by
`training.utonia_features.UtoniaFeatureExtractor`:

- model loading and Point Transformer V3 modules;
- point structures, transforms, registries, utilities, and serialization;
- the Utonia Apache-2.0 license and existing source attribution headers.

Do not vendor Utonia's PCA or trajectory-analysis helpers; the Wan trainer does
not import or execute them.

## Architecture

The vendored source lives at `wan/utonia/` and is imported as `wan.utonia`.
Its internal relative imports remain intact. The feature extractor keeps the
same data flow:

1. Read frame-zero XYZ and RGB from the object HDF5 file.
2. Apply the vendored Utonia default transform.
3. Load official `Pointcept/Utonia` weights through Hugging Face into the
   existing feature-cache checkpoint directory.
4. Produce, validate, and persist frozen dense per-point features.

No model weights are committed or baked into the container.

## Container and Packaging

`current.def` removes both `../utonia /opt/Utonia` from `%files` and the
editable `/opt/Utonia` pip install. The temporary `setuptools<81` pin and
`--no-build-isolation` flag become unnecessary and are removed with that
install.

The CUDA runtime dependencies needed by the vendored modules remain: Torch
2.4/cu124, `torch-scatter`, `spconv-cu124`, `timm`, `addict`, `scipy`,
`huggingface_hub`, and the existing Flash Attention wheel.

Wan packaging is updated to discover `wan.utonia` and
`wan.utonia.serialization` when the project is installed, rather than only the
top-level `wan` package.

## Error Handling

The extractor continues to require CUDA and provides a clear error if the
vendored module cannot be imported. Weight-download and cache-validation
failures retain their existing behavior.

## Verification

- Static container tests assert no external Utonia copy or editable install is
  declared.
- A source-layout test asserts the required vendored inference modules and
  Apache-2.0 license are present while excluded PCA/trajectory helpers are not.
- The existing training and feature-cache test suite remains green.
- A container build from the repository root verifies the image no longer
  requires `../utonia`; a GPU run remains the final runtime verification of
  actual feature extraction.
