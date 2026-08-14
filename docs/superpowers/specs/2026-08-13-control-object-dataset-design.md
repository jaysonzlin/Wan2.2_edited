# Control Object-Dataset Design

## Goal

Run the Utonia-free control against the same object-level trajectory as the
Utonia-conditioned experiment.

## Behavior

`data.object_id` selects `sample_*/objects/<object_id>/pc.hdf5` independently
of `model.utonia_enabled`. Utonia remains responsible only for feature-cache
preparation and for supplying Utonia features to the model.

Configurations without `data.object_id` retain the legacy
`sample_*/pc.hdf5` dataset layout.

## Validation

Add a regression test that constructs the control configuration with
`model.utonia_enabled=false` and verifies the dataset factory receives
`object_id="000"` without requesting an Utonia feature cache.
