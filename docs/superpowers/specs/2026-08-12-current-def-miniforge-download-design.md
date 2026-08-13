# Download Miniforge during the current container build

## Goal

Remove the external Miniforge installer build-context requirement from
`current.def` by downloading the official installer during `%post`.

## Design

Delete the `%files` entry that copies
`./Miniforge3-Linux-x86_64.sh` to `/opt/miniforge.sh`. After system packages
are installed, download the same official Miniforge Linux x86_64 installer to
`/tmp/miniforge.sh`, install it into `/opt/conda`, and remove the temporary
file. This matches the lifecycle in `Utonia/utonia.def`.

No Python, CUDA, Utonia, model-weight, or runtime-environment pins change. A
static test verifies the definition no longer references the build-context
installer and contains the download, install, and cleanup commands.
