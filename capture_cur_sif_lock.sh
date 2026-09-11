#!/usr/bin/env bash
# Capture the exact Conda and pip distributions installed in cur.sif.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SIF_IMAGE="${SIF_IMAGE:-${PROJECT_DIR}/cur.sif}"
SIF_LOCK_DIR="${SIF_LOCK_DIR:-${PROJECT_DIR}/locks/cur-sif}"
SINGULARITY_EXE="${SINGULARITY_EXE:-singularity}"

if [[ ! -f "${SIF_IMAGE}" ]]; then
    echo "SIF image was not found: ${SIF_IMAGE}" >&2
    exit 1
fi

if ! command -v "${SINGULARITY_EXE}" >/dev/null 2>&1; then
    echo "Singularity executable '${SINGULARITY_EXE}' was not found." >&2
    exit 1
fi

mkdir -p "${SIF_LOCK_DIR}"
temporary_dir=$(mktemp -d "${SIF_LOCK_DIR}/.capture.XXXXXX")
trap 'rm -rf "${temporary_dir}"' EXIT

"${SINGULARITY_EXE}" exec --nv "${SIF_IMAGE}" \
    /opt/conda/bin/conda list -p /opt/conda/envs/app --explicit \
    > "${temporary_dir}/conda-explicit.txt"
"${SINGULARITY_EXE}" exec --nv "${SIF_IMAGE}" \
    /opt/conda/envs/app/bin/python -m pip list --format=freeze \
    > "${temporary_dir}/pip-list.txt"
"${SINGULARITY_EXE}" exec --nv "${SIF_IMAGE}" \
    /opt/conda/envs/app/bin/python -m pip freeze --all \
    > "${temporary_dir}/pip-freeze-raw.txt"
{
    # These are the two non-default indexes used to build current.def. The
    # frozen CUDA local-version wheels cannot be reliably rediscovered from
    # PyPI alone.
    printf '%s\n' '--extra-index-url https://download.pytorch.org/whl/cu124'
    printf '%s\n' '--find-links https://data.pyg.org/whl/torch-2.4.0+cu124.html'
    awk -F '==' '
        function canonical_name(name) {
            name = tolower(name)
            gsub(/[-_.]+/, "-", name)
            return name
        }
        NR == FNR {
            versions[canonical_name($1)] = $2
            next
        }
        / @ file:\/\// {
            name = $1
            sub(/ @ .*/, "", name)
            version = versions[canonical_name(name)]
            if (version == "") {
                print "Could not find a portable version for " name > "/dev/stderr"
                exit 1
            }
            print name "==" version
            next
        }
        { print }
    ' "${temporary_dir}/pip-list.txt" "${temporary_dir}/pip-freeze-raw.txt" | LC_ALL=C sort
} > "${temporary_dir}/pip-freeze.txt"
"${SINGULARITY_EXE}" exec --nv "${SIF_IMAGE}" \
    /opt/conda/envs/app/bin/python --version \
    > "${temporary_dir}/python-version.txt"

if command -v sha256sum >/dev/null 2>&1; then
    image_sha256=$(sha256sum "${SIF_IMAGE}" | awk '{print $1}')
else
    image_sha256=$(shasum -a 256 "${SIF_IMAGE}" | awk '{print $1}')
fi

{
    printf 'sif_image=%s\n' "${SIF_IMAGE}"
    printf 'image_sha256=%s\n' "${image_sha256}"
    printf 'python_version=%s\n' "$(<"${temporary_dir}/python-version.txt")"
    printf 'captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${temporary_dir}/provenance.txt"

mv "${temporary_dir}/conda-explicit.txt" "${SIF_LOCK_DIR}/conda-explicit.txt"
mv "${temporary_dir}/pip-freeze.txt" "${SIF_LOCK_DIR}/pip-freeze.txt"
mv "${temporary_dir}/provenance.txt" "${SIF_LOCK_DIR}/provenance.txt"

echo "Captured SIF dependency locks in ${SIF_LOCK_DIR}"
