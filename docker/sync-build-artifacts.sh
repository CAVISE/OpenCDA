#!/usr/bin/env bash

set -euo pipefail

readonly artifact_root="${OPENCDA_ARTIFACT_ROOT:-/opt/opencda-artifacts}"
readonly workspace="${OPENCDA_WORKSPACE:-${HOME}/cavise/opencda}"
readonly native_components="${OPENCDA_NATIVE_COMPONENTS:-}"
readonly protobuf_destination="${workspace}/opencda/core/common/communication/protos/cavise"

sync_protobuf_artifacts() {
    local component_root="${artifact_root}/protobuf"
    local manifest="${component_root}/protobuf-artifacts.manifest"
    local relative_path
    local -a artifact_paths
    local -a installed_artifacts

    if [[ ! -s "${manifest}" ]]; then
        echo "Missing or empty protobuf artifact manifest: ${manifest}" >&2
        exit 1
    fi

    mapfile -t artifact_paths < <(sed "/^[[:space:]]*$/d" "${manifest}")
    mapfile -d "" installed_artifacts < <(
        find "${component_root}" -type f \
            \( -name "*_pb2.py" -o -name "*_pb2.pyi" \) -print0
    )

    if [[ "${#artifact_paths[@]}" -eq 0 || \
          "${#artifact_paths[@]}" -ne "${#installed_artifacts[@]}" ]]; then
        echo "Protobuf artifact count does not match its manifest" >&2
        exit 1
    fi

    mkdir -p "${protobuf_destination}"
    find "${protobuf_destination}" -maxdepth 1 -type f \
        \( -name "*_pb2.py" -o -name "*_pb2.pyi" \) -delete

    for relative_path in "${artifact_paths[@]}"; do
        if [[ "${relative_path}" == /* || "${relative_path}" == ".." || \
              "${relative_path}" == ../* || "${relative_path}" == */../* || \
              "${relative_path}" == */.. || \
              ! -f "${component_root}/${relative_path}" ]]; then
            echo "Invalid or missing protobuf artifact: ${relative_path}" >&2
            exit 1
        fi

        mkdir -p "${workspace}/${relative_path%/*}"
        cp "${component_root}/${relative_path}" "${workspace}/${relative_path}"
    done
}

declare -a components=()
if [[ -n "${native_components}" ]]; then
    read -r -a components <<< "${native_components}"
fi

for component in "${components[@]}"; do
    if [[ "${component}" != "protobuf" ]]; then
        echo "Unsupported OpenCDA native component: ${component}" >&2
        exit 1
    fi
    sync_protobuf_artifacts
done

exec "$@"
