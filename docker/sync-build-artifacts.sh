#!/usr/bin/env bash

set -euo pipefail

readonly artifact_root="${OPENCDA_ARTIFACT_ROOT:-/opt/opencda-artifacts}"
readonly workspace="${OPENCDA_WORKSPACE:-${HOME}/cavise/opencda}"
readonly native_components="${OPENCDA_NATIVE_COMPONENTS:-}"
readonly protobuf_destination="${workspace}/opencda/core/common/communication/protos/cavise"
readonly cuda_destination="${workspace}/OpenCOOD/opencood/pcdet_utils"

load_artifact_manifest() {
    local component="$1"
    local result_name="$2"
    local component_root="${artifact_root}/${component}"
    local manifest="${component_root}/${component}-artifacts.manifest"
    local -n result="${result_name}"
    local -a installed_artifacts

    if [[ ! -s "${manifest}" ]]; then
        echo "Missing or empty ${component} artifact manifest: ${manifest}" >&2
        exit 1
    fi

    mapfile -t result < <(sed "/^[[:space:]]*$/d" "${manifest}")
    if [[ "${#result[@]}" -eq 0 ]]; then
        echo "The ${component} artifact manifest contains no files" >&2
        exit 1
    fi

    case "${component}" in
        protobuf)
            mapfile -d "" installed_artifacts < <(
                find "${component_root}" -type f \
                    \( -name "*_pb2.py" -o -name "*_pb2.pyi" \) -print0
            )
            ;;
        cuda)
            mapfile -d "" installed_artifacts < <(
                find "${component_root}" -type f -name "*_cuda*.so" -print0
            )
            ;;
        *)
            echo "Unsupported artifact component: ${component}" >&2
            exit 1
            ;;
    esac

    if [[ "${#result[@]}" -ne "${#installed_artifacts[@]}" ]]; then
        echo "${component} artifact count does not match its manifest: expected ${#result[@]}, found ${#installed_artifacts[@]}" >&2
        exit 1
    fi

    local relative_path
    for relative_path in "${result[@]}"; do
        if [[ "${relative_path}" == /* || "${relative_path}" == ".." || \
              "${relative_path}" == ../* || "${relative_path}" == */../* || \
              "${relative_path}" == */.. ]]; then
            echo "Invalid path in ${component} artifact manifest: ${relative_path}" >&2
            exit 1
        fi

        if [[ ! -f "${component_root}/${relative_path}" ]]; then
            echo "Missing ${component} artifact: ${relative_path}" >&2
            exit 1
        fi
    done
}

sync_artifacts() {
    local component="$1"
    local artifacts_name="$2"
    local component_root="${artifact_root}/${component}"
    local -n artifact_paths="${artifacts_name}"
    local relative_path
    local destination

    for relative_path in "${artifact_paths[@]}"; do
        destination="${workspace}/${relative_path}"
        mkdir -p "${destination%/*}"
        cp "${component_root}/${relative_path}" "${destination}"
    done
}

clean_workspace_artifacts() {
    local component="$1"

    case "${component}" in
        protobuf)
            mkdir -p "${protobuf_destination}"
            find "${protobuf_destination}" -maxdepth 1 -type f \
                \( -name "*_pb2.py" -o -name "*_pb2.pyi" \) -delete
            ;;
        cuda)
            if [[ -d "${cuda_destination}" ]]; then
                find "${cuda_destination}" -type f -name "*_cuda*.so" -delete
            fi
            ;;
        *)
            echo "Unsupported native component: ${component}" >&2
            exit 1
            ;;
    esac
}

sync_component() {
    local component="$1"
    local -a artifacts

    case "${component}" in
        protobuf|cuda)
            ;;
        *)
            echo "Unsupported native component: ${component}" >&2
            exit 1
            ;;
    esac

    load_artifact_manifest "${component}" artifacts
    clean_workspace_artifacts "${component}"
    sync_artifacts "${component}" artifacts
}

declare -a components=()
if [[ -n "${native_components}" ]]; then
    read -r -a components <<< "${native_components}"
fi

declare -A synchronized_components=()
for component in "${components[@]}"; do
    if [[ -n "${synchronized_components[${component}]:-}" ]]; then
        continue
    fi

    sync_component "${component}"
    synchronized_components["${component}"]=1
done

exec "$@"
