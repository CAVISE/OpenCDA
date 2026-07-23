file(
    GLOB_RECURSE installed_cuda_artifacts
    RELATIVE "${CMAKE_INSTALL_PREFIX}"
    "${CMAKE_INSTALL_PREFIX}/OpenCOOD/opencood/pcdet_utils/*_cuda*.so"
)
list(SORT installed_cuda_artifacts)

string(REPLACE ";" "\n" cuda_manifest "${installed_cuda_artifacts}")
if(NOT cuda_manifest STREQUAL "")
    string(APPEND cuda_manifest "\n")
endif()

file(
    WRITE
    "${CMAKE_INSTALL_PREFIX}/cuda-artifacts.manifest"
    "${cuda_manifest}"
)
