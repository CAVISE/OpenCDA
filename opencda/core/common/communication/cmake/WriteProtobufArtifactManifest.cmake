file(
    GLOB_RECURSE installed_protobuf_artifacts
    RELATIVE "${CMAKE_INSTALL_PREFIX}"
    "${CMAKE_INSTALL_PREFIX}/opencda/core/common/communication/protos/cavise/*_pb2.py"
    "${CMAKE_INSTALL_PREFIX}/opencda/core/common/communication/protos/cavise/*_pb2.pyi"
)
list(SORT installed_protobuf_artifacts)

string(REPLACE ";" "\n" protobuf_manifest "${installed_protobuf_artifacts}")
if(NOT protobuf_manifest STREQUAL "")
    string(APPEND protobuf_manifest "\n")
endif()

file(
    WRITE
    "${CMAKE_INSTALL_PREFIX}/protobuf-artifacts.manifest"
    "${protobuf_manifest}"
)
