find_program(OPENCDA_PROTOC_EXECUTABLE NAMES protoc REQUIRED)
find_program(OPENCDA_PROTOC_GEN_MYPY_EXECUTABLE NAMES protoc-gen-mypy REQUIRED)

set(OPENCDA_PROTO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/messages")
set(OPENCDA_PROTO_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

set(OPENCDA_PROTO_NAMES
    entity
    opencda
    artery
    capi
)

set(OPENCDA_PROTO_FILES)
foreach(proto_name IN LISTS OPENCDA_PROTO_NAMES)
    list(APPEND OPENCDA_PROTO_FILES "${OPENCDA_PROTO_SOURCE_DIR}/${proto_name}.proto")
endforeach()

set(OPENCDA_PROTO_OUTPUTS)
foreach(proto_name IN LISTS OPENCDA_PROTO_NAMES)
    set(proto_python_output "${OPENCDA_PROTO_BUILD_DIR}/${proto_name}_pb2.py")
    set(proto_mypy_output "${OPENCDA_PROTO_BUILD_DIR}/${proto_name}_pb2.pyi")

    add_custom_command(
        OUTPUT
            "${proto_python_output}"
            "${proto_mypy_output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${OPENCDA_PROTO_BUILD_DIR}"
        COMMAND
            "${OPENCDA_PROTOC_EXECUTABLE}"
            "--proto_path=${OPENCDA_PROTO_SOURCE_DIR}"
            "--python_out=${OPENCDA_PROTO_BUILD_DIR}"
            "--mypy_out=${OPENCDA_PROTO_BUILD_DIR}"
            "--plugin=protoc-gen-mypy=${OPENCDA_PROTOC_GEN_MYPY_EXECUTABLE}"
            "${OPENCDA_PROTO_SOURCE_DIR}/${proto_name}.proto"
        DEPENDS ${OPENCDA_PROTO_FILES}
        COMMENT "Generating Python protobuf module ${proto_name}_pb2"
        VERBATIM
    )

    list(APPEND OPENCDA_PROTO_OUTPUTS
        "${proto_python_output}"
        "${proto_mypy_output}"
    )
endforeach()

add_custom_target(opencda_protobuf ALL DEPENDS ${OPENCDA_PROTO_OUTPUTS})

install(
    FILES ${OPENCDA_PROTO_OUTPUTS}
    DESTINATION "opencda/core/common/communication/protos/cavise"
)

install(CODE [=[
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
]=])
