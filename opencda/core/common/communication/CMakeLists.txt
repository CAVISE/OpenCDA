find_package(Protobuf REQUIRED)

find_program(PROTOC_GEN_MYPY NAMES protoc-gen-mypy REQUIRED)

set(PROTO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/messages)
set(PROTO_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/protos/cavise)

set(PROTO_FILES
    ${PROTO_SOURCE_DIR}/capi.proto
    ${PROTO_SOURCE_DIR}/opencda.proto
    ${PROTO_SOURCE_DIR}/artery.proto
    ${PROTO_SOURCE_DIR}/entity.proto
)

protobuf_generate(
    LANGUAGE python
    GENERATE_EXTENSIONS _pb2.py
    OUT_VAR PROTO_PY_OUTPUTS
    PROTOS ${PROTO_FILES}
    IMPORT_DIRS ${PROTO_SOURCE_DIR}
    PROTOC_OUT_DIR ${PROTO_INSTALL_DIR}
)

protobuf_generate(
    LANGUAGE mypy
    PLUGIN ${PROTOC_GEN_MYPY}
    GENERATE_EXTENSIONS _pb2.pyi
    OUT_VAR PROTO_PYI_OUTPUTS
    PROTOS ${PROTO_FILES}
    IMPORT_DIRS ${PROTO_SOURCE_DIR}
    PROTOC_OUT_DIR ${PROTO_INSTALL_DIR}
)

add_custom_target(opencda_protos ALL DEPENDS ${PROTO_PY_OUTPUTS} ${PROTO_PYI_OUTPUTS})

install(FILES ${PROTO_PY_OUTPUTS} ${PROTO_PYI_OUTPUTS}
    DESTINATION opencda/core/common/communication/protos/cavise
)
