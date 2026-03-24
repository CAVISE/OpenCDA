find_package(Protobuf REQUIRED)

find_program(PROTOC_GEN_MYPY NAMES protoc-gen-mypy REQUIRED)

set(PROTO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROTO_INSTALL_DIR ${CMAKE_SOURCE_DIR}/opencda/core/common/communication/protos/cavise)

file(MAKE_DIRECTORY ${PROTO_INSTALL_DIR})

set(PROTO_FILES
    capi.proto
    opencda.proto
    artery.proto
    entity.proto
    ack.proto
)

protobuf_generate(
    LANGUAGE python
    GENERATE_EXTENSIONS _pb2.py _pb2.pyi
    PLUGIN ${PROTOC_GEN_MYPY}
    OUT_VAR PROTO_OUTPUTS
    PROTOS ${PROTO_FILES}
    IMPORT_DIRS ${PROTO_SOURCE_DIR}
    PROTOC_OUT_DIR ${PROTO_INSTALL_DIR}
)

add_custom_target(opencda_protos ALL DEPENDS ${PROTO_OUTPUTS})

install(FILES ${PROTO_OUTPUTS}
    DESTINATION opencda/core/common/communication/protos/cavise
)
