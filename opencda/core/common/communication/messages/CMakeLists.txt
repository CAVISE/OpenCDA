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

set(PROTO_PY_OUTPUTS "")
set(PROTO_PYI_OUTPUTS "")
foreach(PROTO_FILE ${PROTO_FILES})
    get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
    list(APPEND PROTO_PY_OUTPUTS ${PROTO_INSTALL_DIR}/${PROTO_NAME}_pb2.py)
    list(APPEND PROTO_PYI_OUTPUTS ${PROTO_INSTALL_DIR}/${PROTO_NAME}_pb2.pyi)
endforeach()

# Use grpc_tools.protoc for Python protobuf generation with mypy stubs
# Generates both .py and .pyi files for proper type checking support
add_custom_command(
    OUTPUT ${PROTO_PY_OUTPUTS} ${PROTO_PYI_OUTPUTS}
    COMMAND ${Python3_EXECUTABLE} -m grpc_tools.protoc
        -I${PROTO_SOURCE_DIR}
        --python_out=${PROTO_INSTALL_DIR}
        --mypy_out=${PROTO_INSTALL_DIR}
        ${PROTO_FILES}
    DEPENDS ${PROTO_FILES}
    WORKING_DIRECTORY ${PROTO_SOURCE_DIR}
    COMMENT "Generating Python protobuf files with mypy stubs"
    VERBATIM
)

add_custom_target(opencda_protos ALL DEPENDS ${PROTO_PY_OUTPUTS} ${PROTO_PYI_OUTPUTS})

install(FILES ${PROTO_PY_OUTPUTS} ${PROTO_PYI_OUTPUTS}
    DESTINATION opencda/core/common/communication/protos/cavise
)
