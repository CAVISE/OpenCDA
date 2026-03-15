set(PROTO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROTO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(PROTO_INSTALL_DIR ${CMAKE_SOURCE_DIR}/opencda/core/common/communication/protos/cavise)

set(PROTO_FILES capi.proto)

file(MAKE_DIRECTORY ${PROTO_BINARY_DIR})
file(MAKE_DIRECTORY ${PROTO_INSTALL_DIR})

set(PROTO_PY_OUTPUTS "")
foreach(PROTO_FILE ${PROTO_FILES})
    get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
    list(APPEND PROTO_PY_OUTPUTS ${PROTO_BINARY_DIR}/${PROTO_NAME}_pb2.py)
endforeach()

add_custom_command(
    OUTPUT ${PROTO_PY_OUTPUTS}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        --proto_path=${PROTO_SOURCE_DIR}
        --python_out=${PROTO_BINARY_DIR}
        ${PROTO_FILES}
    DEPENDS ${PROTO_SOURCE_DIR}/capi.proto
    WORKING_DIRECTORY ${PROTO_SOURCE_DIR}
    VERBATIM
)

add_custom_target(opencda_protos ALL DEPENDS ${PROTO_PY_OUTPUTS})

add_custom_command(TARGET opencda_protos POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROTO_BINARY_DIR} ${PROTO_INSTALL_DIR}
)

install(FILES ${PROTO_PY_OUTPUTS} DESTINATION opencda/core/common/communication/protos/cavise)