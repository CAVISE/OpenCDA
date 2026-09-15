cmake_minimum_required(VERSION 3.31)

project(OpenCDA LANGUAGES NONE)

option(OPENCDA_BUILD_PROTOBUF "Generate OpenCDA protobuf Python modules" ON)

if(NOT OPENCDA_BUILD_PROTOBUF)
    message(FATAL_ERROR "No OpenCDA build component selected")
endif()

add_subdirectory(opencda/core/common/communication)
