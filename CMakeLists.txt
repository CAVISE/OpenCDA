cmake_minimum_required(VERSION 3.18...3.28)

project(OpenCDA
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(OPENCDA_BUILD_CUDA "Build CUDA extensions" ON)
option(OPENCDA_BUILD_CYTHON "Build Cython extensions" ON)
option(OPENCDA_BUILD_PROTOBUF "Build Protobuf files" ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")
endif()

find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development NumPy)

if(OPENCDA_BUILD_PROTOBUF)
    find_package(Protobuf REQUIRED)
endif()

if(OPENCDA_BUILD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
    else()
        message(FATAL_ERROR "CUDA compiler not found but OPENCDA_BUILD_CUDA is ON. "
            "Either install CUDA toolkit or set OPENCDA_BUILD_CUDA=OFF")
    endif()
endif()

# pybind11 must be installed via pip before building
find_package(pybind11 REQUIRED CONFIG)

if(OPENCDA_BUILD_PROTOBUF)
    add_subdirectory(opencda/core/common/communication/messages)
endif()

if(OPENCDA_BUILD_CUDA)
    add_subdirectory(OpenCOOD/opencood/pcdet_utils)
endif()

if(OPENCDA_BUILD_CYTHON)
    add_subdirectory(OpenCOOD/opencood/utils)
endif()
