cmake_minimum_required(VERSION 3.18...3.28)

project(OpenCDA
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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
    find_package(Protobuf CONFIG REQUIRED)
endif()

if(OPENCDA_BUILD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
    else()
        set(OPENCDA_BUILD_CUDA OFF)
    endif()
endif()

find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

if(OPENCDA_BUILD_PROTOBUF)
    add_subdirectory(opencda/core/common/communication/messages)
endif()

if(OPENCDA_BUILD_CUDA AND CMAKE_CUDA_COMPILER)
    add_subdirectory(OpenCOOD/opencood/pcdet_utils)
endif()

if(OPENCDA_BUILD_CYTHON)
    add_subdirectory(OpenCOOD/opencood/utils)
endif()
