cmake_minimum_required(VERSION 3.31.11...4.3.1)

project(OpenCDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(OPENCDA_BUILD_CUDA "Build CUDA extensions" ON)
option(OPENCDA_BUILD_CYTHON "Build Cython extensions" ON)
option(OPENCDA_BUILD_PROTOBUF "Build Protobuf files" ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86")
endif()

find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development NumPy)

list(APPEND CMAKE_MODULE_PATH "${Python3_SITEARCH}/cython_cmake/cmake")

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

find_package(pybind11 REQUIRED CONFIG
    HINTS "${Python3_SITELIB}/pybind11/share/cmake"
          "${Python3_SITEARCH}/pybind11/share/cmake"
)

if(OPENCDA_BUILD_PROTOBUF)
    add_subdirectory(opencda/core/common/communication)
endif()

if(OPENCDA_BUILD_CUDA)
    add_subdirectory(OpenCOOD/opencood/pcdet_utils)
endif()

if(OPENCDA_BUILD_CYTHON)
    add_subdirectory(OpenCOOD/opencood/utils)
endif()
