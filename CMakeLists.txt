cmake_minimum_required(VERSION 3.31)

project(OpenCDA LANGUAGES NONE)

option(OPENCDA_BUILD_CUDA "Build OpenCOOD CUDA extensions" OFF)
option(OPENCDA_BUILD_PROTOBUF "Generate OpenCDA protobuf Python modules" OFF)

if(NOT OPENCDA_BUILD_CUDA AND NOT OPENCDA_BUILD_PROTOBUF)
    message(FATAL_ERROR
        "No build component selected. Enable OPENCDA_BUILD_CUDA or "
        "OPENCDA_BUILD_PROTOBUF."
    )
endif()

if(OPENCDA_BUILD_PROTOBUF)
    add_subdirectory(opencda/core/common/communication)
endif()

if(OPENCDA_BUILD_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        message(FATAL_ERROR
            "CMAKE_CUDA_ARCHITECTURES must be set explicitly when building "
            "the OpenCOOD CUDA extensions."
        )
    endif()

    set(OPENCDA_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    set(OPENCDA_TORCH_CUDA_ARCH_LIST)
    foreach(cuda_architecture IN LISTS OPENCDA_CUDA_ARCHITECTURES)
        if(NOT cuda_architecture MATCHES "^[0-9]+$")
            message(FATAL_ERROR
                "Unsupported CUDA architecture '${cuda_architecture}'. "
                "Use numeric values such as 86 or 75;86;89."
            )
        endif()

        math(EXPR cuda_architecture_major "${cuda_architecture} / 10")
        math(EXPR cuda_architecture_minor "${cuda_architecture} % 10")
        list(APPEND OPENCDA_TORCH_CUDA_ARCH_LIST
            "${cuda_architecture_major}.${cuda_architecture_minor}"
        )
    endforeach()

    enable_language(CXX)
    enable_language(CUDA)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(CUDAToolkit REQUIRED)

    execute_process(
        COMMAND
            "${Python3_EXECUTABLE}" -c
            "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")

    # PyTorch's CMake package uses TORCH_CUDA_ARCH_LIST and otherwise adds a
    # broad architecture list that includes targets removed from recent CUDA
    # toolkits. Disable CMake's duplicate architecture flags and let PyTorch
    # emit the requested gencode flags.
    unset(CMAKE_CUDA_ARCHITECTURES CACHE)
    unset(CMAKE_CUDA_ARCHITECTURES)
    set(TORCH_CUDA_ARCH_LIST "${OPENCDA_TORCH_CUDA_ARCH_LIST}")
    message(STATUS
        "Building OpenCOOD CUDA extensions for ${TORCH_CUDA_ARCH_LIST}"
    )
    find_package(Torch REQUIRED)

    execute_process(
        COMMAND
            "${Python3_EXECUTABLE}" -c
            "from pathlib import Path; import torch; print(Path(torch.__file__).parent / 'lib')"
        OUTPUT_VARIABLE OPENCDA_DETECTED_TORCH_LIBRARY_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    set(
        OPENCDA_TORCH_RUNTIME_LIBRARY_DIR
        "${OPENCDA_DETECTED_TORCH_LIBRARY_DIR}"
        CACHE PATH
        "Runtime directory containing the PyTorch shared libraries"
    )
    find_library(
        OPENCDA_TORCH_PYTHON_LIBRARY
        NAMES torch_python
        PATHS "${OPENCDA_DETECTED_TORCH_LIBRARY_DIR}"
        NO_DEFAULT_PATH
        REQUIRED
    )

    add_subdirectory(OpenCOOD/opencood/pcdet_utils)
endif()
