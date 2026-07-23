separate_arguments(OPENCDA_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

function(opencda_add_cuda_extension target package_directory)
    Python3_add_library(${target} MODULE WITH_SOABI ${ARGN})

    target_include_directories(${target} PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE
        ${TORCH_LIBRARIES}
        "${OPENCDA_TORCH_PYTHON_LIBRARY}"
        CUDA::cudart
    )
    target_compile_definitions(${target} PRIVATE
        TORCH_API_INCLUDE_EXTENSION_H
        TORCH_EXTENSION_NAME=${target}
    )
    target_compile_options(${target} PRIVATE
        ${OPENCDA_TORCH_CXX_FLAGS}
        $<$<COMPILE_LANGUAGE:CUDA>:-D__CUDA_NO_HALF_OPERATORS__>
        $<$<COMPILE_LANGUAGE:CUDA>:-D__CUDA_NO_HALF_CONVERSIONS__>
        $<$<COMPILE_LANGUAGE:CUDA>:-D__CUDA_NO_BFLOAT16_CONVERSIONS__>
        $<$<COMPILE_LANGUAGE:CUDA>:-D__CUDA_NO_HALF2_OPERATORS__>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    set_target_properties(${target} PROPERTIES
        CUDA_ARCHITECTURES OFF
        INSTALL_RPATH "${OPENCDA_TORCH_RUNTIME_LIBRARY_DIR}"
    )

    install(
        TARGETS ${target}
        LIBRARY DESTINATION "OpenCOOD/opencood/pcdet_utils/${package_directory}"
    )
endfunction()

opencda_add_cuda_extension(
    iou3d_nms_cuda
    iou3d_nms
    iou3d_nms/src/iou3d_cpu.cpp
    iou3d_nms/src/iou3d_nms_api.cpp
    iou3d_nms/src/iou3d_nms.cpp
    iou3d_nms/src/iou3d_nms_kernel.cu
)

opencda_add_cuda_extension(
    roiaware_pool3d_cuda
    roiaware_pool3d
    roiaware_pool3d/src/roiaware_pool3d.cpp
    roiaware_pool3d/src/roiaware_pool3d_kernel.cu
)

opencda_add_cuda_extension(
    pointnet2_stack_cuda
    pointnet2/pointnet2_stack
    pointnet2/pointnet2_stack/src/pointnet2_api.cpp
    pointnet2/pointnet2_stack/src/ball_query.cpp
    pointnet2/pointnet2_stack/src/ball_query_gpu.cu
    pointnet2/pointnet2_stack/src/group_points.cpp
    pointnet2/pointnet2_stack/src/group_points_gpu.cu
    pointnet2/pointnet2_stack/src/sampling.cpp
    pointnet2/pointnet2_stack/src/sampling_gpu.cu
    pointnet2/pointnet2_stack/src/interpolate.cpp
    pointnet2/pointnet2_stack/src/interpolate_gpu.cu
)

opencda_add_cuda_extension(
    pointnet2_batch_cuda
    pointnet2/pointnet2_batch
    pointnet2/pointnet2_batch/src/pointnet2_api.cpp
    pointnet2/pointnet2_batch/src/ball_query.cpp
    pointnet2/pointnet2_batch/src/ball_query_gpu.cu
    pointnet2/pointnet2_batch/src/group_points.cpp
    pointnet2/pointnet2_batch/src/group_points_gpu.cu
    pointnet2/pointnet2_batch/src/interpolate.cpp
    pointnet2/pointnet2_batch/src/interpolate_gpu.cu
    pointnet2/pointnet2_batch/src/sampling.cpp
    pointnet2/pointnet2_batch/src/sampling_gpu.cu
)

install(CODE [=[
    file(
        GLOB_RECURSE installed_cuda_artifacts
        RELATIVE "${CMAKE_INSTALL_PREFIX}"
        "${CMAKE_INSTALL_PREFIX}/OpenCOOD/opencood/pcdet_utils/*_cuda*.so"
    )
    list(SORT installed_cuda_artifacts)
    string(REPLACE ";" "\n" cuda_manifest "${installed_cuda_artifacts}")
    if(NOT cuda_manifest STREQUAL "")
        string(APPEND cuda_manifest "\n")
    endif()
    file(
        WRITE
        "${CMAKE_INSTALL_PREFIX}/cuda-artifacts.manifest"
        "${cuda_manifest}"
    )
]=])
