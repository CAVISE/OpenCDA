set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(Torch REQUIRED HINTS "${Python3_SITELIB}/torch/share/cmake")

function(opencda_configure_cuda_extension target package_dir)
    set_target_properties(${target} PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${package_dir}"
    )
    target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(${target} PRIVATE TORCH_EXTENSION_NAME=${target})
    install(TARGETS ${target} LIBRARY DESTINATION "opencood/pcdet_utils/${package_dir}")
endfunction()

# iou3d_nms_cuda
set(IOU3D_NMS_SOURCES
    iou3d_nms/src/iou3d_cpu.cpp
    iou3d_nms/src/iou3d_nms_api.cpp
    iou3d_nms/src/iou3d_nms.cpp
    iou3d_nms/src/iou3d_nms_kernel.cu
)
pybind11_add_module(iou3d_nms_cuda ${IOU3D_NMS_SOURCES})
opencda_configure_cuda_extension(iou3d_nms_cuda iou3d_nms)

# roiaware_pool3d_cuda
set(ROIAWARE_POOL3D_SOURCES
    roiaware_pool3d/src/roiaware_pool3d.cpp
    roiaware_pool3d/src/roiaware_pool3d_kernel.cu
)
pybind11_add_module(roiaware_pool3d_cuda ${ROIAWARE_POOL3D_SOURCES})
opencda_configure_cuda_extension(roiaware_pool3d_cuda roiaware_pool3d)

# pointnet2_stack_cuda
set(POINTNET2_STACK_SOURCES
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
pybind11_add_module(pointnet2_stack_cuda ${POINTNET2_STACK_SOURCES})
opencda_configure_cuda_extension(pointnet2_stack_cuda pointnet2/pointnet2_stack)

# pointnet2_batch_cuda
set(POINTNET2_BATCH_SOURCES
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
pybind11_add_module(pointnet2_batch_cuda ${POINTNET2_BATCH_SOURCES})
opencda_configure_cuda_extension(pointnet2_batch_cuda pointnet2/pointnet2_batch)
