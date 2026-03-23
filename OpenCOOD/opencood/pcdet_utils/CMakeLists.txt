set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Enable relaxed constexpr for CUDA kernel code (required for std::complex and similar types)
if(NOT CMAKE_CUDA_FLAGS MATCHES "--expt-relaxed-constexpr")
    string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")
endif()

# Fast math improves CUDA performance significantly for these kernels
# Can be disabled by setting CMAKE_CUDA_FLAGS without this flag
if(NOT CMAKE_CUDA_FLAGS MATCHES "--use_fast_math")
    string(APPEND CMAKE_CUDA_FLAGS " --use_fast_math")
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
find_package(Torch REQUIRED)

# iou3d_nms_cuda
set(IOU3D_NMS_SOURCES
    iou3d_nms/src/iou3d_cpu.cpp
    iou3d_nms/src/iou3d_nms_api.cpp
    iou3d_nms/src/iou3d_nms.cpp
    iou3d_nms/src/iou3d_nms_kernel.cu
)
pybind11_add_module(iou3d_nms_cuda ${IOU3D_NMS_SOURCES})
target_link_libraries(iou3d_nms_cuda PRIVATE ${TORCH_LIBRARIES})
target_compile_definitions(iou3d_nms_cuda PRIVATE TORCH_EXTENSION_NAME=iou3d_nms_cuda)
set_target_properties(iou3d_nms_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# roiaware_pool3d_cuda
set(ROIAWARE_POOL3D_SOURCES
    roiaware_pool3d/src/roiaware_pool3d.cpp
    roiaware_pool3d/src/roiaware_pool3d_kernel.cu
)
pybind11_add_module(roiaware_pool3d_cuda ${ROIAWARE_POOL3D_SOURCES})
target_link_libraries(roiaware_pool3d_cuda PRIVATE ${TORCH_LIBRARIES})
target_compile_definitions(roiaware_pool3d_cuda PRIVATE TORCH_EXTENSION_NAME=roiaware_pool3d_cuda)
set_target_properties(roiaware_pool3d_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

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
target_link_libraries(pointnet2_stack_cuda PRIVATE ${TORCH_LIBRARIES})
target_compile_definitions(pointnet2_stack_cuda PRIVATE TORCH_EXTENSION_NAME=pointnet2_stack_cuda)
set_target_properties(pointnet2_stack_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

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
target_link_libraries(pointnet2_batch_cuda PRIVATE ${TORCH_LIBRARIES})
target_compile_definitions(pointnet2_batch_cuda PRIVATE TORCH_EXTENSION_NAME=pointnet2_batch_cuda)
set_target_properties(pointnet2_batch_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

install(TARGETS iou3d_nms_cuda roiaware_pool3d_cuda pointnet2_stack_cuda pointnet2_batch_cuda
    LIBRARY DESTINATION opencood/pcdet_utils
)