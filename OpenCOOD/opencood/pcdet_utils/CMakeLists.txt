set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math")

set(CUDA_COMMON_INCLUDES
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# iou3d_nms_cuda
set(IOU3D_NMS_SOURCES
    iou3d_nms/src/iou3d_cpu.cpp
    iou3d_nms/src/iou3d_nms_api.cpp
    iou3d_nms/src/iou3d_nms.cpp
    iou3d_nms/src/iou3d_nms_kernel.cu
)
pybind11_add_module(iou3d_nms_cuda ${IOU3D_NMS_SOURCES})
target_include_directories(iou3d_nms_cuda PRIVATE ${CUDA_COMMON_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/iou3d_nms/src)
target_compile_options(iou3d_nms_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=sm_86> $<$<COMPILE_LANGUAGE:CXX>:-O3>)

# roiaware_pool3d_cuda
set(ROIAWARE_POOL3D_SOURCES
    roiaware_pool3d/src/roiaware_pool3d.cpp
    roiaware_pool3d/src/roiaware_pool3d_kernel.cu
)
pybind11_add_module(roiaware_pool3d_cuda ${ROIAWARE_POOL3D_SOURCES})
target_include_directories(roiaware_pool3d_cuda PRIVATE ${CUDA_COMMON_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/roiaware_pool3d/src)

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
target_include_directories(pointnet2_stack_cuda PRIVATE ${CUDA_COMMON_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/pointnet2/pointnet2_stack/src)

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
target_include_directories(pointnet2_batch_cuda PRIVATE ${CUDA_COMMON_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/pointnet2/pointnet2_batch/src)

install(TARGETS iou3d_nms_cuda roiaware_pool3d_cuda pointnet2_stack_cuda pointnet2_batch_cuda
    LIBRARY DESTINATION opencood/pcdet_utils
)