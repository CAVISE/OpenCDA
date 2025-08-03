#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "sampling_gpu.h"


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int furthest_point_sampling_wrapper(int b, int n, int m,
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

    CHECK_INPUT(points_tensor);
    CHECK_INPUT(temp_tensor);
    CHECK_INPUT(idx_tensor);

    const float* points = points_tensor.data_ptr<float>();
    float* temp = temp_tensor.data_ptr<float>();
    int* idx = idx_tensor.data_ptr<int>();

    furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx);
    return 1;
}
