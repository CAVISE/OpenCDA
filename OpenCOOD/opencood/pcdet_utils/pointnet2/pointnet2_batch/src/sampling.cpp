/*
Batch version of point sampling and gathering, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/

#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#include "sampling_gpu.h"


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


int gather_points_wrapper_fast(int b, int c, int n, int npoints,
    at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor){

    CHECK_INPUT(points_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(out_tensor);

    const float* points = points_tensor.data_ptr<float>();
    const int* idx = idx_tensor.data_ptr<int>();
    float* out = out_tensor.data_ptr<float>();

    gather_points_kernel_launcher_fast(b, c, n, npoints, points, idx, out);
    return 1;
}


int gather_points_grad_wrapper_fast(int b, int c, int n, int npoints,
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor) {

    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(grad_points_tensor);

    const float* grad_out = grad_out_tensor.data_ptr<float>();
    const int* idx = idx_tensor.data_ptr<int>();
    float* grad_points = grad_points_tensor.data_ptr<float>();

    gather_points_grad_kernel_launcher_fast(b, c, n, npoints, grad_out, idx, grad_points);
    return 1;
}


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
