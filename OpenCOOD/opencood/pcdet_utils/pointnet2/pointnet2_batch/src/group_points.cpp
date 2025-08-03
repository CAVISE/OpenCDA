/*
Batch version of point grouping, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/

#include <torch/serialize/tensor.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>

#include "group_points_gpu.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


int group_points_grad_wrapper_fast(int b, int c, int n, int npoints, int nsample,
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor) {

    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(grad_points_tensor);

    float* grad_points = grad_points_tensor.data_ptr<float>();
    const int* idx = idx_tensor.data_ptr<int>();
    const float* grad_out = grad_out_tensor.data_ptr<float>();

    group_points_grad_kernel_launcher_fast(b, c, n, npoints, nsample, grad_out, idx, grad_points);
    return 1;
}


int group_points_wrapper_fast(int b, int c, int n, int npoints, int nsample,
    at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor) {

    CHECK_INPUT(points_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(out_tensor);

    const float* points = points_tensor.data_ptr<float>();
    const int* idx = idx_tensor.data_ptr<int>();
    float* out = out_tensor.data_ptr<float>();

    group_points_kernel_launcher_fast(b, c, n, npoints, nsample, points, idx, out);
    return 1;
}
