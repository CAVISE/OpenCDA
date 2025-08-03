/*
Stacked-batch-data version of point interpolation, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <torch/serialize/tensor.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "interpolate_gpu.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void three_nn_wrapper_stack(at::Tensor unknown_tensor,
    at::Tensor unknown_batch_cnt_tensor, at::Tensor known_tensor,
    at::Tensor known_batch_cnt_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor) {

    CHECK_INPUT(unknown_tensor);
    CHECK_INPUT(unknown_batch_cnt_tensor);
    CHECK_INPUT(known_tensor);
    CHECK_INPUT(known_batch_cnt_tensor);
    CHECK_INPUT(dist2_tensor);
    CHECK_INPUT(idx_tensor);

    int batch_size = unknown_batch_cnt_tensor.size(0);
    int N = unknown_tensor.size(0);
    int M = known_tensor.size(0);

    const float* unknown = unknown_tensor.data_ptr<float>();
    const int* unknown_batch_cnt = unknown_batch_cnt_tensor.data_ptr<int>();
    const float* known = known_tensor.data_ptr<float>();
    const int* known_batch_cnt = known_batch_cnt_tensor.data_ptr<int>();
    float* dist2 = dist2_tensor.data_ptr<float>();
    int* idx = idx_tensor.data_ptr<int>();

    three_nn_kernel_launcher_stack(batch_size, N, M, unknown, unknown_batch_cnt, known, known_batch_cnt, dist2, idx);
}


void three_interpolate_wrapper_stack(at::Tensor features_tensor,
    at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor out_tensor) {

    CHECK_INPUT(features_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(weight_tensor);
    CHECK_INPUT(out_tensor);

    int N = out_tensor.size(0);
    int channels = features_tensor.size(1);

    const float* features = features_tensor.data_ptr<float>();
    const float* weight = weight_tensor.data_ptr<float>();
    const int* idx = idx_tensor.data_ptr<int>();
    float* out = out_tensor.data_ptr<float>();

    three_interpolate_kernel_launcher_stack(N, channels, features, idx, weight, out);
}


void three_interpolate_grad_wrapper_stack(at::Tensor grad_out_tensor, at::Tensor idx_tensor,
    at::Tensor weight_tensor, at::Tensor grad_features_tensor) {

    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(weight_tensor);
    CHECK_INPUT(grad_features_tensor);

    int N = grad_out_tensor.size(0);
    int channels = grad_out_tensor.size(1);

    const float* grad_out = grad_out_tensor.data_ptr<float>();
    const float* weight = weight_tensor.data_ptr<float>();
    const int* idx = idx_tensor.data_ptr<int>();
    float* grad_features = grad_features_tensor.data_ptr<float>();

    three_interpolate_grad_kernel_launcher_stack(N, channels, grad_out, idx, weight, grad_features);
}
