#include <ATen/ATen.h>
//#include <ATen/native/cuda/GridSampler.cuh>
//#include <ATen/native/cuda/UpSample.cuh>
//#include <ATen/cuda/CUDAContext.h>
//#include <ATen/cuda/CUDAApplyUtils.cuh>
//#include <ATen/cuda/detail/TensorInfo.cuh>
//#include <ATen/cuda/detail/IndexUtils.cuh>
//#include <ATen/cuda/detail/KernelUtils.h>
//#include <c10/macros/Macros.h>
//#include "sparse_warp.h"
//namespace at { namespace native {
#include <torch/extension.h>
#include <vector>
#include <ATen/NativeFunctions.h>
//#include <THC/THCAtomics.cuh>


torch::Tensor BuildImageVolume(const torch::Tensor left_image, const torch::Tensor right_image, 
                    int64_t max_disp, const torch::Tensor mask_template){
    //std::vector<torch::Tensor> image_volume;
    torch::Tensor image_volume_tensor= at::empty({left_image.size(0), left_image.size(1)*2, max_disp, left_image.size(2), left_image.size(3)}, left_image.options());
    for(int64_t i=0;i<max_disp;i++){
        //torch::Tensor right = torch::roll(right_image, {i}, {3});
        //torch::Tensor cat = torch::cat({left_image, torch::roll(right_image, {i}, {3})}, 1);
        //image_volume.push_back(torch::cat({left_image, torch::roll(right_image, {i}, {3})}, 1));
        image_volume_tensor.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), i, torch::indexing::Slice(), torch::indexing::Slice()}, torch::cat({left_image, torch::roll(right_image, {i}, {3})}, 1) );
    }
    //torch::Tensor image_volume_tensor = torch::stack(image_volume, 2);
    return image_volume_tensor*mask_template;
    }

/*
std::vector<torch::Tensor> BuildImageVolume(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, dim=1);
  auto gates = torch::addmm(bias, X, weights.transpose(0, 1));
  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);
  auto new_h = torch::zeros_like(old_cell);
  auto new_cell = torch::zeros_like(old_cell);
  auto input_gate = torch::zeros_like(old_cell);
  auto output_gate = torch::zeros_like(old_cell);
  auto candidate_cell = torch::zeros_like(old_cell);


  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);
  AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
    BuildImageVolume_kernel<scalar_t><<<blocks, threads>>>(
        gates.data<scalar_t>(),
        old_cell.data<scalar_t>(),
        new_h.data<scalar_t>(),
        new_cell.data<scalar_t>(),
        input_gate.data<scalar_t>(),
        output_gate.data<scalar_t>(),
        candidate_cell.data<scalar_t>(),
        state_size);
  }));
  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}


template <typename scalar_t>
__global__ void BuildImageVolume_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> left_image,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> right_image,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> max_disp,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> mask_template,
) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < gates.size(2)){
    input_gate[n][c] = sigmoid(gates[n][0][c]);
    output_gate[n][c] = sigmoid(gates[n][1][c]);
    candidate_cell[n][c] = elu(gates[n][2][c]);
    new_cell[n][c] =
        old_cell[n][c] + candidate_cell[n][c] * input_gate[n][c];
    new_h[n][c] = tanh(new_cell[n][c]) * output_gate[n][c];
  }
}

*/