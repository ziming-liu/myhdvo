/*
 * @Developer: ACENTAURI team, INRIA institute
 * @Author: Ziming Liu
 * @Date: 2023-06-02 17:37:54
 * @LastEditors: Ziming Liu
 * @LastEditTime: 2023-06-15 17:17:40
 */
#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

namespace at {namespace native {
std::vector<torch::Tensor> grid_sample2d_cuda_ic(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &grid,
    const torch::Tensor &gt_map,
    const torch::Tensor &output,
    bool interpolation_mode,
    bool padding_mode,
    bool align_corners,
    bool input_requires_grad);
 
}}

std::vector<torch::Tensor> grid_sample2d_ic(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &grid,
    const torch::Tensor &gt_map,
    const torch::Tensor &output,
    bool interpolation_mode,
    bool padding_mode,
    bool align_corners, 
    bool input_requires_grad) {
  
  return at::native::grid_sample2d_cuda_ic( grad_output, input, grid, 
                            gt_map, output, interpolation_mode, padding_mode, align_corners,input_requires_grad);
}

 


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grad2_2d_ic", &grid_sample2d_ic, "grid_sample2d IC derivative");
}
