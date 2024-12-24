/*
 * @Developer: ACENTAURI team, INRIA institute
 * @Author: Ziming Liu
 * @Date: 2023-06-13 19:00:40
 * @LastEditors: Ziming Liu
 * @LastEditTime: 2023-06-13 19:02:35
 */
#include <torch/extension.h>
#include <vector>



std::vector<torch::Tensor> grid_sample2d_grad2(
    const torch::Tensor &grad2_grad_input,
    const torch::Tensor &grad2_grad_grid,
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &grid,
    bool padding_mode,
    bool align_corners) 


std::vector<torch::Tensor> grid_sample3d_grad2(
    const torch::Tensor &grad2_grad_input,
    const torch::Tensor &grad2_grad_grid,
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &grid,
    bool padding_mode,
    bool align_corners) 