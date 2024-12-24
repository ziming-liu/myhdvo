/*
 * @Developer: ACENTAURI team, INRIA institute
 * @Author: Ziming Liu
 * @Date: 2023-06-02 17:37:54
 * @LastEditors: Ziming Liu
 * @LastEditTime: 2023-06-03 19:10:06
 */
#include <torch/extension.h>
#include <vector>




// forward propagation
torch::Tensor esm_grid_sample_forward_cpu(const torch::Tensor& source_depth, const torch::Tensor& pr, 
                                const torch::Tensor& gt_map,
                                int interpolation_mode,
                                  int padding_mode,
                                  bool align_corners);
// backbward propagation
std::vector<torch::Tensor> esm_grid_sample_backward_cpu(const torch::Tensor& gradOutput,  torch::Tensor& source_depth, torch::Tensor& pr,
                                                        const torch::Tensor& gt_map,
                                                        int interpolation_mode,
                                                        int padding_mode,
                                                        bool align_corners);




