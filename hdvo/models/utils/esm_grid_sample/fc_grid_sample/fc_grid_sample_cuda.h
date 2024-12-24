#include <torch/extension.h>
#include <vector>



torch::Tensor fc_grid_sample_forward_cuda(const torch::Tensor& input, const torch::Tensor& grid,
                        const torch::Tensor& gt_map,
                    int64_t interpolation_mode, int64_t padding_mode,
                    bool align_corners)


std::vector<torch::Tensor> fc_grid_sample_backward_cuda(const torch::Tensor& output_grad, 
                    const torch::Tensor& input, const torch::Tensor& grid, const torch::Tensor& gt_map,const torch::Tensor& output,
                    int64_t interpolation_mode, int64_t padding_mode,
                    bool align_corners)

