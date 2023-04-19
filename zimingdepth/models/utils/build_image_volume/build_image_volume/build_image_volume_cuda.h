/*
 * @Developer: ACENTAURI team, INRIA institute
 * @Author: Ziming Liu
 * @Date: 2023-04-18 17:49:50
 * @LastEditors: Ziming Liu
 * @LastEditTime: 2023-04-19 01:50:13
 */
#include <torch/extension.h>
#include <vector>

 
at::Tensor build_image_volume_cuda(const at::Tensor left_image, const at::Tensor right_image, 
                    int64_t max_disp, const at::Tensor mask_template);
 