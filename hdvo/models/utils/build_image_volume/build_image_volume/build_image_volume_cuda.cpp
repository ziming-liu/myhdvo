/*
 * @Developer: ACENTAURI team, INRIA institute
 * @Author: Ziming Liu
 * @Date: 2023-04-18 17:28:16
 * @LastEditors: Ziming Liu
 * @LastEditTime: 2023-04-19 01:50:03
 */

#include <torch/extension.h>					// header file
#include <torch/torch.h>
#include <cmath>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
 

at::Tensor BuildImageVolume(const at::Tensor left_image, const at::Tensor right_image, 
                    int64_t max_disp, const at::Tensor mask_template);

at::Tensor build_image_volume_cuda(const at::Tensor left_image, const at::Tensor right_image, 
                    int64_t max_disp, const at::Tensor mask_template){
  /*
  TORCH_CHECK(
    left_image.defined() && right_image.defined(),
    "build_image_volume_cuda(): expected left_image and right_image to not be undefined, but left_image "
    "is ", left_image, " and right_image is ", right_image);
  auto left_image_opt = left_image.options();
  auto right_image_opt = right_image.options();
  TORCH_CHECK(
    left_image_opt.device() == right_image_opt.device(),
    "right_image_sampler(): expected left_image and right_image to be on same device, but left_image "
    "is on ", left_image_opt.device(), " and right_image is on ", right_image_opt.device());
  TORCH_CHECK(
    left_image_opt.dtype() == right_image_opt.dtype(),
    "right_image_sampler(): expected left_image and right_image to have same dtype, but left_image "
    "has ", left_image_opt.dtype(), " and right_image has ", right_image_opt.dtype());
  TORCH_CHECK(
    left_image_opt.layout() == c10::kStrided && right_image_opt.layout() == c10::kStrided,
    "right_image_sampler(): expected left_image and right_image to have torch.strided layout, but "
    "left_image has ", left_image_opt.layout(), " and right_image has ", right_image_opt.layout());
  TORCH_CHECK(
    (left_image.dim() == 4 ) && left_image.dim() == right_image.dim(),
    "right_image_sampler(): expected 4D or 5D left_image and right_image with same number of "
    "dimensions, but got left_image with sizes ", left_image.sizes(),
    " and right_image with sizes ", right_image.sizes());
  TORCH_CHECK(
    left_image.size(0) == right_image.size(0),
    "right_image_sampler(): expected right_image and left_image to have same batch size, but got "
    "left_image with sizes ", left_image.sizes(), " and right_image with sizes ", right_image.sizes());

  for (int64_t i = 2; i < left_image.dim(); i++) {
    TORCH_CHECK(left_image.size(i) > 0,
      "right_image_sampler(): expected left_image to have non-empty spatial dimensions, "
      "but left_image has sizes ", left_image.sizes(), " with dimension ", i, " being "
      "empty");
  }*/

  return BuildImageVolume(left_image, right_image, max_disp, mask_template);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {	// binding
    m.def("forward", &build_image_volume_cuda, "build_image_volume_cuda forward");
}
