#include <torch/extension.h>
#include <c10/macros/Macros.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/Dispatch.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <iostream>

namespace at { namespace native {
namespace {

using namespace at::cuda::detail;

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_2d_ic_kernel(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> grad_output,
      TensorInfo<scalar_t, index_t> input,
      TensorInfo<scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> gt_map,
      TensorInfo<scalar_t, index_t> output,
      TensorInfo<scalar_t, index_t> grad_input,
      TensorInfo<scalar_t, index_t> grad_grid,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners,
      bool input_requires_grad,
      const index_t grad_input_memory_span) {

    index_t C = input.sizes[1];
    index_t inp_H = input.sizes[2];
    index_t inp_W = input.sizes[3];
    index_t oup_H = output.sizes[2];
    index_t oup_W = output.sizes[3];
    index_t gt_H = gt_map.sizes[2];
    index_t gt_W = gt_map.sizes[3];
    index_t out_H = grid.sizes[1];
    index_t out_W = grid.sizes[2];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sH = input.strides[2];
    index_t inp_sW = input.strides[3];
    index_t oup_sN = output.strides[0];
    index_t oup_sC = output.strides[1];
    index_t oup_sH = output.strides[2];
    index_t oup_sW = output.strides[3];
    index_t gt_sN = gt_map.strides[0];
    index_t gt_sC = gt_map.strides[1];
    index_t gt_sH = gt_map.strides[2];
    index_t gt_sW = gt_map.strides[3];
    index_t grid_sN = grid.strides[0];
    index_t grid_sH = grid.strides[1];
    index_t grid_sW = grid.strides[2];
    index_t grid_sCoor = grid.strides[3];
    index_t gOut_sN = grad_output.strides[0];
    index_t gOut_sC = grad_output.strides[1];
    index_t gOut_sH = grad_output.strides[2];
    index_t gOut_sW = grad_output.strides[3];
    // gInp_* (and NC_offset below) are not really needed if input_requires_grad is false.
    index_t gInp_sN;
    index_t gInp_sC;
    index_t gInp_sH;
    index_t gInp_sW;
    if (input_requires_grad) {
      gInp_sN = grad_input.strides[0];
      gInp_sC = grad_input.strides[1];
      gInp_sH = grad_input.strides[2];
      gInp_sW = grad_input.strides[3];
    }
    index_t gGrid_sW = grad_grid.strides[2];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t n = index / (out_H * out_W);
      const auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y co-ordinates from grid
      scalar_t x = grid.data[grid_offset];
      scalar_t y = grid.data[grid_offset + grid_sCoor];

      // multipliers for gradients on ix and iy
      scalar_t gix_mult, giy_mult;
      scalar_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &gix_mult);
      scalar_t iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &giy_mult);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get NE, NW, SE, SW pixel values from (x, y)
        index_t ix_nw = static_cast<index_t>(std::floor(ix));
        index_t iy_nw = static_cast<index_t>(std::floor(iy));
        index_t ix_ne = ix_nw + 1;
        index_t iy_ne = iy_nw;
        index_t ix_sw = ix_nw;
        index_t iy_sw = iy_nw + 1;
        index_t ix_se = ix_nw + 1;
        index_t iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        scalar_t nw = (ix_se - ix)    * (iy_se - iy);
        scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
        scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
        scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

        index_t up = h-1;
        index_t down = h+1;
        index_t left = w-1;
        index_t right = w+1;

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
        scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
        
        index_t NC_offset = n * gInp_sN;
        scalar_t *inp_ptr_NC = input.data + n * inp_sN;
        scalar_t *oup_ptr_NC = output.data + n * oup_sN; // new output 
        scalar_t *GT_ptr_NC = gt_map.data + n * gt_sN; // NEW gt map ptr
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC, GT_ptr_NC += gt_sC, oup_ptr_NC += oup_sC) {
          scalar_t gOut = *gOut_ptr_NCHW;

          if (input_requires_grad) {
            // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
            safe_add_2d(grad_input.data, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut, NC_offset, grad_input_memory_span);
            safe_add_2d(grad_input.data, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut, NC_offset, grad_input_memory_span);
            safe_add_2d(grad_input.data, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut, NC_offset, grad_input_memory_span);
            safe_add_2d(grad_input.data, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut, NC_offset, grad_input_memory_span);
          }
          
           // calculate grad_grid
          if (within_bounds_2d(up, left, inp_H, inp_W)) {
            scalar_t up_val = GT_ptr_NC[up * gt_sH + w * gt_sW];
            scalar_t left_val = GT_ptr_NC[h * gt_sH + left * gt_sW];
            giy -= up_val * 0.5 * gOut;
            gix -= left_val * 0.5 * gOut;
          }
          if (within_bounds_2d(down, right, inp_H, inp_W)) {
            scalar_t down_val =  GT_ptr_NC[down * gt_sH + w * gt_sW];
            scalar_t right_val = GT_ptr_NC[h * gt_sH + right * gt_sW];
            giy += down_val * 0.5 * gOut;
            gix += right_val * 0.5 * gOut;
          }
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
        //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
        scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NHW[0] = gix_mult * gix;
        gGrid_ptr_NHW[1] = giy_mult * giy;
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        if (input_requires_grad) {
          index_t ix_nearest = static_cast<index_t>(std::nearbyint(ix));
          index_t iy_nearest = static_cast<index_t>(std::nearbyint(iy));

          // assign nearest neighor pixel value to output pixel
          scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
          index_t NC_offset = n * gInp_sN;
          for (index_t c = 0; c < C; ++c, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
            // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
            safe_add_2d(grad_input.data, iy_nearest, ix_nearest, gInp_sH, gInp_sW, inp_H, inp_W, *gOut_ptr_NCHW, NC_offset, grad_input_memory_span);
          }
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
        //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
        scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NHW[0] = static_cast<scalar_t>(0);
        gGrid_ptr_NHW[1] = static_cast<scalar_t>(0);
      } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {

        ix = grid_sampler_unnormalize_set_grad(x, inp_W, align_corners, &gix_mult);
        iy = grid_sampler_unnormalize_set_grad(y, inp_H, align_corners, &giy_mult);

        scalar_t ix_nw = std::floor(ix);
        scalar_t iy_nw = std::floor(iy);

        const scalar_t tx = ix - ix_nw;
        const scalar_t ty = iy - iy_nw;

        scalar_t x_coeffs[4];
        scalar_t y_coeffs[4];
        scalar_t x_coeffs_grad[4];
        scalar_t y_coeffs_grad[4];

        get_cubic_upsampling_coefficients<scalar_t>(x_coeffs, tx);
        get_cubic_upsampling_coefficients<scalar_t>(y_coeffs, ty);
        get_cubic_coefficients_grad<scalar_t>(x_coeffs_grad, tx);
        get_cubic_coefficients_grad<scalar_t>(y_coeffs_grad, ty);

        scalar_t gix = static_cast<scalar_t>(0);
        scalar_t giy = static_cast<scalar_t>(0);

        scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
        index_t NC_offset = n * gInp_sN;
        scalar_t *inp_ptr_NC = input.data + n * inp_sN;

        for (index_t c = 0; c < C; ++c, gOut_ptr_NCHW += gOut_sC, NC_offset += gInp_sC, inp_ptr_NC+= inp_sC) {
          scalar_t gOut = *gOut_ptr_NCHW;

          #pragma unroll 4
          for (index_t i = 0; i < 4; ++i) {
            #pragma unroll 4
            for (index_t j = 0; j < 4; ++j) {

              if (input_requires_grad) {
                // set input gradient. See Note [Passing pointer and offset to fastAtomicAdd].
                add_value_bounded<scalar_t>(grad_input.data, ix_nw - 1 + i, iy_nw - 1 + j, inp_W, inp_H, gInp_sW, gInp_sH,
                  gOut * x_coeffs[i] * y_coeffs[j],
                  padding_mode,
                  align_corners,
                  NC_offset,
                  grad_input_memory_span);
              }

              // set grid gradient
              scalar_t val = get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw - 1 + i, iy_nw - 1 + j,
                inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners);

              gix -= val * x_coeffs_grad[i] * y_coeffs[j] * gOut;
              giy -= val * y_coeffs_grad[j] * x_coeffs[i] * gOut;
            }
          }
        }

        scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NHW[0] = gix_mult * gix;
        gGrid_ptr_NHW[1] = giy_mult * giy;
      }
    }
  }
  
}

std::vector<torch::Tensor> grid_sample2d_cuda_ic(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &grid,
    const torch::Tensor &gt_map,
    const torch::Tensor &output,
    bool interpolation_mode,
    bool padding_mode,
    bool align_corners,
    bool input_requires_grad) {

    const auto batch_size = input.size(0);
    const auto C = input.size(1);
    const auto H_IN = input.size(2);
    const auto W_IN = input.size(3);

    const auto H_OUT = grid.size(1);
    const auto W_OUT = grid.size(2);

    torch::Tensor grad_input = torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor grad_grid = torch::zeros_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

    int64_t count = batch_size * H_OUT * W_OUT;
 
    if (count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_2d_grad2_cuda", [&] {
          if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
              canUse32BitIndexMath(grad_output)) {
            grid_sampler_2d_ic_kernel<scalar_t>
              <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<int>(count),
                getTensorInfo<scalar_t, int>(grad_output),
                getTensorInfo<scalar_t, int>(input),
                getTensorInfo<scalar_t, int>(grid),
                getTensorInfo<scalar_t, int>(gt_map),
                getTensorInfo<scalar_t, int>(output),
                getTensorInfo<scalar_t, int>(grad_input),
                getTensorInfo<scalar_t, int>(grad_grid),
                static_cast<GridSamplerInterpolation>(interpolation_mode),
                static_cast<GridSamplerPadding>(padding_mode),
                align_corners,
                input_requires_grad,
                static_cast<int>(grad_input.numel()));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else {
            grid_sampler_2d_ic_kernel<scalar_t>
              <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                getTensorInfo<scalar_t, int64_t>(grad_output),
                getTensorInfo<scalar_t, int64_t>(input),
                getTensorInfo<scalar_t, int64_t>(grid),
                getTensorInfo<scalar_t, int64_t>(gt_map),
                getTensorInfo<scalar_t, int64_t>(output),
                getTensorInfo<scalar_t, int64_t>(grad_input),
                getTensorInfo<scalar_t, int64_t>(grad_grid),
                static_cast<GridSamplerInterpolation>(interpolation_mode),
                static_cast<GridSamplerPadding>(padding_mode),
                align_corners,
                input_requires_grad,
                grad_input.numel());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        });
    }
  
  return {grad_input, grad_grid};
}
 

}}  // namespace at::native::cuda
