#include "esm_grid_sample.h"
#include <iostream>
#include <math.h>
//pragma once

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>


namespace detail {

  enum class GridSamplerInterpolation {Bilinear, Nearest, Bicubic};
  enum class GridSamplerPadding {Zeros, Border, Reflection};

}  // namespace detail

using detail::GridSamplerInterpolation;
using detail::GridSamplerPadding;

// Unnormalizes a coordinate from the -1 to +1 scale to its pixel index value,
// where we view each pixel as an area between (idx - 0.5) and (idx + 0.5).
// if align_corners: -1 and +1 get sent to the centers of the corner pixels
//     -1 --> 0
//     +1 --> (size - 1)
//     scale_factor = (size - 1) / 2
// if not align_corners: -1 and +1 get sent to the image edges
//     -1 --> -0.5
//     +1 --> (size - 1) + 0.5 == size - 0.5
//     scale_factor = size / 2
template <typename scalar_t>
static inline scalar_t grid_sampler_unnormalize(scalar_t coord, int64_t size,
                                                bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    return ((coord + 1) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1) * size - 1) / 2;
  }
}

// grid_sampler_unnormalize_set_grad works the same as grid_sampler_unnormalize
// except that it also returns the `d output / d input` via pointer argument
// `grad_in`.
// This is useful in the backward pass of grid_sampler.
template <typename scalar_t>
static inline scalar_t grid_sampler_unnormalize_set_grad(scalar_t coord, int64_t size,
                                                         bool align_corners, scalar_t *grad_in) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    *grad_in = static_cast<scalar_t>(size - 1) / 2;
    return ((coord + 1) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    *grad_in = static_cast<scalar_t>(size) / 2;
    return ((coord + 1) * size - 1) / 2;
  }
}

// Clips coordinates to between 0 and clip_limit - 1
template<typename scalar_t>
static inline scalar_t clip_coordinates(scalar_t in, int64_t clip_limit) {
  return std::min(static_cast<scalar_t>(clip_limit - 1), std::max(in, static_cast<scalar_t>(0)));
}

// clip_coordinates_set_grad works similarly to clip_coordinates except that
// it also returns the `d output / d input` via pointer argument `grad_in`.
// This is useful in the backward pass of grid_sampler.
template<typename scalar_t>
static inline scalar_t clip_coordinates_set_grad(scalar_t in, int64_t clip_limit,
                                                 scalar_t *grad_in) {
  // Note that it is important for the gradient calculation that borders
  // are considered out of bounds.
  if (in <= static_cast<scalar_t>(0)) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  } else {
    scalar_t max = static_cast<scalar_t>(clip_limit - 1);
    if (in >= max) {
      *grad_in = static_cast<scalar_t>(0);
      return max;
    } else {
      *grad_in = static_cast<scalar_t>(1);
      return in;
    }
  }
}

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template<typename scalar_t>
static inline scalar_t reflect_coordinates(scalar_t in, int64_t twice_low,
                                           int64_t twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = std::fabs(in - min);
  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  scalar_t extra = std::fmod(in, span);
  int flips = static_cast<int>(std::floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

// reflect_coordinates_set_grad works similarly to reflect_coordinates except
// that it also returns the `d output / d input` via pointer argument
// `grad_in`.
// This is useful in the backward pass of grid_sampler.
template<typename scalar_t>
static inline scalar_t reflect_coordinates_set_grad(scalar_t in, int64_t twice_low,
                                                    int64_t twice_high, scalar_t *grad_in) {
  if (twice_low == twice_high) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  }
  int grad_in_mult_;
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = in - min;
  if (in < static_cast<scalar_t>(0)) {
    grad_in_mult_ = -1;
    in = -in;
  } else {
    grad_in_mult_ = 1;
  }
  // `fmod` returns same sign as `in`, which is positive after the `if` above.
  scalar_t extra = std::fmod(in, span);
  int flips = static_cast<int>(std::floor(in / span));
  if (flips % 2 == 0) {
    *grad_in = static_cast<scalar_t>(grad_in_mult_);
    return extra + min;
  } else {
    *grad_in = static_cast<scalar_t>(-grad_in_mult_);
    return span - extra + min;
  }
}

// Mapping the out-of-boundary points back into boundary
// This would only affect padding_mode=border or reflection
template<typename scalar_t>
static inline scalar_t compute_coordinates(scalar_t coord, int64_t size,
                                           GridSamplerPadding padding_mode,
                                           bool align_corners=true) {
  if (padding_mode == GridSamplerPadding::Border) {
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2*(size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2*size - 1);
    }
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  }
  return coord;
}

// Computes the pixel source index value for a grid coordinate
template <typename scalar_t>
static inline scalar_t grid_sampler_compute_source_index(
    scalar_t coord,
    int64_t size,
    GridSamplerPadding padding_mode,
    bool align_corners) {
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  coord = compute_coordinates(coord, size, padding_mode, align_corners);
  return coord;
}

// grid_sampler_compute_source_index_set_grad works similarly to
// grid_sampler_compute_source_index except that it also returns the
// `d output / d input` via pointer argument `grad_in`.
// This is useful in the backward pass of grid_sampler.
template <typename scalar_t>
static inline scalar_t grid_sampler_compute_source_index_set_grad(
    scalar_t coord,
    int64_t size,
    GridSamplerPadding padding_mode,
    bool align_corners,
    scalar_t *grad_in) {
  scalar_t grad_clip, grad_refl;
  coord = grid_sampler_unnormalize_set_grad(coord, size, align_corners, grad_in);
  if (padding_mode == GridSamplerPadding::Border) {
    // clip coordinates to image borders
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_clip;
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = reflect_coordinates_set_grad(coord, 0, 2*(size - 1), &grad_refl);
    } else {
      coord = reflect_coordinates_set_grad(coord, -1, 2*size - 1, &grad_refl);
    }
    // clip coordinates to image borders
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_refl * grad_clip;
  }
  return coord;
}

static inline bool within_bounds_2d(int64_t h, int64_t w, int64_t H, int64_t W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

static inline bool within_bounds_3d(int64_t d, int64_t h, int64_t w, int64_t D, int64_t H, int64_t W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

template<typename scalar_t>
static inline scalar_t get_value_bounded(
    scalar_t* data,
    scalar_t x,
    scalar_t y,
    int64_t W,
    int64_t H,
    int64_t sW,
    int64_t sH,
    GridSamplerPadding padding_mode,
    bool align_corners) {

  x = compute_coordinates(x, W, padding_mode, align_corners);
  y = compute_coordinates(y, H, padding_mode, align_corners);

  int64_t ix = static_cast<int64_t>(x);
  int64_t iy = static_cast<int64_t>(y);

  if (within_bounds_2d(iy, ix, H, W)) {
    return data[iy * sH + ix * sW];
  }
  return static_cast<scalar_t>(0);
}

template<typename scalar_t>
static inline void safe_add_2d(scalar_t *data, int64_t h, int64_t w,
                               int64_t sH, int64_t sW, int64_t H, int64_t W,
                               scalar_t delta) {
  if (within_bounds_2d(h, w, H, W)) {
    data[h * sH + w * sW] += delta;
  }
}

template<typename scalar_t>
static inline void safe_add_3d(scalar_t *data, int64_t d, int64_t h, int64_t w,
                               int64_t sD, int64_t sH, int64_t sW,
                               int64_t D, int64_t H, int64_t W,
                               scalar_t delta) {
  if (within_bounds_3d(d, h, w, D, H, W)) {
    data[d * sD + h * sH + w * sW] += delta;
  }
}

template<typename scalar_t>
static inline void add_value_bounded(
    scalar_t* data,
    scalar_t x,
    scalar_t y,
    int64_t W,
    int64_t H,
    int64_t sW,
    int64_t sH,
    scalar_t delta,
    GridSamplerPadding padding_mode,
    bool align_corners) {

  x = compute_coordinates(x, W, padding_mode, align_corners);
  y = compute_coordinates(y, H, padding_mode, align_corners);

  int64_t ix = static_cast<int64_t>(x);
  int64_t iy = static_cast<int64_t>(y);

  safe_add_2d(data, iy, ix, sH, sW, H, W, delta);
}

// Calculate the differential of the cubic convolution, i.e. `d coeff / d x`
template<typename scalar_t>
static inline void get_cubic_coefficients_grad(
    scalar_t coeffs[4],
    scalar_t t) {

  // Must be the same as forward calculation in
  // aten/src/ATen/native/UpSample.h:get_cubic_upsample_coefficients
  scalar_t A = -0.75;

  scalar_t x;
  x = -1 - t; // 1 < x = |-1 - tx| < 2
  coeffs[0] = (-3 * A * x - 10 * A ) * x - 8 * A;
  x = -t;     // x = |0 - tx| <= 1
  coeffs[1] = (-3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 1 - t;  // x = |1 - tx| <= 1
  coeffs[2] = (3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 2 - t;  // 1 < x = |2 - tx| < 2
  coeffs[3] = (3 * A * x - 10 * A) * x + 8 * A;
}

/*
torch::Tensor xxesm_grid_sample_forward_cpu1(const torch::Tensor& input, const torch::Tensor& grid) {
    torch::Tensor output = torch::zeros(input.sizes());
    //int bs, height, width;
    int bs = input.size(0);
    int channel = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    //auto bs, height, width= input.sizes()[0], input.sizes()[1], input.sizes()[2];
    std::cout << "the shape of bs height width " << bs << height << width << std::endl;
    //torch::Tensor output = torch.zeros((bs,height,width))
    std::cout << "the shape of input " << input.sizes() << std::endl;
    std::cout << "the shape of grid  " << grid.sizes() << std::endl;
    at::parallel_for(0, bs, 0, [&](int64_t start, int64_t end) {
      for(int i=start; i<end; i++){
        at::parallel_for(0, height, 0, [&](int64_t vstart, int64_t vend) {
          for(int v=vstart; v<vend; v++){
            at::parallel_for(0, width, 0, [&](int64_t ustart, int64_t uend) {
              for(int u=ustart; u<uend; u++){
                  //std::cout << "grid " << grid.index({"...", v, u, 1}) << grid.index({"...", v, u, 0}) << std::endl;
                  torch::Tensor  v_grid = grid.index({i, v, u, 1}).clamp(-32768 , 32767);
                  torch::Tensor  u_grid = grid.index({i, v, u, 0}).clamp(-32768 , 32767);
                  //torch::Tensor  v_grid = grid.index({"...", v, u, 1}).clamp(-2147483648, 2147483647);
                  //torch::Tensor  u_grid = grid.index({"...", v, u, 0}).clamp(-2147483648, 2147483647);
                  int roundv = torch::round(v_grid).item<int>();
                  int roundu = torch::round(u_grid).item<int>();
                  if(roundv>0 && roundv<height && roundu>0 && roundu<width){
                    // std::cout << "u, v : " << v << "; " << u << std::endl;
                    output.index({i, 0,  roundv, roundu}) = input.index({i, 0, v, u});
                  }
              }
            });
          }
        });
      }
    });

    return output;
}
*/


/*
std::vector<torch::Tensor> esm_grid_sample_backward_cpu(const torch::Tensor& gradOutput,const torch::Tensor& input, const torch::Tensor& grid){
    //torch::Tensor gradOutputY =  (1/y) * gradOutput * torch::ones(gradOutput.sizes());
    int bs = gradOutput.size(0);
    int channel = gradOutput.size(1);
    int height = gradOutput.size(2);
    int width = gradOutput.size(3);
    std::cout << "Backward: the shape of grad output " << gradOutput.sizes() << std::endl;
    torch::Tensor gradSourceDepth = torch::zeros(gradOutput.sizes());
    for(int i=0; i<bs; i++){
      for(int v=0; v<height; v++){
        for(int u=0; u<width; u++){
            torch::Tensor  v_grid = grid.index({i, v, u, 1}).clamp(-32768 , 32767);
            torch::Tensor  u_grid = grid.index({i, v, u, 0}).clamp(-32768 , 32767);
            int roundv = torch::round(v_grid).item<int>();
            int roundu = torch::round(u_grid).item<int>();
            if(roundv>0 && roundv<height && roundu>0 && roundu<width){
              gradSourceDepth.index({i, 0, v, u }) = gradOutput.index({i, 0, roundv, roundu});
            }
        }
      }
    }
    return {gradSourceDepth,gradSourceDepth};
}
*/

//////////////////////////////////////////////////////
// cpu function  
/////////////////////////////////////////////////////


torch::Tensor esm_grid_sample_forward_cpu(const torch::Tensor& input, const torch::Tensor& grid,
                                    const torch::Tensor& gt_map,
                                    int interpolation_mode,
                                  int padding_mode_,
                                  bool align_corners) {

    using scalar_t = float;
    //bool align_corners=true;
    auto padding_mode = static_cast<GridSamplerPadding>(padding_mode_);
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inp_H = input.size(2);
    int64_t inp_W = input.size(3);
    int64_t out_H = grid.size(1);
    int64_t out_W = grid.size(2);
    auto output = at::zeros({N, C, out_H, out_W}, input.options());
    int64_t inp_sN = input.stride(0);
    int64_t inp_sC = input.stride(1);
    int64_t inp_sH = input.stride(2);
    int64_t inp_sW = input.stride(3);
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sH = grid.stride(1);
    int64_t grid_sW = grid.stride(2);
    int64_t grid_sCoor = grid.stride(3);
    int64_t out_sN = output.stride(0);
    int64_t out_sC = output.stride(1);
    int64_t out_sH = output.stride(2);
    int64_t out_sW = output.stride(3);
    scalar_t *inp_ptr = input.data_ptr<scalar_t>();
    scalar_t *out_ptr = output.data_ptr<scalar_t>();
    scalar_t *grid_ptr = grid.data_ptr<scalar_t>();
    
    // loop over each output pixel
    at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
      for (int64_t n = start; n < end; ++n) {
        scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
        scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
        for (int64_t h = 0; h < inp_H; ++h) {
          for (int64_t w = 0; w < inp_W; ++w) {
            // get the corresponding input x, y, z co-ordinates from grid
            scalar_t *grid_ptr_NHW = grid_ptr_N + h * grid_sH + w * grid_sW;
            scalar_t x = *grid_ptr_NHW;
            scalar_t y = grid_ptr_NHW[grid_sCoor];

            scalar_t ix = grid_sampler_compute_source_index(x, out_W, padding_mode, align_corners);
            scalar_t iy = grid_sampler_compute_source_index(y, out_H, padding_mode, align_corners);
            // nearst 
            int64_t ix_nearest = static_cast<int64_t>(std::nearbyint(ix));
            int64_t iy_nearest = static_cast<int64_t>(std::nearbyint(iy));


            /*
            // assign nearest neighor pixel value to output pixel
            scalar_t *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
            scalar_t *inp_ptr_NC = inp_ptr_N;
            for (int64_t c = 0; c < C; ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
              if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
                *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
              } else {
                *out_ptr_NCHW = static_cast<scalar_t>(0);
              }
            }*/
            // assign nearest neighor pixel value to output pixel
            bool label = within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W);
            //std::cout << "  iy neast  " << iy_nearest << "  ix nearst " << ix_nearest << "if in bound" << label << std::endl;
            if(label) {
              scalar_t *out_ptr_NCHW = out_ptr + n * out_sN + iy_nearest * out_sH + ix_nearest * out_sW;
              scalar_t *inp_ptr_NC = inp_ptr_N;
              //std::cout << "start for " << std::endl;
              for (int64_t c = 0; c < C; ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {                
                  //std::cout << "give value if" << std::endl;
                  *out_ptr_NCHW = inp_ptr_NC[h * inp_sH + w * inp_sW];
              } 
            }
            /*
            else {
              scalar_t *out_ptr_NCHW = out_ptr + n * out_sN + iy_nearest * out_sH + ix_nearest * out_sW;
              scalar_t *inp_ptr_NC = inp_ptr_N;
              std::cout << "start for " << std::endl;
              for (int64_t c = 0; c < C; ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
                                  
                std::cout << "give value else" << std::endl;
                *out_ptr_NCHW = static_cast<scalar_t>(0);
              }
            }
            */
            
          }
        }
      }
    });

    return output;
}


std::vector<torch::Tensor> esm_grid_sample_backward_cpu(const torch::Tensor& grad_output,
                                       const torch::Tensor& input, const torch::Tensor& grid,
                                       const torch::Tensor& gt_map,
                                       int interpolation_mode,
                                  int padding_mode_,
                                  bool align_corners){
  using scalar_t = float;
  const auto padding_mode = static_cast<GridSamplerPadding>(padding_mode_);
  //bool align_corners=true;
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // If interpolation mode is Nearest, then grad_grid is not filled in the
  // loop below.
  //if (interpolation_mode == GridSamplerInterpolation::Nearest) {
  grad_grid.zero_();
  //}
  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t inp_H = input.size(2);
  int64_t inp_W = input.size(3);
  int64_t out_H = grid.size(1);
  int64_t out_W = grid.size(2);
  int64_t inp_sN = input.stride(0);
  int64_t inp_sC = input.stride(1);
  int64_t inp_sH = input.stride(2);
  int64_t inp_sW = input.stride(3);
  int64_t grid_sN = grid.stride(0);
  int64_t grid_sH = grid.stride(1);
  int64_t grid_sW = grid.stride(2);
  int64_t grid_sCoor = grid.stride(3);
  int64_t gOut_sN = grad_output.stride(0);
  int64_t gOut_sC = grad_output.stride(1);
  int64_t gOut_sH = grad_output.stride(2);
  int64_t gOut_sW = grad_output.stride(3);
  int64_t gInp_sN = grad_input.stride(0);
  int64_t gInp_sC = grad_input.stride(1);
  int64_t gInp_sH = grad_input.stride(2);
  int64_t gInp_sW = grad_input.stride(3);
  int64_t gGrid_sN = grad_grid.stride(0);
  int64_t gGrid_sW = grad_grid.stride(2);
  scalar_t *inp_ptr = input.data_ptr<scalar_t>();
  scalar_t *grid_ptr = grid.data_ptr<scalar_t>();
  scalar_t *gOut_ptr = grad_output.data_ptr<scalar_t>();
  scalar_t *gInp_ptr = grad_input.data_ptr<scalar_t>();
  scalar_t *gGrid_ptr = grad_grid.data_ptr<scalar_t>();
  // loop over each output pixel
  at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
    for (int64_t n = start; n < end; ++n) {
      scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
      scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
      scalar_t *gGrid_ptr_NHW = gGrid_ptr + n * gGrid_sN;
      for (int64_t h = 0; h < inp_H; ++h) {
        for (int64_t w = 0; w < inp_W; ++w, gGrid_ptr_NHW += gGrid_sW  ) { // grad_grid is contiguous 
          // get the corresponding input x, y co-ordinates from grid
          scalar_t *grid_ptr_NHW = grid_ptr_N + h * grid_sH + w * grid_sW;
          scalar_t x = *grid_ptr_NHW;
          scalar_t y = grid_ptr_NHW[grid_sCoor];

          // multipliers for gradients on ix, iy
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          scalar_t gix_mult, giy_mult;
          scalar_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &gix_mult);
          scalar_t iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &giy_mult);
          // nearest
        
          int64_t ix_nearest = static_cast<int64_t>(std::nearbyint(ix));
          int64_t iy_nearest = static_cast<int64_t>(std::nearbyint(iy));

          // assign nearest neighor pixel value to output pixel
          scalar_t *gOut_ptr_NCHW = gOut_ptr + n * gOut_sN + iy_nearest * gOut_sH + ix_nearest * gOut_sW; //Dw
          scalar_t *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
          for (int64_t c = 0; c < C; ++c, gOut_ptr_NCHW += gOut_sC, gInp_ptr_NC += gInp_sC) {
            // calculate and set grad_input
            safe_add_2d(gInp_ptr_NC, h, w, gInp_sH, gInp_sW,
                        inp_H, inp_W, *gOut_ptr_NCHW);
          }
          
        }
      }
    }
  });
  return {grad_input, grad_grid,};
}



/*
// pybind11 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward",   &esm_grid_sample_forward_cpu, "esm_grid_sample forward");
  m.def("backward",  &esm_grid_sample_backward_cpu, "esm_grid_sample backward");
}
*/
// pybind11 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward",   &esm_grid_sample_forward_cpu, "esm_grid_sample forward");
  m.def("backward",  &esm_grid_sample_backward_cpu, "esm_grid_sample backward");
}