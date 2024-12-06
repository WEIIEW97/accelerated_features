/*
 * Copyright (c) 2022-2024, William Wei. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "torchimpl/interp_sparse_2d.h"

namespace xfeat {
  InterpolateSparse2d::InterpolateSparse2d(const std::string& mode,
                                           bool align_corners)
      : mode(mode), align_corners(align_corners) {}

  torch::Tensor InterpolateSparse2d::normgrid(torch::Tensor x, int H, int W) {
    // normalize coordinates to [-1, 1]
    torch::Tensor size_tensor = torch::tensor({W - 1, H - 1}, x.options());
    return 2.0 * (x / size_tensor) - 1.0;
  }

  torch::Tensor InterpolateSparse2d::forward(torch::Tensor x, torch::Tensor pos,
                                             int H, int W) {
    // normalize the positions
    torch::Tensor grid = normgrid(pos, H, W).unsqueeze(-2).to(x.dtype());

    // grid sampling
    if (mode == "bilinear") {
      x = torch::nn::functional::grid_sample(
          x, grid,
          torch::nn::functional::GridSampleFuncOptions()
              .mode(torch::kBilinear)
              .align_corners(align_corners));
    } else if (mode == "nearest") {
      x = torch::nn::functional::grid_sample(
          x, grid,
          torch::nn::functional::GridSampleFuncOptions()
              .mode(torch::kNearest)
              .align_corners(align_corners));
    } else {
      std::cerr << "Choose either 'bilinear' or 'nearest'." << std::endl;
      exit(EXIT_FAILURE);
    }

    // reshape output to [B, N, C]
    return x.permute({0, 2, 3, 1}).squeeze(-2);
  }
} // namespace xfeat