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
#pragma once
#include <torch/torch.h>

namespace xfeat {
  class InterpolateSparse2d : public torch::nn::Module {
  public:
    InterpolateSparse2d(const std::string& mode = "bilinear",
                            bool align_corners = false);
    torch::Tensor forward(torch::Tensor x, torch::Tensor pos, int H, int W);

  private:
    torch::Tensor normgrid(torch::Tensor x, int H, int W);

    std::string mode;
    bool align_corners;
  };

} // namespace xfeat