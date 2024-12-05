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
#include <tuple>

namespace xfeat {
  struct BasicLayerImpl : torch::nn::Module {
    /*
    Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    */
    torch::nn::Sequential layer;
    BasicLayerImpl(int in_channels, int out_channels, int kernel_size,
                   int stride, int padding, int dilation, bool bias=false);
    torch::Tensor forward(torch::Tensor& x);
  };

  TORCH_MODULE(BasicLayer);

  struct XFeatModel : torch::nn::Module {
    /*
    Implementation of architecture described in
       "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    */

    torch::nn::InstanceNorm2d norm{nullptr};
    torch::nn::Sequential skip1{nullptr};
    torch::nn::Sequential block1{nullptr}, block2{nullptr}, block3{nullptr},
        block4{nullptr}, block5{nullptr};
    torch::nn::Sequential block_fusion{nullptr}, heatmap_head{nullptr},
        keypoint_head{nullptr};
    torch::nn::Sequential fine_matcher{nullptr};
    XFeatModel();
    torch::Tensor unfold2d(torch::Tensor& x, int ws = 2);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    forward(torch::Tensor& x);
  };

  // TORCH_MODULE(XFeatModel);
} // namespace xfeat