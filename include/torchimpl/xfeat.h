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

#include "net.h"
#include "interp_sparse_2d.h"
#include <tuple>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace xfeat {
  class Detector {
  public:
    Detector(const std::string& weights_path, int top_k = 4096,
             float detection_threshold = 0.05, bool use_cuda = true);
    void
    detect_and_compute(torch::Tensor& x,
                       std::unordered_map<std::string, torch::Tensor>& result);
    void match(torch::Tensor& feats1, torch::Tensor& feats2,
               torch::Tensor& idx0, torch::Tensor& idx1,
               float _min_cossim = -1.0);
    void match_xfeat(cv::Mat& img1, cv::Mat& img2, cv::Mat& mkpts_0,
                     cv::Mat& mkpts_1);
    torch::Tensor parse_input(cv::Mat& img);
    std::tuple<torch::Tensor, double, double>
    preprocess_tensor(torch::Tensor& x);
    cv::Mat tensor2mat(const torch::Tensor& tensor);

  private:
    torch::Tensor get_kpts_heatmap(torch::Tensor& kpts,
                                   float softmax_temp = 1.0);
    torch::Tensor NMS(torch::Tensor& x, float threshold = 0.05,
                      int kernel_size = 5);
    std::string get_weights_path(std::string weights);

    std::string weights;
    int top_k;
    float min_cossim;
    float detection_threshold;
    torch::DeviceType device_type;
    std::shared_ptr<XFeatModel> model;
    std::shared_ptr<InterpolateSparse2d> bilinear, nearest;
  };
} // namespace xfeat