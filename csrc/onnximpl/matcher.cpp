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

#include "onnximpl/matcher.h"

namespace xfeatonnx {
  template <typename T>
  static void ReduceVector(std::vector<T>& v, std::vector<uchar>& status) {
    int j = 0;
    for (int i = 0; i < (int)(v.size()); i++) {
      if (status[i]) {
        v[j++] = v[i];
      }
    }
    v.resize(j);
  }

  void Matcher::Match(const cv::Mat& descs1, const cv::Mat& descs2,
                      std::vector<cv::DMatch>& matches, float minScore) {
    cv::Mat scores12 = descs1 * descs2.t();
    cv::Mat scores21 = descs2 * descs1.t();
    std::vector<int> match12(descs1.rows, -1);
    for (int i = 0; i < scores12.rows; i++) {
      auto* row = scores12.ptr<float>(i);
      float maxScore = row[0];
      int maxIdx = 0;
      for (int j = 1; j < scores12.cols; j++) {
        if (row[j] > maxScore) {
          maxScore = row[j];
          maxIdx = j;
        }
      }
      match12[i] = maxIdx;
    }

    std::vector<int> match21(descs2.rows, -1);
    for (int i = 0; i < scores21.rows; i++) {
      auto* row = scores21.ptr<float>(i);
      float maxScore = row[0];
      int maxIdx = 0;
      for (int j = 1; j < scores21.cols; j++) {
        if (row[j] > maxScore) {
          maxScore = row[j];
          maxIdx = j;
        }
      }
      match21[i] = maxIdx;
    }

    // cross-check
    matches.clear();
    for (int i = 0; i < descs1.rows; i++) {
      int j = match12[i];
      if (match21[j] == i && scores12.at<float>(i, j) > minScore) {
        matches.emplace_back(i, j, scores12.at<float>(i, j));
      }
    }
  }

  bool Matcher::RejectBadMatchesF(std::vector<cv::Point2f>& pts1,
                                  std::vector<cv::Point2f>& pts2,
                                  std::vector<cv::DMatch>& matches,
                                  float thresh) {
    assert(pts1.size() == pts2.size() && pts1.size() == matches.size());
    if (pts1.size() < 8) {
      return false;
    }

    std::vector<uchar> status;
    cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, thresh, 0.999, status);
    ReduceVector(matches, status);
    return true;
  }

} // namespace xfeatonnx