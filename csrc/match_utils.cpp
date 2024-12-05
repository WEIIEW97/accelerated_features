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

#include "match_utils.h"

cv::Mat warp_corners_and_draw_matches(cv::Mat& ref_points, cv::Mat& dst_points,
                                      cv::Mat& img1, cv::Mat& img2,
                                      bool is_show) {
  // Check if there are enough points to find a homography
  if (ref_points.rows < 4 || dst_points.rows < 4) {
    std::cerr << "Not enough points to compute homography" << std::endl;
    return cv::Mat();
  }

  cv::Mat mask;
  cv::Mat H = cv::findHomography(ref_points, dst_points, cv::USAC_MAGSAC, 10.0,
                                 mask, 1000, 0.994);
  if (H.empty()) {
    std::cerr << "Homography matrix is empty" << std::endl;
    return cv::Mat();
  }
  mask = mask.reshape(1);

  float h = img1.rows;
  float w = img1.cols;
  std::vector<cv::Point2f> corners_img1 = {
      cv::Point2f(0, 0), cv::Point2f(w - 1, 0), cv::Point2f(w - 1, h - 1),
      cv::Point2f(0, h - 1)};
  std::vector<cv::Point2f> warped_corners;
  cv::perspectiveTransform(corners_img1, warped_corners, H);

  cv::Mat img2_with_corners = img2.clone();
  for (size_t i = 0; i < warped_corners.size(); ++i) {
    cv::line(img2_with_corners, warped_corners[i],
             warped_corners[(i + 1) % warped_corners.size()],
             cv::Scalar(0, 255, 0), 4);
  }

  // prepare keypoints and matches for drawMatches function
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  std::vector<cv::DMatch> matches;
  for (int i = 0; i < mask.rows; ++i) {
    keypoints1.emplace_back(ref_points.at<cv::Point2f>(i, 0), 5);
    keypoints2.emplace_back(dst_points.at<cv::Point2f>(i, 0), 5);
    if (mask.at<uchar>(i, 0))
      matches.emplace_back(i, i, 0);
  }

  // Draw inlier matches
  cv::Mat img_matches;
  if (!keypoints1.empty() && !keypoints2.empty() && !matches.empty()) {
    cv::drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches,
                    img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // // Uncomment to save the matched image
    // std::string output_path = "doc/image_matches.png";
    // if (cv::imwrite(output_path, img_matches)) {
    //     std::cout << "Saved image matches to " << output_path << std::endl;
    // } else {
    //     std::cerr << "Failed to save image matches to " << output_path <<
    //     std::endl;
    // }

  } else {
    std::cerr << "Keypoints or matches are empty, cannot draw matches"
              << std::endl;
  }
  return img_matches;
}