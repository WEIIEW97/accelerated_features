import numpy as np
import torch
from modules.xfeat import XFeat
import matplotlib.pyplot as plt

import cv2

XFEAT_CKPT = "/home/william/Codes/accelerated_features/weights/xfeat.pt"


def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(
        ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999
    )
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    ).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i - 1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(
            img2_with_corners, start_point, end_point, (0, 255, 0), 4
        )  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(
        img1,
        keypoints1,
        img2_with_corners,
        keypoints2,
        matches,
        None,
        matchColor=(0, 255, 0),
        flags=2,
    )

    return img_matches


class XFeatInferencer:
    def __init__(self, xfeat_ckpt, top_k=4096, detection_thr=0.05):
        self.xfeat = XFeat(
            weights=xfeat_ckpt, top_k=top_k, detection_threshold=detection_thr
        )

        self.top_k = top_k

    def match(self, im0, im1):
        mkpts_0, mkpts_1 = self.xfeat.match_xfeat(im0, im1, top_k=self.top_k)

        self.mkpts_0 = mkpts_0
        self.mkpts_1 = mkpts_1

    def plot_match_result(self, im0, im1):
        canvas = warp_corners_and_draw_matches(self.mkpts_0, self.mkpts_1, im0, im1)
        plt.figure(figsize=(12, 12))
        plt.imshow(canvas[..., ::-1])
        plt.show()


xfeat_inference = XFeatInferencer(XFEAT_CKPT)