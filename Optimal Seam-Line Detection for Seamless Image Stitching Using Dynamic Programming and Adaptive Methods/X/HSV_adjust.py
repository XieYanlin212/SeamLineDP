import cv2
import numpy as np

def HSV(img1,img2):
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    v1 = img1_hsv[:, :, 2]
    v2 = img2_hsv[:, :, 2]
    mean_v1 = np.mean(v1)
    mean_v2 = np.mean(v2)
    adjustment_factor = mean_v2 / mean_v1

    v1_adjusted = np.clip(v1 * adjustment_factor, 0, 255).astype(np.uint8)
    img1_hsv_adjusted = img1_hsv.copy()
    img1_hsv_adjusted[:, :, 2] = v1_adjusted

    img1_adjusted = cv2.cvtColor(img1_hsv_adjusted, cv2.COLOR_HSV2BGR)
    return img1_adjusted


def adjust_H(transform_img1,transform_img2):

    overlap_region_img1_hsv = cv2.cvtColor(transform_img1, cv2.COLOR_BGR2HSV)
    overlap_region_img2_hsv = cv2.cvtColor(transform_img2, cv2.COLOR_BGR2HSV)
    h1_overlap, s1_overlap, v1_overlap = cv2.split(overlap_region_img1_hsv)
    h2_overlap, s2_overlap, v2_overlap = cv2.split(overlap_region_img2_hsv)

    mean_h1_overlap = np.mean(h1_overlap)
    mean_h2_overlap = np.mean(h2_overlap)

    adjustment_factor_h = mean_h2_overlap / mean_h1_overlap

    h1_overlap_adjusted = np.clip(h1_overlap * adjustment_factor_h, 0, 255).astype(np.uint8)

    overlap_region_img1_hsv_adjusted = cv2.merge([h1_overlap_adjusted, s1_overlap, v1_overlap])
    overlap_region_img1_adjusted = cv2.cvtColor(overlap_region_img1_hsv_adjusted, cv2.COLOR_HSV2BGR)

    return overlap_region_img1_adjusted