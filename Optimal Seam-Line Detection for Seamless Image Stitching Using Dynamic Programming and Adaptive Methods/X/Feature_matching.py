import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import DBSCAN
from collections import Counter
import pymagsac

def align_images(img1, img2):

    orb = cv2.ORB_create(10000)
    descriptor = cv2.xfeatures2d.BEBLID_create(0.8)
    kpts1 = orb.detect(img1, None)
    kpts2 = orb.detect(img2, None)
    kp1, des1 = descriptor.compute(img1, kpts1)
    kp2, des2 = descriptor.compute(img2, kpts2)

    keypoints1 = np.array([kp.pt for kp in kp1])
    keypoints2 = np.array([kp.pt for kp in kp2])

    X = np.vstack([keypoints1, keypoints2])
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dbscan = DBSCAN(eps=4.66, min_samples=2)
    labels = dbscan.fit_predict(X_tensor)
    non_noise_indices = labels != -1

    filtered_kp1 = [kp1[i] for i in range(len(kp1)) if non_noise_indices[i]]
    filtered_des1 = [des1[i] for i in range(len(kp1)) if non_noise_indices[i]]

    offset = len(kp1)
    filtered_kp2 = [kp2[i] for i in range(len(kp2)) if non_noise_indices[i + offset]]
    filtered_des2 = [des2[i] for i in range(len(kp2)) if non_noise_indices[i + offset]]
    cluster_counts1 = Counter(labels)
    for cluster, count in cluster_counts1.items():
        print(f"Cluster {cluster}: {count} points")

    print(labels)


    filtered_kp1 = np.array(filtered_kp1)
    filtered_des1 = np.array(filtered_des1)
    filtered_kp2 = np.array(filtered_kp2)
    filtered_des2 = np.array(filtered_des2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(filtered_des1, filtered_des2)
    matches = sorted(matches, key=lambda x: x.distance)
    num_good_matches = int(len(matches) * 0.8)
    good_matches = matches[:num_good_matches]
    pts1 = np.float32([filtered_kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([filtered_kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    result_img = cv2.drawMatches(img1, filtered_kp1, img2, filtered_kp2, good_matches, None, flags=2)

    cv2.imshow('matches', result_img[::, ::])

    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    result_img_ransac = cv2.drawMatches(img1, filtered_kp1, img2, filtered_kp2, good_matches, None, **draw_params)

    cv2.imshow('RANSAC', result_img_ransac)
    probabilities = np.ones((len(matchesMask),), dtype=np.float64)
    correspondences = np.hstack((pts1.reshape(-1, 2), pts2.reshape(-1, 2))).astype(np.float64)
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]
    H, mask = pymagsac.findHomography(
        correspondences, w1, h1, w2, h2, probabilities,
        sampler=4, use_magsac_plus_plus=True, sigma_th=0.8, conf=0.99, min_iters=50, max_iters=1000, partition_num=5
    )

    # print(H)
    height, width = img2.shape[:2]
    aligned_img1 = cv2.warpPerspective(img1, H, (width, height))
    h, w = img1.shape[:2]
    original_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    transformed_corners = cv2.perspectiveTransform(original_corners, M)
    return aligned_img1, H ,transformed_corners
