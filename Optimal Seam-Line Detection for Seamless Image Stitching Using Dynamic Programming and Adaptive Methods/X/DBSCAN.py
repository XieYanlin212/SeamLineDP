import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

def DBSCAN_EPS(img1, img2):
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

    nbrs = NearestNeighbors(n_neighbors=2).fit(keypoints1)
    distances, indices = nbrs.kneighbors(keypoints1)


    sorted_distances = np.sort(distances[:, 1])

    plt.rcParams['font.family'] = 'serif'

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances)
    plt.xlabel('Points', fontsize=20)
    plt.ylabel('Distance to Nearest Neighbor', fontsize=20)
    plt.tick_params(axis='both', labelsize=16)

    plt.grid(True)

    optimal_index = np.argmax(
        np.abs(sorted_distances - np.linspace(sorted_distances[0], sorted_distances[-1], len(sorted_distances))))
    plt.plot(optimal_index, sorted_distances[optimal_index], 'ro')  # 'ro'代表红色圆点

    plt.show()


    def find_optimal_eps(distances):
        line = np.linspace(distances[0], distances[-1], len(distances))

        distances_from_line = np.abs(distances - line)

        optimal_index = np.argmax(distances_from_line)
        return distances[optimal_index]


    optimal_eps = find_optimal_eps(sorted_distances)


    print("Optimal eps:", optimal_eps)

    return optimal_eps

