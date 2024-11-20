import cv2
import numpy as np

def transform(img1,img2,M):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    dst_pts = cv2.perspectiveTransform(pts1, M)
    resulting_points = np.concatenate((pts2, dst_pts), axis=0)
    [x_min_r, y_min_r] = np.int32(resulting_points.min(axis=0).ravel() - 0.5)
    [x_max_r, y_max_r] = np.int32(resulting_points.max(axis=0).ravel() + 0.5)
    transformation_dist = [-x_min_r, -y_min_r]
    transform_matrix = np.array([[1, 0, transformation_dist[0]], [0, 1, transformation_dist[1]], [0, 0, 1]])

    result_img = cv2.warpPerspective(img1, transform_matrix.dot(M), (x_max_r - x_min_r, y_max_r - y_min_r))
    transform_img1=result_img.copy()
    result_img[transformation_dist[1]:h2 + transformation_dist[1],transformation_dist[0]:w2 + transformation_dist[0]] = img2

    return result_img,transform_img1,x_min_r, y_min_r,x_max_r, y_max_r,transformation_dist,transform_matrix,h1, w1,h2, w2