import cv2
import numpy as np

def find_intersections(H, img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    pts1_top = np.float32([[0, 0], [w1, 0]]).reshape(-1, 1, 2)
    pts1_bottom = np.float32([[0, h1], [w1, h1]]).reshape(-1, 1, 2)

    transformed_corners_img1_top = cv2.perspectiveTransform(pts1_top, H)
    transformed_corners_img1_bottom = cv2.perspectiveTransform(pts1_bottom, H)

    left_top_img1 = transformed_corners_img1_top[0].ravel()
    right_top_img1 = transformed_corners_img1_top[1].ravel()

    left_bottom_img1 = transformed_corners_img1_bottom[0].ravel()
    right_bottom_img1 = transformed_corners_img1_bottom[1].ravel()

    left_top_img2 = np.array([0, 0])
    right_top_img2 = np.array([w2, 0])
    left_bottom_img2 = np.array([0, h2])
    right_bottom_img2 = np.array([w2, h2])

    all_corners = np.concatenate((
        transformed_corners_img1_top, transformed_corners_img1_bottom,
        [[left_top_img2], [right_top_img2], [left_bottom_img2], [right_bottom_img2]]
    ), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    transformation_dist = np.array([-x_min, -y_min])

    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return None

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div

        if min(line1[0][0], line1[1][0]) <= x <= max(line1[0][0], line1[1][0]) and min(line1[0][1], line1[1][1]) <= y <= max(line1[0][1], line1[1][1]):
            if min(line2[0][0], line2[1][0]) <= x <= max(line2[0][0], line2[1][0]) and min(line2[0][1], line2[1][1]) <= y <= max(line2[0][1], line2[1][1]):
                return x, y
        return None

    intersections = []

    lines_img1 = [
        [left_top_img1, right_top_img1],
        [left_bottom_img1, right_bottom_img1],
        [left_top_img1, left_bottom_img1],
        [right_top_img1, right_bottom_img1]
    ]

    lines_img2 = [
        [left_top_img2, right_top_img2],
        [left_bottom_img2, right_bottom_img2],
        [left_top_img2, left_bottom_img2],
        [right_top_img2, right_bottom_img2]
    ]

    for line1 in lines_img1:
        for line2 in lines_img2:
            intersection = line_intersection(line1, line2)
            if intersection:
                intersections.append(intersection)

    intersections = np.array(intersections)

    x_range = (0, x_max - x_min)
    y_range = (0, y_max - y_min)

    valid_mask = (
        (x_range[0] <= intersections[:, 0] + transformation_dist[0]) & (intersections[:, 0] + transformation_dist[0] <= x_range[1]) &
        (y_range[0] <= intersections[:, 1] + transformation_dist[1]) & (intersections[:, 1] + transformation_dist[1] <= y_range[1])
    )

    valid_intersections = intersections[valid_mask]

    intersection_dict = {}
    intersection_1={}
    for i, intersection in enumerate(valid_intersections):
        label = chr(65 + i)
        transformed_intersection = (int(intersection[0] + transformation_dist[0]), int(intersection[1] + transformation_dist[1]))
        intersection_dict[label] = transformed_intersection
        intersection_1[label] = intersection
    return intersection_dict, intersection_1

