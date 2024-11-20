import cv2
import numpy as np
from X.Feature_matching import align_images
from X.find_optimal_seam import find_optimal_seam
import matplotlib.pyplot as plt
from X.DBSCAN import DBSCAN_EPS
from X.HSV_adjust import adjust_H
from X.HSV_adjust import HSV
from X.transformation import transform
from X.Fusion import image_fusion
from X.intersections import find_intersections

img1 = cv2.imread('image_data/sedona_left_01.png')
img2 = cv2.imread('image_data/sedona_right_01.png')

if img1 is None or img2 is None:
    raise ValueError("One or both images could not be loaded.")

# print(DBSCAN_EPS(img1, img2))

img1 = HSV(img1,img2)

aligned_img1, M,_ = align_images(img1, img2)

overlap_mask = ((aligned_img1[:, :, 0] > 0) & (img2[:, :, 0] > 0) &
                (aligned_img1[:, :, 1] > 0) & (img2[:, :, 1] > 0) &
                (aligned_img1[:, :, 2] > 0) & (img2[:, :, 2] > 0))

if len(overlap_mask.shape) == 3:
    overlap_mask = cv2.cvtColor(overlap_mask, cv2.COLOR_BGR2GRAY)

overlap_mask = overlap_mask.astype(np.uint8)
kernel = np.ones((3, 3), np.uint8)
eroded_overlap_mask = cv2.erode(overlap_mask, kernel, iterations=1)
coords = np.column_stack(np.where(eroded_overlap_mask > 0))

y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0)

overlap_region_img1 = aligned_img1[y_min:y_max+1, x_min:x_max+1]
overlap_region_img2 = img2[y_min:y_max+1, x_min:x_max+1]

mask = np.zeros_like(aligned_img1, dtype=np.uint8)
mask[y_min:y_max, x_min:x_max] = 255


intersection_dict,intersection_1 = find_intersections(M, img1, img2)
# 打印交点
for label, point in intersection_dict.items():
    print(f"{label}: {point}")

POINT1 = intersection_dict['A']
POINT2 = intersection_dict['B']

intersection_A=intersection_1['A']
intersection_B=intersection_1['B']

aligned_img1_overlap = aligned_img1[y_min:y_max, x_min:x_max]
img2_overlap = img2[y_min:y_max, x_min:x_max]
seam = find_optimal_seam(aligned_img1_overlap, img2_overlap,intersection_A,intersection_B)
print('seam',seam)

result_img,transform_img1,x_min_r, y_min_r,x_max_r, y_max_r,transformation_dist,transform_matrix,h1, w1,h2, w2\
    = transform(img1,img2,M)

panorama_size = (x_max_r - x_min_r, y_max_r - y_min_r)
transform_img2 = np.zeros((panorama_size[1], panorama_size[0], 3), dtype=np.uint8)

x_offset = transformation_dist[0]
y_offset = transformation_dist[1]
transform_img2[y_offset:y_offset + h2, x_offset:x_offset + w2] = img2

# transform_img1 = adjust_H(transform_img1,transform_img2)

# cv2.imshow('result_img', result_img)
final_transform_matrix = transform_matrix.dot(M)

t_x = final_transform_matrix[0, 2]
t_y = final_transform_matrix[1, 2]

mask1 = np.zeros_like(transform_img1, dtype=np.uint8)
mask1[transform_img1 > 0] = 1

mask2 = np.zeros_like(transform_img2, dtype=np.uint8)
mask2[transform_img2 > 0] = 1

overlap_mask = cv2.bitwise_and(mask1, mask2)

overlap_x_min = np.min(np.where(overlap_mask.any(axis=0)))
overlap_x_max = np.max(np.where(overlap_mask.any(axis=0)))
overlap_y_min = np.min(np.where(overlap_mask.any(axis=1)))
overlap_y_max = np.max(np.where(overlap_mask.any(axis=1)))

x_offset = transformation_dist[0]
y_offset = transformation_dist[1]

top_left = (x_offset, y_offset)
top_right = (x_offset + w2, y_offset)
bottom_left = (x_offset, y_offset + h2)
bottom_right = (x_offset + w2, y_offset + h2)

panorama = np.zeros_like(result_img, dtype=np.uint8)

h3,w3=result_img.shape[:2]

W = w3 - w2
last_seam_point = seam[-1]
last_seam_y, last_seam_x = last_seam_point

H = y_offset + h2 -last_seam_y-1

adjusted_seam = [(int(round(seam_y + H)),
                  int(round(seam_x + W)))
                 for seam_y, seam_x in seam]

first_seam_y, first_seam_x = adjusted_seam[0]

new_first_seam_y = first_seam_y - 1
new_first_seam_x = first_seam_x

adjusted_seam = [(new_first_seam_y, new_first_seam_x)] + adjusted_seam

panorama = image_fusion(panorama, overlap_mask, adjusted_seam, transform_img1, transform_img2, W, H, seam, mask1, mask2)

panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
plt.imshow(panorama)
plt.title('Blended Panorama')
plt.axis('off')
plt.show()