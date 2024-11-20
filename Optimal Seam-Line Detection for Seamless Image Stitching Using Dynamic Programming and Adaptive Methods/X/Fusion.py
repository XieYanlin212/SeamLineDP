import cv2
import numpy as np
import matplotlib.pyplot as plt

alpha_left = 1
beta_left = 0
alpha_right = 0
beta_right = 1

def image_fusion(panorama,overlap_mask,adjusted_seam,transform_img1,transform_img2,W,H,seam,mask1,mask2):
    overlap_mask = cv2.cvtColor(overlap_mask, cv2.COLOR_BGR2GRAY)
    for y in range(panorama.shape[0]):
        for x in range(panorama.shape[1]):
            if overlap_mask[y, x] > 0:
                for adjusted_seam_y, adjusted_seam_x in adjusted_seam:
                    if y == adjusted_seam_y:
                        if x < adjusted_seam_x:
                            panorama[y, x] = np.clip(
                                alpha_left * transform_img1[y, x] + beta_left * transform_img2[y, x], 0, 255)
                        else:
                            panorama[y, x] = np.clip(
                                alpha_right * transform_img1[y, x] + beta_right * transform_img2[y, x], 0, 255)
            else:
                if len(mask1.shape) == 3 and mask1.shape[2] == 3:
                    mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
                if mask1[y, x] > 0:
                    panorama[y, x] = transform_img1[y, x]
                if len(mask2.shape) == 3 and mask2.shape[2] == 3:
                    mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
                elif mask2[y, x] > 0:
                    panorama[y, x] = transform_img2[y, x]

    output_image_path = "Location"
    # cv2.imwrite(output_image_path, panorama)
    # panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
    # plt.imshow(panorama, cmap='gray')
    # plt.title('Blended Panorama')
    # plt.axis('off')
    # plt.show()
    #
    # for seam_y_prime, seam_x_prime in seam:
    #     seam_y_prime = int(round(seam_y_prime))
    #     seam_x_prime = int(round(seam_x_prime))
    #     adjusted_seam_x = int(seam_x_prime + W)
    #     adjusted_seam_y = int(seam_y_prime + H)
    #     cv2.circle(panorama, (adjusted_seam_x, adjusted_seam_y), 1, (255, 0, 0), -1)
    return panorama