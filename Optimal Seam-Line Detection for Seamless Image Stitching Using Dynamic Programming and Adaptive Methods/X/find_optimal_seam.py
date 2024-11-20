import cv2
import numpy as np
import random

def find_optimal_seam(overlap_region_img1, overlap_region_img2,intersection_A,intersection_B):

    if len(overlap_region_img1.shape) == 3:
        overlap_region_img1 = cv2.cvtColor(overlap_region_img1, cv2.COLOR_BGR2GRAY)
    if len(overlap_region_img2.shape) == 3:
        overlap_region_img2 = cv2.cvtColor(overlap_region_img2, cv2.COLOR_BGR2GRAY)
    E_color_squared = np.square(overlap_region_img1 - overlap_region_img2)

    def rgb_to_hsi(image):
        """
        Converts an RGB image to the HSI color space.

        Parameters:
        image (numpy array): RGB image.

        Returns:
        H, S, I: The HSI components of the image (each a 2D array).
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = image.astype(np.float32) / 255.0

        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        I = (R + G + B) / 3.0
        min_RGB = np.minimum(np.minimum(R, G), B)
        S = np.zeros_like(I)
        non_zero_I = I != 0
        S[non_zero_I] = 1 - (min_RGB[non_zero_I] / I[non_zero_I])
        numerator = 0.5 * ((R - G) + (R - B))
        denominator = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
        theta = np.arccos(numerator / (denominator + 1e-6))

        H = np.where(B > G, 2 * np.pi - theta, theta)
        H = H / (2 * np.pi)

        return H, S, I

    def compute_hsi_energy(I0_H, I0_S, I0_I, I1_H, I1_S, I1_I, w_3):
        """
        Calculates the energy function E_hsi based on the HSI color space.
        """
        energy_hsi = np.sqrt(
            w_3 * (I0_H - I1_H) ** 2 +
            w_3 * (I0_S - I1_S) ** 2 +
            w_3 * (I0_I - I1_I) ** 2
        )

        return energy_hsi

    I0_H, I0_S, I0_I = rgb_to_hsi(overlap_region_img1)
    I1_H, I1_S, I1_I = rgb_to_hsi(overlap_region_img2)

    w_1 = 0.2
    w_2 = 0.3
    w_3 = 0.5
    energy_hsi = compute_hsi_energy(I0_H, I0_S, I0_I, I1_H, I1_S, I1_I, w_3)

    def compute_gradients(image):

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        direction = np.arctan2(grad_y, grad_x)
        return grad_x, grad_y, magnitude, direction

    def compute_direction_variance(direction):
        mean_direction = np.mean(direction)
        direction_variance = np.mean((direction - mean_direction) ** 2)
        return direction_variance

    def compute_gradient_ratio(grad_x, grad_y):
        mean_grad_x = np.mean(np.abs(grad_x))
        mean_grad_y = np.mean(np.abs(grad_y))
        max_grad_x = np.max(np.abs(grad_x))
        max_grad_y = np.max(np.abs(grad_y))
        return (max_grad_x+mean_grad_x) / (max_grad_y+mean_grad_y)

    def compute_j(variance1, variance2, ratio1, ratio2, alpha=0.5, beta=0.5):
        avg_variance = (variance1 + variance2) / 2
        avg_ratio = (ratio1 + ratio2) / 2
        return alpha * avg_variance + beta * abs(avg_ratio - 1)

    def compute_normalized_direction_variance(direction_variance):
        normalized_variance = direction_variance / np.max(direction_variance)
        return normalized_variance

    def compute_normalized_gradient_ratio(ratio):
        normalized_ratio = np.abs(ratio - 1) / np.max(np.abs(ratio - 1))
        return normalized_ratio

    grad_x1, grad_y1, _, direction1 = compute_gradients(overlap_region_img1)
    grad_x2, grad_y2, _, direction2 = compute_gradients(overlap_region_img2)
    direction_variance1 = compute_direction_variance(direction1)
    direction_variance2 = compute_direction_variance(direction2)
    R1 = compute_gradient_ratio(grad_x1, grad_y1)
    R2 = compute_gradient_ratio(grad_x2, grad_y2)
    direction_variance1_avg=compute_normalized_direction_variance(direction_variance1)
    direction_variance2_avg=compute_normalized_direction_variance(direction_variance2)
    R1_avg = compute_normalized_gradient_ratio(R1)
    R2_avg = compute_normalized_gradient_ratio(R2)

    J = compute_j(direction_variance1_avg, direction_variance2_avg, R1_avg, R2_avg, alpha=0.6, beta=0.4)

    print('J',J)

    def compute_avg_ratio(variance1, variance2, ratio1, ratio2, alpha=0.5, beta=0.5):
        avg_variance = (variance1 + variance2) / 2
        avg_ratio = (ratio1 + ratio2) / 2
        return abs(avg_ratio - 1)

    avg_ratio=compute_avg_ratio(direction_variance1_avg, direction_variance2_avg, R1_avg, R2_avg, alpha=0.6, beta=0.4)
    print(f"R1: {R1}, R2: {R2}, avg_ratio: {avg_ratio}")


    def compute_gradient_r(grad_x, grad_y):
        max_grad_x = np.max(np.abs(grad_x))
        max_grad_y = np.max(np.abs(grad_y))
        return max_grad_x ,max_grad_y

    max_grad_x1, max_grad_y1=compute_gradient_r(grad_x1, grad_y1)
    print(f"max_grad_x1: {max_grad_x1}, max_grad_y: {max_grad_y1}")

    if J < 0.5:
        print("Using the first formula (sum of squares formula)")
        if len(overlap_region_img1.shape) == 3:
            overlap_region_img1 = cv2.cvtColor(overlap_region_img1, cv2.COLOR_BGR2GRAY)
        if len(overlap_region_img2.shape) == 3:
            overlap_region_img2 = cv2.cvtColor(overlap_region_img2, cv2.COLOR_BGR2GRAY)
        diff = overlap_region_img1 - overlap_region_img2
        grad_x = cv2.Sobel(diff, cv2.CV_64F, 1, 0, ksize=3)
        grad_x_squared = np.square(grad_x)
        grad_y = cv2.Sobel(diff, cv2.CV_64F, 0, 1, ksize=3)
        grad_y_squared = np.square(grad_y)
        E_geometry = grad_x_squared + grad_y_squared
    else:
        print("Using the second formula (product formula)")
        if len(overlap_region_img1.shape) == 3:
            overlap_region_img1 = cv2.cvtColor(overlap_region_img1, cv2.COLOR_BGR2GRAY)
        if len(overlap_region_img2.shape) == 3:
            overlap_region_img2 = cv2.cvtColor(overlap_region_img2, cv2.COLOR_BGR2GRAY)
        Sx1 = cv2.Sobel(overlap_region_img1, cv2.CV_64F, 1, 0, ksize=3)
        Sy1 = cv2.Sobel(overlap_region_img1, cv2.CV_64F, 0, 1, ksize=3)
        Sx2 = cv2.Sobel(overlap_region_img2, cv2.CV_64F, 1, 0, ksize=3)
        Sy2 = cv2.Sobel(overlap_region_img2, cv2.CV_64F, 0, 1, ksize=3)
        E_geometry = (Sx1 - Sx2) * (Sy1 - Sy2)

        def zscore_normalize(E_geometry):
            mean_val = np.mean(E_geometry)
            std_val = np.std(E_geometry)
            if std_val == 0:
                return E_geometry
            return (E_geometry - mean_val) / std_val

        E_geometry_normalized = zscore_normalize(E_geometry)
        def zscore_normalize(E):
            mean_val = np.mean(E)
            std_val = np.std(E)
            if std_val == 0:
                return E
            return (E - mean_val) / std_val

        E_color_squared_normalized = zscore_normalize(E_color_squared)
        def zscore_normalize(E):
            mean_val = np.mean(E)
            std_val = np.std(E)
            if std_val == 0:
                return E
            return (E - mean_val) / std_val

        energy_hsi_normalized = zscore_normalize(energy_hsi)

    energy = w_1*E_color_squared_normalized+w_2*E_geometry_normalized + w_3*energy_hsi_normalized

    dp = np.zeros_like(energy)
    dp[0, :] = energy[0, :]
    for i in range(1, dp.shape[0]):
        for j in range(dp.shape[1]):
            min_cost = dp[i - 1, j]
            if j > 0:
                min_cost = min(min_cost, dp[i - 1, j - 1])
            if j < dp.shape[1] - 1:
                min_cost = min(min_cost, dp[i - 1, j + 1])
            dp[i, j] = energy[i, j] + min_cost


    start_point = (int(intersection_A[1]), int(intersection_A[0]))
    end_point = (int(intersection_B[1]), int(intersection_B[0]))

    seam = []
    i, j = end_point
    seam.append((i, j))

    def directional_preference(candidates, last_j):
        preferred_candidates = []
        for candidate in candidates:
            cost, j_offset = candidate
            distance = abs(j_offset - last_j)
            adjusted_cost = cost + 0.1 * distance
            preferred_candidates.append((adjusted_cost, j_offset))
        return preferred_candidates


    threshold = 1.1

    for i in range(dp.shape[0] - 2, -1, -1):
        j = seam[-1][1]
        candidates = []

        for offset in range(-5, 6):
            if 0 <= j + offset < dp.shape[1]:
                candidates.append((dp[i, j + offset], j + offset))

        preferred_candidates = directional_preference(candidates, j)
        best_candidate = min(preferred_candidates, key=lambda x: x[0])
        second_best_candidates = [c for c in preferred_candidates if c[0] <= best_candidate[0] * threshold]

        if second_best_candidates:
            j = random.choice(second_best_candidates)[1]
        else:
            j = best_candidate[1]
        seam.append((i, j))

    seam[-1] = start_point

    if seam[0] != end_point:
        seam[0] = end_point

    def fill_discontinuous_seam(seam):
        filled_seam = [seam[0]]
        for idx in range(1, len(seam)):
            prev_i, prev_j = filled_seam[-1]
            curr_i, curr_j = seam[idx]
            while abs(curr_i - prev_i) > 1 or abs(curr_j - prev_j) > 1:
                step_i = 1 if curr_i > prev_i else -1
                step_j = 1 if curr_j > prev_j else -1

                filled_seam.append((prev_i + step_i, prev_j + step_j))
                prev_i, prev_j = filled_seam[-1]
            filled_seam.append(seam[idx])
        return filled_seam

    seam = fill_discontinuous_seam(seam)
    seam.reverse()
    return seam