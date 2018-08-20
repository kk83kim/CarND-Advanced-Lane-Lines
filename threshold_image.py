import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Apply gradient and color thresholds to get binary image
def threshold_img(img, s_thresh=(170, 255), x_thresh=(20, 100), y_thresh=(10, 255), mag_thresh=(10, 255), dir_thresh=(0.7, 1.3)):
    img = np.copy(img)

    # Convert to HLS color space and separate out color channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_x = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Sobel y
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)  # Take the derivative in x
    abs_sobely = np.absolute(sobely)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_y = np.uint8(255 * abs_sobely / np.max(abs_sobely))

    # Sobel Magnitude
    sobel_mag = np.sqrt(np.square(sobelx), np.square(sobely))
    scaled_sobel_mag = np.uint8(255 * sobel_mag / np.max(sobel_mag))

    # Sobel Gradient Direction
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=15)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=15)  # Take the derivative in x
    abs_sobely = np.absolute(sobely)  # Absolute x derivative to accentuate lines away from horizontal
    sobel_dir = np.arctan2(abs_sobely, abs_sobelx)

    # # Threshold x gradient
    # x_binary = np.zeros_like(scaled_sobel_x)
    # x_binary[(scaled_sobel_x >= x_thresh[0]) & (scaled_sobel_x <= x_thresh[1])] = 1
    #
    # # Threshold y gradient
    # y_binary = np.zeros_like(scaled_sobel_y)
    # y_binary[(scaled_sobel_y >= y_thresh[0]) & (scaled_sobel_y <= y_thresh[1])] = 1
    #
    # # Threshold magnitude
    # mag_binary = np.zeros_like(scaled_sobel_mag)
    # mag_binary[(scaled_sobel_mag >= mag_thresh[0]) & (scaled_sobel_mag <= mag_thresh[1])] = 1
    #
    # # Threshold gradient direction
    # dir_binary = np.zeros_like(scaled_sobel_mag)
    # dir_binary[(sobel_dir >= dir_thresh[0]) & (sobel_dir <= dir_thresh[1])] = 1
    #
    # # Combine thresholds from sobel operation
    # combined_sobel = np.zeros_like(x_binary)
    # combined_sobel[((x_binary == 1) & (y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined_sobel = np.zeros_like(scaled_sobel_x)
    combined_sobel[(scaled_sobel_x >= x_thresh[0]) & (scaled_sobel_x <= x_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # # Stack each channel
    # combined_binary = np.dstack((np.zeros_like(combined_sobel), combined_sobel, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(combined_sobel)
    combined_binary[(s_binary == 1) | (combined_sobel == 1)] = 1

    return combined_binary