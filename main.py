import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from threshold_image import threshold_img
from measure_curvature import fit_polynomial
from measure_curvature import measure_curvature
from measure_curvature import Line
from moviepy.editor import VideoFileClip


# 4 points in input image and in warped image to apply for perspective transform
src = np.float32([[180, 720],
                  [600, 445],
                  [680, 445],
                  [1125, 720]])
dst = np.float32([[300, 720],
                  [300, 0],
                  [900, 0],
                  [900, 720]])

img_size = (1280, 720)
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load(open("distortion_param.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def process_image(img):
    # undistort image
    img_undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    # threshold image
    img_threshold = threshold_img(img_undistorted)

    # Apply perspective transform to the image
    img_warped = cv2.warpPerspective(img_threshold, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    # Find polynomial coefficients for left and right lanes
    left_fit_poly, right_fit_poly, left_fit_poly_pixel, right_fit_poly_pixel, img_out = fit_polynomial(img_warped, Line_obj)

    # Calculate the radius of curvature in pixels for both lane lines
    left_curverad, right_curverad = measure_curvature(left_fit_poly, right_fit_poly)
    curvature = (left_curverad + right_curverad) * 0.5

    ### Create image for visualization
    img_warp = np.zeros_like(img_undistorted).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, img_warp.shape[0] - 1, img_warp.shape[0])
    try:
        left_fitx = left_fit_poly_pixel[0] * ploty ** 2 + left_fit_poly_pixel[1] * ploty + left_fit_poly_pixel[2]
        right_fitx = right_fit_poly_pixel[0] * ploty ** 2 + right_fit_poly_pixel[1] * ploty + right_fit_poly_pixel[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(img_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    img_unwarped = cv2.warpPerspective(img_warp, Minv, (img_warp.shape[1], img_warp.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, img_unwarped, 0.3, 0)

    # Calculate vehicle offset
    xm_per_pix = 3.7 / 580
    offset = ((left_fitx[-1] + right_fitx[-1]) / 2 - img.shape[1] / 2) * xm_per_pix

    # Print out curvature and offset on image
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(result, "Radius of Curvature = %d (m)" % curvature, (20, 50), font, 3, (255, 255, 255), 3)
    cv2.putText(result, "Vehicle is %.2f m left of center" % offset, (20, 100), font, 3, (255, 255, 255), 3)

    return result


Line_obj = Line()
output = 'OUT.mp4'
clip1 = VideoFileClip("project_video.mp4")#.subclip(20,21)
clip = clip1.fl_image(process_image)
clip.write_videofile(output, audio=False)


#
# # Read in an image
# # img = cv2.imread('./test_images/straight_lines1.jpg')
# img = cv2.imread('./test_images/test2.jpg')
# # img = cv2.imread('./test_images/test3.jpg')
#
# # undistort image
# img_undistorted = cv2.undistort(img, mtx, dist, None, mtx)
#
# # threshold image
# img_threshold = threshold_img(img_undistorted)
#
# # Apply perspective transform to the image
# img_warped = cv2.warpPerspective(img_threshold, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
#
# # Find polynomial coefficients for left and right lanes
# left_fit_poly, right_fit_poly, left_fit_poly_pixel, right_fit_poly_pixel, img_out = fit_polynomial(img_warped, Line_obj)
# #
# # # Calculate the radius of curvature in pixels for both lane lines
# # left_curverad, right_curverad = measure_curvature(left_fit_poly, right_fit_poly)


# cv2.imwrite("./output_images/img_poly.png", img_out)
# f, ax1 = plt.subplots(1, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img_threshold)
# ax1.set_title('Original Image', fontsize=10)
#
# plt.savefig("./output_images/img_threshold_2.png")
# plt.show()

# img_undistorted = np.dstack([img_undistorted[:, :, 2], img_undistorted[:, :, 1], img_undistorted[:, :, 0]])
# img_warped = np.dstack([img_warped[:, :, 2], img_warped[:, :, 1], img_warped[:, :, 0]])
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img_undistorted)
# ax1.set_title('Original Image', fontsize=10)
# ax2.imshow(img_warped)
# ax2.set_title('Warped Image', fontsize=10)
# plt.savefig("./output_images/img_warped.png")

#
# # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.savefig("test.png")
# # plt.show()