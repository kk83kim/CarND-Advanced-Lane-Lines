import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle

images = glob.glob('./camera_cal/calibration*.jpg')

# Arrays to store object and image points from all the images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) # x, y coordinates

for fname in images:
    # read in each image
    img = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, draw corners
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        # Draw and display the corners
        # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        # plt.imshow(img)
        # plt.show()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1280, 720), None, None)
distortion_param = {"mtx": mtx, "dist": dist}
pickle.dump(distortion_param, open("distortion_param.p", "wb"))


for fname in images:
    # read in each image
    img = cv2.imread(fname)

    # undistort image
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    # show distorted and undistorted images side by side
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig("./output_images/distortion_correction.png")
    plt.show()