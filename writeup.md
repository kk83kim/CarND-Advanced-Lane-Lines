# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/distortion_correction.png "Checkerboard Undistorted"
[image2]: ./output_images/img_undistorted.png "Undistorted Image"
[image3]: ./output_images/img_threshold.png "Thresholded Image"
[image4]: ./output_images/img_warped.png "Warped Image"
[image5]: ./output_images/img_poly.png "Lines Image"
[image6]: ./output_images/img_final.png "Final Image"
[video1]: ./OUT.mp4 "Video"


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in correct_distortion.py

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.  For the color threshold, I used S channel of the HLS color space.  For the gradient threshold, I used a combination of gradient in x, gradient in y, gradient magnitude, and gradient direction.  The code for this step is contained in threshold_image.py  Here's an example of my output for this step. 
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in line 41 in main.py.  I chose the hardcode the source and destination points.  I found the points with the assumption that the road is perfectly flat and the space between left/right lane when vehicle is driving straight should map to a retangular region on a warped image.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 180, 720      | 300, 720        | 
| 600, 445      | 300, 0      |
| 680, 445     | 900, 0      |
| 1125, 720      | 900, 720        |

I verified that my perspective transform was working as expected by comparing the test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

From the warped binary image, I used a sliding window technique to find the left and right lane-line pixels.  This technique involves first finding peaks in a histogram across horizontal direction of the image starting at the bottom of the image.  Then, moving upward in the image (further along the road) using sliding windows to determine where the lane lines go.  The code for this step is contained in find_lane_pixels function in measure_curvature.py  

Below image shows sliding windows identifying the left and right lane-line pixels.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

After identifying the lane-line pixels, I found a second degree polynomial coefficients that best fit the left-lane and the right-lane pixels.  Using the coefficients, I then calculated the radius of curvature and the position at the bottom of the image for each of the left and right lane lines and averaged the two to finally come up with the radius of the curvature / position to output on a video.  The code for this step is in fit_polynomial and measure_curvature functions in measure_curvature.py.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./OUT.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Although my current implementation works quite well on the given test video, it does not perform as well on the two challenging videos.  The reason is because these challenging videos contain more dynamics (light, traffic, tighter curves, etc.).  There are at least a couple of ways to improve the current implementation.

1.  Fine tuning:  There are many knobs that can be tuned to improve performance.  The major ones are the min / max thresholds for thresholding images in color space and gradients.

2.  Sanity Check:  Current implementation stores past 15 polynomial coefficients for left / right lane lines and averages them to calculate the radius of curvature.  One way to improve this is to do sanity check on the incoming coefficents and discard if they deviate too much from the past.  Even better wasy would be to weigh the coefficients differently based on some factors, such as number of pixels used to calculate the coefficients.

3.  Curve fitting with outlier detection:  When fitting a polynomial on the lane-line pixels, a smarter curve fitting could be used to reject outlier pixels.  RANSAC is a good algorithm to use in this case.



