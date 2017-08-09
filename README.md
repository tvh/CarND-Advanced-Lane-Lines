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

## Camera Calibration

The code for this step can be found in `calibration.py`.
I used opencv's `cv2.findChessboardCorners` and `cv2.calibrateCamera` to extract the points and do the calibration.
To undistort the images after, I used `cv2.undistort`.

Results of this process for all calibration images can be found in `output_images`.
Here an example:
![undistorted](output_images/undistorted_calibration1.jpg)

To avoid having to calibrate the camera every time, I saved the result to `calobration.pkl`.

## Pipeline (single images)

Her I will describe the individual steps of the pipeline briefly.
To do this, I will use the following image:
![source image](test_images/test5.jpg)

### 1. Distortion-correction

After loading the calibration parameters, I just use `cv2.undistort` to get an undistorted image.
This is the result:

![undistorted](output_images/undistorted_test5.jpg) |

### 2. Perspective transform

### 3. Extracting binary image

### 4. Finding lane line pixels and fitting curve

### 5. Road curvature and vehicle position

### 6. Plotting lane back on original image

## Pipeline (video)

## Discussion
