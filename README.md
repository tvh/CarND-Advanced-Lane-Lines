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

Here I will describe the individual steps of the pipeline briefly.
All the code for this is located in `pipeline.py`
Throughout this section I will use the following image to illustrate the individual steps:

![source image](test_images/test5.jpg)

More examples can be found in `output_images`.

### 1. Distortion-correction

After loading the calibration parameters, I just use `cv2.undistort` to get an undistorted image.
This is the result:

![undistorted](output_images/undistorted_test5.jpg)

### 2. Perspective transform

After undistoring th image, I transform it to a top-down view.
Code for this step can be found is in the function `warp`.

I decided to use 256x512 as the target image size, as this is more similar to the actual road dimensions.
I used gimp to extract goo pixel coordinates for the source image and used 1/4 and 3/4 for the horizontal target points respectively.

The mapping I used:

| Source        | Destination   |
|:-------------:|:-------------:|
| 596, 450      | 64, 0         |
| 688, 450      | 192, 0        |
| 1048, 681     | 192, 512      |
| 276, 681      | 64, 512       |

This results in the followig output:

| Straight road                                         | Example image                              |
|:-----------------------------------------------------:|:------------------------------------------:|
| ![Straight](output_images/warped_straight_lines1.jpg) | ![Example](output_images/warped_test5.jpg) |

### 3. Extracting binary image

To extract a binary image I use a combination of thresholds on color and the sobel operator.
I found that using the `R` channel exclusively is enough to get a decent result.
Code for this step can be found in lines 26-70 (`thresh_r`, `thresh_sobel`, `cleanup_thresholds`, `combine_thresholds`).

I first generate the individual thresholds.
To do this, I use `cv2.adaptiveThreshold`.
I found that this improved the output somewhat over the non-adaptive version.

For the sobel operator, I use a blurred image as input.
This makes it more predictable where exactly the lane boundaries are found and removes much of the noise.

I then remove all detected regions that are too wide to represent a lane line.
This is easily done by subtracting horizontally shifted versions of the found binary images.
Afterwards I am left with the following for the 3 channels:

![thresholded](output_images/warped_test5.jpg)

As you can see, not all masks are always successful.
To finally find the real lane line pixels, I use a majority vore between the 3 masks.
I know that for a good lane detection the sobel masks are offset by a fixed amount from the actual lane line.
This means I can just translate them back onto the color threshold masks, add them all up and get the final result via majority vote:

![thresholded](output_images/combined_test5.jpg)

### 4. Finding lane line pixels and fitting curve

### 5. Road curvature and vehicle position

### 6. Plotting lane back on original image

## Pipeline (video)

## Discussion
