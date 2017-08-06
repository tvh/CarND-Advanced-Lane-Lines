import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
import os

# prepare object points
nx = 9
ny = 6

def calibrate(images, nx, ny):
    """
    Calibrate a camara with chessboard images. Returns the result of
    `cv2.calibrateCamera`.
    """

    # Arrays to store object- and image points
    objpoints = []
    imgpoints = []

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    for fname in images:
        img = cv2.imread(fname)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add to collected points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration1*.jpg')

ret, mtx, dist, rvecs, tvecs = calibrate(images, nx, ny)

# Write the calibration parameters to disk
output = open('calibration.pkl', 'wb')
pickle.dump({'mtx': mtx, 'dist': dist}, output)
output.close()

# transform the test images and save
test_images = glob.glob('./test_images/*.jpg')

for fname in test_images:
    img = cv2.imread(fname)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('output_images/undistorted_'+os.path.basename(fname), undist)

