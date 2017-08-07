import numpy as np
import cv2
import glob
import pickle
import os

def warp(img, src=np.float32([[596,450],[688,450],[1048,681],[276,681]])):
    """Warp the image for top-down view"""
    img_size = (256,512)
    x1 = img_size[0]/4
    x2 = img_size[0]/4*3
    dst = np.float32([[x1,0],[x2,0],[x2,img_size[1]],[x1,img_size[1]]])

    # Calculate the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Do the transform
    warped = cv2.warpPerspective(img, M, img_size)

    return warped, M

def thresh_r(img):
    """Apply a binary threshold to the R channel"""
    b, g, r = cv2.split(img)
    binary = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, -70)
    return binary

def thresh_sobel(img):
    """Apply a binary threshold on the sobel operator"""
    b, g, r = cv2.split(img)
    sobelx = cv2.Sobel(cv2.blur(r, (25,17)), cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sobel_binary = cv2.adaptiveThreshold(scaled_sobel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, -40)
    return sobel_binary

def load_calibration_params():
    cal_file = open('calibration.pkl','rb')
    res = pickle.load(cal_file)
    cal_file.close()
    return res['mtx'], res['dist']

def output_test_images():
    """Save the individual stages applied to the test images"""
    mtx, dist = load_calibration_params()

    test_images = glob.glob('./test_images/*.jpg')
    for fname in test_images:
        img = cv2.imread(fname)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite('output_images/undistorted_'+os.path.basename(fname), undist)
        warped, M = warp(undist)
        cv2.imwrite('output_images/warped_'+os.path.basename(fname), warped)
        binary_r = thresh_r(warped)
        binary_sobel = thresh_sobel(warped)
        thresholded = cv2.merge([np.zeros_like(binary_r), binary_r, binary_sobel])
        cv2.imwrite('output_images/thresholded_'+os.path.basename(fname), thresholded)

output_test_images()
