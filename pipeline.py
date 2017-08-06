import numpy as np
import cv2
import glob
import pickle
import os

def warp(img, src=np.float32([[574,465],[711,465],[1048,681],[276,681]])):
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

def threshold(img):
    """Apply a binary threshold"""
    b, g, r = cv2.split(img)
    binary = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, -75)
    return binary

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
        binary = threshold(warped)
        cv2.imwrite('output_images/binary_'+os.path.basename(fname), binary)

output_test_images()
