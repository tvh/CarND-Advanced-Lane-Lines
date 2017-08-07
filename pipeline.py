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
    binary_r = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, -70)
    return cleanup_threshold(binary_r)

def thresh_sobel(img):
    """Apply a binary threshold to the sobel operator"""
    b, g, r = cv2.split(img)
    sobelx = cv2.Sobel(cv2.blur(r, (25,17)), cv2.CV_64F, 1, 0)
    min_sobel = np.min(sobelx)
    max_sobel = np.max(sobelx)
    scaled_sobel_l = np.uint8(255*(sobelx-min_sobel)/(max_sobel-min_sobel))
    scaled_sobel_r = 255-scaled_sobel_l
    sobel_binary_l = cv2.adaptiveThreshold(scaled_sobel_l, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, -25)
    sobel_binary_r = cv2.adaptiveThreshold(scaled_sobel_r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, -25)
    return cleanup_threshold(sobel_binary_l), cleanup_threshold(sobel_binary_r)

def cleanup_threshold(binary):
    """
    Cleans up a given thresholded channel.
    This removes many false positives by removing wide finds
    """
    height, width = binary.shape[:2]
    trans_m_l = np.float32([[1, 0, 12], [0, 1, 0]])
    trans_m_r = np.float32([[1, 0, -12], [0, 1, 0]])
    shifted_binary_l = cv2.warpAffine(binary, trans_m_l, (width, height))
    shifted_binary_r = cv2.warpAffine(binary, trans_m_r, (width, height))
    return binary-shifted_binary_l-shifted_binary_r

def combine_thresholds(binary_r, binary_sobel_l, binary_sobel_r):
    """
    Combine the calculated thresholds
    This is done by adding up the different channels with
    horizontally shifted versions of each other.
    """
    height, width = binary_r.shape[:2]
    trans_m_l = np.float32([[1, 0, 12], [0, 1, 0]])
    shifted_sobel_l = cv2.warpAffine(binary_sobel_l, trans_m_l, (width, height))
    trans_m_r = np.float32([[1, 0, -12], [0, 1, 0]])
    shifted_sobel_r = cv2.warpAffine(binary_sobel_r, trans_m_r, (width, height))
    combined = (shifted_sobel_l//3+shifted_sobel_r//3+binary_r//3)
    res = np.zeros_like(combined)
    res[combined>150]=255
    return res

def fit_lane_line(binary_warped, nwindows=9, margin=32, minpix=20):
    """
    Finds the lane lines and fits a curve to it.
    This is largely based on the function presented in the course.
    """
    # Blur the bottom quarter of the image. and take the histogram.
    # This is needed as the detected lines are quite narrow.
    lower_quarter = binary_warped[binary_warped.shape[0]//2:,:]
    blurred = cv2.GaussianBlur(lower_quarter, (31, 1), 11)
    histogram = np.sum(blurred[blurred.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = np.int16(left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2])
    right_fitx = np.int16(right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2])

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    out_img[np.int16(ploty), left_fitx] = [0, 255, 255]
    out_img[np.int16(ploty), right_fitx] = [0, 255, 255]
    return out_img

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
        binary_sobel_l, binary_sobel_r = thresh_sobel(warped)
        thresholded = cv2.merge([binary_sobel_l, binary_r, binary_sobel_r])
        cv2.imwrite('output_images/thresholded_'+os.path.basename(fname), thresholded)
        combined = combine_thresholds(binary_r, binary_sobel_l, binary_sobel_r)
        cv2.imwrite('output_images/combined_'+os.path.basename(fname), combined)
        fitted = fit_lane_line(combined)
        cv2.imwrite('output_images/fitted_'+os.path.basename(fname), fitted)

output_test_images()
