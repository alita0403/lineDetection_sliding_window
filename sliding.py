import numpy as np
import matplotlib.pyplot as plt
import cv2

def find_histogram(img):
    return np.sum(img[img.shape[0]//2:,:], axis=0)

def find_line(img):
    Histogram = find_histogram(img)
    out_img = np.dstack((img, img, img))
    midpoint = np.int64(Histogram.shape[0]//2)
    leftx_base = np.argmax(Histogram[:midpoint])
    rightx_base = np.argmax(Histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 20
    minpix = 5

    window_height = np.int64(img.shape[0]//nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base


    left_lines = []
    right_lines = []

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        

        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        # (win_xleft_high,win_y_high),(0,255,0), 2) 
        # cv2.rectangle(out_img,(win_xright_low,win_y_low),
        # (win_xright_high,win_y_high),(0,255,0), 2) 
        

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        

        left_lines.append(good_left_inds)
        right_lines.append(good_right_inds)
        

        if len(good_left_inds) > minpix:
            leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))


    try:
        left_lines = np.concatenate(left_lines)
        right_lines = np.concatenate(right_lines)
    except ValueError:
        pass


    leftx = nonzerox[left_lines]
    lefty = nonzeroy[left_lines] 
    rightx = nonzerox[right_lines]
    righty = nonzeroy[right_lines]

    return leftx, lefty, rightx, righty, out_img,midpoint

def fit_polynomial(img):
    leftx, lefty, rightx, righty, result , midpoint = find_line(img)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        mid_fitx = ((right_fit[0]+left_fit[0])/2)*ploty**2 + ((right_fit[1]+left_fit[1])/2)*ploty + ((right_fit[2]+left_fit[2])/2)
        def_mid_fitx= ((right_fit[0]+left_fit[0]))*ploty + ((right_fit[1]+left_fit[1])/2)
        m_d=(right_fit[0]+left_fit[0])
    except TypeError:
        pass
    dis=abs(midpoint-mid_fitx[0])
    y_line = np.stack((left_fitx,ploty),axis=1)
    x_line = np.stack((right_fitx,ploty),axis=1)
    midel_line = np.stack((mid_fitx,ploty),axis=1)


    cv2.polylines(result, np.int32([y_line]), isClosed=False, color=(0, 0, 255), thickness=5)
    cv2.polylines(result, np.int32([x_line]), isClosed=False, color=(255, 0,0 ), thickness=5)
    cv2.polylines(result, np.int32([midel_line]), isClosed=False, color=(0, 255, 0), thickness=2)

    degre = np.arctan(m_d)

    return result,dis,left_fit,right_fit,degre