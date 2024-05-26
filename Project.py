import cv2
import numpy as np
from sliding import fit_polynomial
def calibration():
    for i in range(1, 21):
        im_url = f"./camera_cal/calibration{i}.jpg"
        img = cv2.imread(im_url)
        nx = 9
        ny = 6
        objpoints = []
        imgpoints=[]
        objp = np.zeros((6*9,3) , np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        mtx_s=[]
        dist_s=[]
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            img2 = np.copy(img)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            mtx_s.append(mtx)
            dist_s.append(dist)
    answer=0
    answer2=0
    for i in range(len(mtx_s)):
        answer+=mtx_s[i]
    for i in range(len(dist_s)):
        answer2+=dist_s[i]
    answer= answer/len(mtx_s)
    answer2= answer2/len(dist_s)

    return(answer,answer2)

def pipeline(img, s_thresh=(140, 255), sx_thresh=(40, 120)):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls = cv2.GaussianBlur(hls,(9,9),0)
    l = hls[:,:,1]
    s = hls[:,:,2]
    
    sobelx = cv2.Sobel(l, cv2.CV_64F, 1, 0) 
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    sxbinary =sxbinary*255
    
    sobels = cv2.Sobel(s, cv2.CV_64F, 1, 0) 
    abs_sobels = np.absolute(sobels)
    scaled_sobel = np.uint8(255*abs_sobels/np.max(abs_sobels))
    s_binary = np.zeros_like(s)
    s_binary[(s >= s_thresh[0]) & (s <= s_thresh[1])] = 1
    s_binary = s_binary*255

    color_binary = s_binary+sxbinary

    return color_binary

def warp(img):
    src = np.float32([[530, 477], [775, 477], [1120, 670], [235, 670]])
    dst = np.float32([[0, 0], [200, 0], [200, 300], [0, 300]])
    M = cv2.getPerspectiveTransform(src, dst) 
    miv= cv2.getPerspectiveTransform(dst, src) 
    warped_img =cv2.warpPerspective(img, M, (200,300), flags=cv2.INTER_LINEAR)
    return warped_img,miv

def draw_lane(original_img, binary_img, l_fit, r_fit, Minv):
    new_img = np.copy(original_img)
    if l_fit is None or r_fit is None:
        return original_img
   
    # warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.zeros((300,200,3), dtype='uint8')
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)
    left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]

   
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))


    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)
    
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 
   
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result

def combine(result,warp):
    overlay_height = warp.shape[0]
    overlay_width = warp.shape[1]
    top_left_y = 0
    top_left_x = 0
    bottom_right_y = top_left_y + overlay_height
    bottom_right_x = top_left_x + overlay_width
    result[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = warp
    return(result)

mtx,dist=calibration()
def image_processing():
    for i in range(1,9):
        im_f = f"./test_images/test{i}.jpg"
        img=cv2.imread(im_f)
        image = np.copy(img)
        warp_orginal=warp(image)
        dst = cv2.undistort(image, mtx, dist, None, mtx)
        pipeline_result = pipeline(dst)
        warp_image=warp(pipeline_result)
        result_window = fit_polynomial(warp_image)

        cv2.imshow('warp_orginal',warp_orginal)
        cv2.imshow('result_window',result_window)
        cv2.waitKey(0)

def video_processing():
    cap = cv2.VideoCapture("project_video.mp4")
    while(cap.isOpened()):
        _, img = cap.read()
        image = np.copy(img)
        warp_orginal=warp(image)
        dst = cv2.undistort(image, mtx, dist, None, mtx)
        pipeline_result = pipeline(dst)
        warp_image,miv=warp(pipeline_result)
        result_window,distance, left, right,degree = fit_polynomial(warp_image)
        output=draw_lane(image, pipeline_result, left, right, miv)


        output = combine(output,result_window)

        text = f"DISTANCE: {round(distance,2)}"
        position = (1000, 50) 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 0) 
        thickness = 2
        cv2.putText(output, text, position, font, font_scale, color, thickness)

        text = f"DEGREE:{round(degree,4)}"
        position = (1003, 100) 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 0) 
        thickness = 2
        cv2.putText(output, text, position, font, font_scale, color, thickness)

        cv2.imshow('Output',output)
        # cv2.imshow('Output',result_window)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break



    cap.release()
    cv2.destroyAllWindows()

# image_processing()
video_processing()
