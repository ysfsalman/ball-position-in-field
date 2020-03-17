import cv2 
import numpy as np
from transform.transform import get_transform, get_ROI
from test import test
import imutils

def get_ROI(image):
    # return region of interest of the input image
    _,invTrans = get_transform(image)
    maxWidth, maxHeight = fieldSize
    offset = 100
    
    fieldCrop = np.array([
        [0, 0, 1],
        [maxWidth + 2*offset, 0, 1],
        [maxWidth + 2*offset, maxHeight + 2*offset, 1],
        [0, maxHeight + 2*offset, 1]], dtype = "float32")
    
    crop = np.zeros(np.shape(fieldCrop),dtype="float32")
    
    for i in range(len(fieldCrop)):		
        crop[i] = invTrans.dot(fieldCrop[i])
        crop[i] = crop[i]/crop[i][-1]		
        # cv2.circle(image, tuple(crop[i][:-1]), 10, (255,0, 0), -1)
    
    crop = (crop[:,[0,1]])
    
    if crop[3,[1]] < crop[1,[1]]:
        # correct error if crop points incorrect
        crop[3] = [crop[2][0]+100, crop[0][1]+500]
        
    #test(image,crop)    
    mask = np.zeros_like(image)
    match_mask_color = (255,) * image.shape[2]
    cv2.fillPoly(mask, np.int32([crop]), match_mask_color)
    image = cv2.bitwise_and(image, mask)        
    return image

# source pysource.com
def nothing(x):
    pass

def get_filter_param(image): 
    # return HSV value for ball detection using color filtering
    image = get_ROI(image)  
    
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)    
    
    
    while True:
        hsv =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)        
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        colorLower = np.array([l_h, l_s, l_v])
        colorUpper = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, colorLower, colorUpper)
        result = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow('filtering..', result)
        key = cv2.waitKey(1)
        if key == 27:
            break
    return colorLower, colorUpper        

#source: pyimagesearch.com
def ball(image):
    image = get_ROI(image) # this seems not necessary
    colorLower, colorUpper = get_filter_param(image)  
    # resize the image, blur it, and convert it to the HSV
    # color space
    #image = imutils.resize(image, width=600)
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for selected ball color, then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the image,
            # then update the list of tracked points
            cv2.circle(image, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(image, center, 5, (0, 0, 255), -1)
            cv2.putText(image, '({})'.format(center), center,  
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 5, cv2.LINE_AA) 
    
    # show the frame to our screen
    cv2.imshow("Ball position", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.array(center)


if __name__=='__main__':
    fieldSize = (1200,800)
    img_init = cv2.imread('field.jpg')
    #get_filter_param(img_init)
    ball = ball(img_init)
    ball = np.append(ball,1)
    print(ball)
    trans,_ = get_transform(img_init)
    ball = trans.dot(ball)
    ball = ball/ball[-1]
    print(ball)
