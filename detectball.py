import cv2 
import numpy as np
from .transformfield import get_transform, get_ROI
import imutils

# source pysource.com
def nothing(x):
    pass

def get_filter_param(image): 
    # return HSV value for ball detection using color filtering    
    cv2.namedWindow("filtering..")
    cv2.createTrackbar("L - H", "filtering..", 0, 255, nothing)
    cv2.createTrackbar("L - S", "filtering..", 0, 255, nothing)
    cv2.createTrackbar("L - V", "filtering..", 0, 255, nothing)
    cv2.createTrackbar("U - H", "filtering..", 255, 255, nothing)
    cv2.createTrackbar("U - S", "filtering..", 255, 255, nothing)
    cv2.createTrackbar("U - V", "filtering..", 255, 255, nothing)    
    
    while True:
        hsv =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)        
        l_h = cv2.getTrackbarPos("L - H", "filtering..")
        l_s = cv2.getTrackbarPos("L - S", "filtering..")
        l_v = cv2.getTrackbarPos("L - V", "filtering..")
        u_h = cv2.getTrackbarPos("U - H", "filtering..")
        u_s = cv2.getTrackbarPos("U - S", "filtering..")
        u_v = cv2.getTrackbarPos("U - V", "filtering..")
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
def ballPos(image, colorParam):
    # detect ball using color mask
    colorLower, colorUpper = colorParam
    # resize the image, blur it, 
    # and convert it to the HSV color space
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
    center = (0,0)
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
        '''' # only to test detectball.py
        if radius > 10:
            # draw the circle and centroid on the image,
            # then update the list of tracked points
            cv2.circle(image, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(image, center, 5, (0, 0, 255), -1)
            cv2.putText(image, '({})'.format(center), center,  
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 5, cv2.LINE_AA) 
        center = np.array(center)
        center[1] = center[1]+radius
    # show the frame to our screen
    cv2.imshow("Ball position", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    center = np.array(center)
    return center


if __name__=='__main__':
    fieldSize = (1200,800)
    img_init = cv2.imread('field.jpg')
    trans,invTrans = get_transform(img_init)
    mask = get_ROI(img_init, invTrans)
    img_init = cv2.bitwise_and(img_init, mask)
    colorParam = get_filter_param(img_init)
    pos = ballPos(img_init, colorParam)
    pos = np.append(pos,1)    
    pos = trans.dot(pos)
    pos = pos/pos[-1]
    print(pos)
