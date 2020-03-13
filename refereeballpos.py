'''
Ball positioning system for drone referee project. 
The system uses vision input from the corner/side of the field to acquire ball location, with following pipeline:
-> get image 
-> get transformation matrix from image coor to field coor 
-> color filter ball to get mask 
-> locate mask location in image 
-> transform location from image coordinate to field coordinate 
-> return value

written by  : Yusuf Salman, github.com/ysfsalman
date        : March 2020 
'''

import cv2
import numpy as np
from transform.transform import get_transform as gt


def get_ref_pts_in_img(event,x,y, flags, params):
    # get reference point from image
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pts_in_img.append((x,y))
        #print(ref_pts_in_img)

def pin_ref_pts():
    for point in ref_pts_in_img:
        cv2.circle(img_init, point, 5, (0, 0, 255), -1)
    cv2.imshow("Set Reference",img_init)
    #cv2.waitKey(1)


''' get video stream '''


''' get image input/ first frame '''
img_init = cv2.imread('field.jpg')


''' get transformation '''
ref_pts_in_img = []
cv2.namedWindow("Set Reference")
cv2.setMouseCallback("Set Reference", get_ref_pts_in_img)
while True:
    # prompt user to choose for point
    pin_ref_pts()
    if cv2.waitKey(1) & len(ref_pts_in_img) >= 4:
        pin_ref_pts()
        break
   
warped = gt(img_init, ref_pts_in_img)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()


''' get ball location '''
# transform to field orientation
