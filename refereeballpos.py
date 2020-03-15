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
from transform.transform import get_transform


''' get video stream '''


''' get image input/ first frame '''
img_init = cv2.imread('field.jpg') # test from image


''' get transformation '''
trans = get_transform(img_init)

# test

#print(np.linalg.inv(trans))
offset = 100
warped = cv2.warpPerspective(img_init, trans, \
     (1200+2*offset, 800+2*offset))

cv2.imshow("Warped", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()


''' get ball location '''
# detect ball : option : color mask and learning model (e.g YOLO)

# get ball in image coordinate

# transform to field orientation
