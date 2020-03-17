'''
Ball positioning system for drone referee project. 
The system uses vision input from the corner/side 
of the field to acquire ball location, with following pipeline:
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


class ball():
     def __init__(self, fieldSize = (1200,800)):
          self.fieldSize = fieldSize
          self.offset = 100
          # start video stream

          # get image input/ first frame '''
          img_init = cv2.imread('field.jpg') # test from image
          # get transformation
          self.trans, self.invTrans = get_transform(img_init)
          
          
          # get color mask value
          self.colorLow, self.colorUp = getFilterParam(img_init)
          cv2.destroyAllWindows()
          

     def ballPos(self):
          
     ''' get ball location '''
     # detect ball : option : color mask and learning model (e.g YOLO)

     # get ball in image coordinate

     # transform to field orientation
