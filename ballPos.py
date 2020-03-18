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
import validators as vd
from transformfield import get_transform, get_ROI
from detectball import get_filter_param 
from detectball import ball as bp
import threading
import time
from multiprocessing import Process


class ball():
     # Object to acquire ball position
     def __init__(self, cam = 0, stream = 'off', fieldSize = (1200,800) ):
          # general variable
          self.fieldSize = fieldSize
          self.offset = 100
          self.stream = stream
          self.pos = np.zeros(2)

          # initialize video stream.  
          # define webcamera, something to change
          #if vd.url(cam):
           #    self.cap = cv2.VideoCapture(cam)
          #else:
          self.cap = cv2.VideoCapture(0) 
          
          # get image input/ first frame
          time.sleep(1)           
          _, img_init = self.cap.read()
          # get transformation
          self.trans, self.invTrans = get_transform(img_init,
                                   self.fieldSize, self.offset)
          # get color mask value
          self.mask = get_ROI(img_init, self.invTrans, self.fieldSize, self.offset)
          img_init = cv2.bitwise_and(img_init, self.mask) 
          self.colorParam = get_filter_param(img_init)
          cv2.destroyAllWindows()
          # start process
          #thread = threading.Thread(target=self.runBallPos, args=())
          #thread.start()
          
          # alternative
          p = Process(target=self.runBallPos, args=())
          p.start()
          p.join()
          time.sleep(1)           

     def locateBall(self, image):
          # cut image
          image = cv2.bitwise_and(image, self.mask)
          # detect ball using color mask 
          # get ball position in image coordinate
          ball = bp(image, self.colorParam)
          ball = np.append(ball,1)
          # transform to field orientation
          ball = self.trans.dot(ball)
          ball = ball/ball[-1]          
          self.pos = np.delete(ball,2) # in cm

     def runBallPos(self):
          # running in background(threading), 
          # possible to use multiprocessing
          while True:
               _, image = self.cap.read()
               self.locateBall(image)
               if True:
                    cv2.putText(image, 'a', , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 5, cv2.LINE_AA) 
                    cv2.imshow("frame",image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                         break         
                    print('oio')
               print('io')
          # When everything done, release the capture
          cap.release()
          cv2.destroyAllWindows()
     
     def getBallPos(self):
          return self.pos
          

if __name__=='__main__':
    url = 'http://192.168.1.16:4747/video'
    a = ball(cam=0,stream='on')
    while True:
         print(a.getBallPos())
         time.sleep(0.2)