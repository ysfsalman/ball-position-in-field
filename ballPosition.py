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
from transformfield import get_transform, get_ROI
from detectball import get_filter_param 
from detectball import ballPos as bp
import threading
import time
from multiprocessing import Process


class ball():
     # Object to acquire ball position
     def __init__(self, cam = 0, stream = 'off', fieldSize = (1200,800) ):
          # general variable
          self.fieldSize = fieldSize
          self.offset = 100
          self.wantStream = stream
          self.pos = np.zeros(2)
          self.posinImg = np.zeros(2)
          # initialize video stream.  
          # define webcamera, something to change
          self.cam = cam
          if type(stream) == str:
               self.cap = cv2.VideoCapture(cam)
          else:
               self.cap = cv2.VideoCapture(0) 
          
          # get image input/ first frame
          time.sleep(1)           
          _, self.frame = self.cap.read()
          img_init = self.frame
          # get transformation
          self.trans, self.invTrans = get_transform(img_init,
                                   self.fieldSize, self.offset)
          # get color mask value
          self.mask = get_ROI(img_init, self.invTrans, self.fieldSize, self.offset)
          img_init = cv2.bitwise_and(img_init, self.mask) 
          self.colorParam = get_filter_param(img_init)
          cv2.destroyAllWindows()
          # start process
          self.thread = threading.Thread(target=self.runBallPos, args=())
          self.thread.start()
          
          # alternative
          #p = Process(target=self.runBallPos, args=())
          #p.start()
          #p.join()
          time.sleep(1)           

     def locateBall(self, image):
          # cut image
          image = cv2.bitwise_and(image, self.mask)
          # detect ball using color mask 
          # get ball position in image coordinate
          ball = bp(image, self.colorParam)
          self.posinImg = ball
          ball = np.append(ball,1)
          # transform to field orientation
          ball = self.trans.dot(ball)
          ball = ball/ball[-1]          
          self.pos = np.delete(ball,2) # in cm
          self.pos = np.around(self.pos,0) # set decimal point
     
     # bug 1: camera failed to read which break the loop 
     # https://www.pyimagesearch.com/2016/12/26/opencv-resolving-nonetype-errors/
     def runBallPos(self):
          # running in background(threading), 
          # possible to use multiprocessing
          while True:
               _, self.frame = self.cap.read()
               image = self.frame
               if image is None:
                    # handle bug 1        
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.cam)
                    time.sleep(0.01)
                    continue
               # bug 2: fail if center return not array
               self.locateBall(image)
               if self.wantStream =='on':
                    self.show(image)
               
     def getBallPos(self):
          return self.pos
          
     def show(self, image):
          position = self.pos
          center = tuple(self.posinImg.astype(int))
          cv2.putText(image, str(position), center, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255, 0, 0) , 3, cv2.LINE_AA)  
          cv2.imshow("frame", image)
          if cv2.waitKey(1) & 0xFF == ord('q'):
               cap.release()
               cv2.destroyAllWindows()
      

if __name__=='__main__':
    # use droidcam to use phone as input 
    url = 'http://192.168.1.16:4747/video'
    a = ball(cam=url,stream ='on')
    while True:
         print(f'Ball Position: {a.getBallPos()}')
         time.sleep(0.5)


         