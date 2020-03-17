import numpy as np
import cv2


def test(image, crop):
    imgSize=maskSize= np.array(image.shape)
    maskSize[0:2] *= 2
    #crop[:,0] += image.shape[1]
    #crop[:,1] += image.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX   
    # fontScale 
    fontScale = 1   
    # Blue color in BGR 
    color = (255, 0, 0) 
    newmask = np.zeros(maskSize)
    match_mask_color = (255,) * image.shape[2]
    cv2.fillPoly(newmask, np.int32([crop]), match_mask_color)
    for i in range(len(crop)):	
        cv2.putText(newmask, '{}'.format(i), tuple(crop[i]), font,  
                   fontScale, color, 5, cv2.LINE_AA) 
    cv2.imshow("mask", newmask)
    print(np.int32(crop))



