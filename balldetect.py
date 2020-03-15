import cv2 
import numpy as np
from transform.transform import get_transform


# source: @mrhwick, medium.com
def get_ROI(image):
    mask = np.zeros_like(image)
    channel_count = image.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# source pysource.com
def nothing(x):
    pass


def get_filter_param(image):    
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    while True:
        hsv =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = get_ROI(hsv)
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        lower_blue = np.array([l_h, l_s, l_v])
        upper_blue = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow('filtered', result)
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    cv2.destroyAllWindows()        

if __name__=='__main__':
    img_init = cv2.imread('field.jpg')
    trans = get_transform(img_init)
    inv_trans = np.linalg.inv(trans)
    fieldCrop = np.array([
		[0, 0, 1],
		[maxWidth + 2*offset,0, 1],
		[maxWidth + 2*offset, maxHeight + 2*offset, 1],
		[0, maxHeight + 2*offset, 1]], dtype = "float32")

    
    get_filter_param(img_init, crop_vertices)

