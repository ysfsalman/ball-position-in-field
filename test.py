import cv2 

img_init = cv2.imread('field.jpg')

cv2.imshow('name',img_init)
cv2.waitKey(0)

cv2.destroyAllWindows()