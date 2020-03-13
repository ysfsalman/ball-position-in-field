import numpy as np
import cv2


def order_points(pts):
	# order the points to top-left, top-right, bottom-right, and bottom-left
	# in case user mistakenly insert wrong order
	# status : WIP
	pts = np.array(pts, dtype="float32")
	return pts

def get_transform(image, pts):
	imgRef = order_points(pts)
	maxWidth = 400
	maxHeight = 600
	
	marginTop = 0
	marginLeft = 0

	fieldRef = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	trans = cv2.getPerspectiveTransform(imgRef, fieldRef)
	print(M) #test
	warped = cv2.warpPerspective(image, trans, (800, 1200))
	return warped