import numpy as np
import cv2

ref_pts_in_img = []

def get_ref_pts_in_img(event,x,y, flags, params):
    # get reference point from image
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pts_in_img.append((x,y))
        #print(ref_pts_in_img)

def pin_ref_pts(image):
    # user interface for field setup
    for point in ref_pts_in_img:
        cv2.circle(image, point, 5, (0, 0, 255), -1)
    cv2.imshow("Set Reference", image)

def order_points(pts):
	# order the points to top-left, top-right, 
	# bottom-right, and bottom-left
	# in case user mistakenly insert wrong order
	# status : WIP
	pts = np.array(pts, dtype="float32")
	return pts

# field size example : 12*8
def get_transform(image):
	# return transformation and inverse transformation matrix 
	# with points given by user	
	cv2.namedWindow("Set Reference")
	cv2.setMouseCallback("Set Reference", get_ref_pts_in_img)
	
	while True:
		# prompt user to select points
		pin_ref_pts(image)
		if cv2.waitKey(1) & len(ref_pts_in_img) >= 4:
			pin_ref_pts(image)
			break
	
	imgRef = order_points(ref_pts_in_img)
	maxWidth = 1200
	maxHeight = 800
	offset = 100
	
	fieldRef = np.array([
		[maxWidth + offset, offset],
		[maxWidth + offset, maxHeight/2 + offset],
		[maxWidth/2 + offset, maxHeight/2 + offset],
		[maxWidth/2 + offset, offset]], dtype = "float32")
	

	'''test3'''
	fieldCrop = np.array([
		[0, 0, 1],
		[maxWidth + 2*offset, 0, 1],
		[maxWidth + 2*offset, maxHeight + 2*offset, 1],
		[0, maxHeight + 2*offset, 1]], dtype = "float32")

	'''' test'''
	invTrans = cv2.getPerspectiveTransform(fieldRef, imgRef)
	#field_box = np.append(fieldRef,np.ones((len(fieldRef),1)), axis=1)
	crop = np.zeros(np.shape(fieldCrop),dtype="float32")
	for i in range(len(fieldCrop)):		
		crop[i] = invTrans.dot(fieldCrop[i])
		crop[i] = crop[i]/crop[i][-1]		
		cv2.circle(image, tuple(crop[i][:-1]), 10, (255,0, 0), -1)
	cropPts = (crop[:,[0,-2]])
	
	
	
	'''test2'''
	mask = np.zeros_like(image)
	match_mask_color = (255,) * image.shape[2]
	cv2.fillPoly(mask, np.int32([cropPts]), match_mask_color)
	image = cv2.bitwise_and(image, ~mask)
	cv2.imshow("Set Reference", image)
	
	trans = cv2.getPerspectiveTransform(imgRef, fieldRef)
	return trans
