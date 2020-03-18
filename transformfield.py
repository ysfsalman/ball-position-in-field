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
    imageCopy = image.copy()
    for point in ref_pts_in_img:
        cv2.circle(imageCopy, point, 5, (0, 0, 255), -1)
	
    cv2.imshow("Set Reference", imageCopy)

def order_points(pts):
	# order the points to top-left, top-right, 
	# bottom-right, and bottom-left
	# in case user mistakenly insert wrong order
	# status : WIP
	pts = np.array(pts, dtype="float32")
	return pts

# field size example : 12*8
def get_transform(image,fieldSize = (1200,800), offset = 100):
	# return transformation and inverse transformation matrix 
	# with points given by user	
	maxWidth, maxHeight = fieldSize
	offset = 100
	cv2.namedWindow("Set Reference")
	cv2.setMouseCallback("Set Reference", get_ref_pts_in_img)
	
	while True:
		# prompt user to select points
		pin_ref_pts(image)
		if cv2.waitKey(1) & len(ref_pts_in_img) >= 4:
			pin_ref_pts(image)
			break
	
	imgRef = order_points(ref_pts_in_img)		
	fieldRef = np.array([
		[maxWidth + offset, offset],
		[maxWidth + offset, maxHeight/2 + offset],
		[maxWidth/2 + offset, maxHeight/2 + offset],
		[maxWidth/2 + offset, offset]], dtype = "float32")
	
	trans = cv2.getPerspectiveTransform(imgRef, fieldRef)
	invTrans = cv2.getPerspectiveTransform(fieldRef, imgRef)
	warped = cv2.warpPerspective(image, trans, (maxWidth, maxHeight))
	cv2.imshow("warped", warped)
	return trans, invTrans

def get_ROI(image, invTrans, fieldSize = (1200,800), offset = 100):
	# return mask to crop image
    maxWidth, maxHeight = fieldSize    
    fieldCrop = np.array([
        [0, 0, 1],
        [maxWidth + 2*offset, 0, 1],
        [maxWidth + 2*offset, maxHeight + 2*offset, 1],
        [0, maxHeight + 2*offset, 1]], dtype = "float32")
    
    crop = np.zeros(np.shape(fieldCrop),dtype="float32")
    
    for i in range(len(fieldCrop)):		
        crop[i] = invTrans.dot(fieldCrop[i])
        crop[i] = crop[i]/crop[i][-1]		
        # cv2.circle(image, tuple(crop[i][:-1]), 10, (255,0, 0), -1)
    
    crop = (crop[:,[0,1]])
    
    if crop[3,[1]] < crop[1,[1]]:
        # correct error if crop points incorrect
        crop[3] = [crop[2][0]+100, crop[0][1]+500]
        
    #test(image,crop)    
    mask = np.zeros_like(image)
    match_mask_color = (255,) * image.shape[2]
    cv2.fillPoly(mask, np.int32([crop]), match_mask_color)
    return mask

