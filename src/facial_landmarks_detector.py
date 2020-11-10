import os
os.chdir(os.getcwd() + '/src')

import cv2
import numpy as np
from PIL import Image
import dlib
import imutils
from imutils import face_utils

COLORS = [(19, 199, 109), (79, 76, 240), (230, 159, 23), (168, 100, 168), (158, 163, 32), (163, 38, 32), (180, 42, 220)]*2

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

os.chdir(os.path.dirname(os.getcwd()))

def detect_facial_landmarks(image):
    # copy original image and convert another copy to grayscale
    output = np.array(image)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # determine the facial landmarks for the face image
    rect = dlib.rectangle(0, 0, *gray.shape[::-1])
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    
    # visualize all facial landmarks with a transparent overlay
    output = face_utils.visualize_facial_landmarks(output, shape, colors=COLORS)
    
    # loop over the face parts individually
    output_info = dict({})
    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        # loop over the subset of facial landmarks, drawing the specific face part
        pts = list([])
        for (x, y) in shape[i:j]:
            pts.append((x, y))
            cv2.circle(output, (x, y), 1, (0, 0, 255), -1)
        output_info[name] = pts
    
    output = Image.fromarray(output)
    return output_info, output

def get_face_above_jaw(image, jawline_pts, draw_roi=False):
    output = np.array(image)
    output_roi = np.array(image)
    
    jawline_pts = sorted(jawline_pts, key=lambda k: k[0])
    top_left_corner = (jawline_pts[0][0], 0)
    top_right_corner = (jawline_pts[-1][0], 0)
    
    face_polygon_pts = list([top_left_corner, *jawline_pts, top_right_corner])
    face_polygon = cv2.convexHull(np.array(face_polygon_pts, dtype=np.float32))
    
    # prepare mask image
    mask_img = np.zeros((image.height, image.width), dtype=np.uint8)
    for i in range(image.height):
        for j in range(image.width):
            dist_coeff = cv2.pointPolygonTest(face_polygon, (j, i), True)
            if dist_coeff >= 0:
                mask_img[i][j] = 255
                
    # draw face ROI
    if draw_roi:
    	for i in range(len(face_polygon_pts)):
    	    start = face_polygon_pts[i]
    	    end = face_polygon_pts[(i+1)%len(face_polygon_pts)]
    	    cv2.line(output_roi, start, end, (0,255,0), 2)
    	
                
    output = cv2.bitwise_and(output, output, mask = mask_img)
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
    
    output_roi = Image.fromarray(output_roi)
    mask_img = Image.fromarray(mask_img)
    output = Image.fromarray(output)
    
    return output_roi, mask_img, output
