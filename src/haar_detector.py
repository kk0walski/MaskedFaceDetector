import os
os.chdir(os.getcwd() + '/src')

import cv2
import numpy as np
from PIL import Image

SCALE = 1.3
NEIGHBORS = 3

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('models/Mouth.xml')
nose_cascade = cv2.CascadeClassifier('models/Nariz.xml')

os.chdir(os.path.dirname(os.getcwd()))

def detect_face_parts(image, scale=SCALE, neighbors=NEIGHBORS):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    output_info = dict({'eyes': list([]), 'mouths': list([]), 'noses': list([])})
    output = np.array(image)
        
    eyes = eye_cascade.detectMultiScale(gray)
    for ex, ey, ew, eh in eyes:
    	output_info['eyes'].append(((ex, ey), (ex+ew, ey+eh)))
    	output = cv2.rectangle(output, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
        
    mouths = mouth_cascade.detectMultiScale(gray)
    for mx, my, mw, mh in mouths:
    	output_info['mouths'].append(((mx, my), (mx+mw, my+mh)))
    	output = cv2.rectangle(output, (mx, my), (mx+mw, my+mh), (0,0,255), 2)
    
    noses = nose_cascade.detectMultiScale(gray)
    for nx, ny, nw, nh in noses:
    	output_info['noses'].append(((nx, ny), (nx+nw, ny+nh)))
    	output = cv2.rectangle(output, (nx, ny), (nx+nw, ny+nh), (255,255,0), 2)
    
    output = Image.fromarray(output)
    return output_info, output
