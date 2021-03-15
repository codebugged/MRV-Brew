from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2
import pandas as pd
import numpy as np
import dlib
from math import hypot
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
vs = VideoStream(src=0).start()
class_labels = ['Angry','Digust','Fear','Happy','Sad', 'Surprise','Neutral']
model=tf.keras.models.load_model('model_filter.h5')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def midpoint(p1 ,p2):
	return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
def get_blinking_ratio(eye_points, facial_landmarks):
	left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
	right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
	center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
	center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
	hor_line = cv2.line(frame, left_point, right_point, (0,0, 255), 2)
	ver_line = cv2.line(frame, center_top, center_bottom, (0, 0, 255), 2)
	hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
	ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
	ratio = hor_line_lenght / ver_line_lenght
	return ratio
while True:
	frame = vs.read()
	rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	boxes = face_recognition.face_locations(gray,model="hog")
	r = frame.shape[1] / float(rgb.shape[1])
	faces = detector(gray)
	for face in faces:
		landmarks = predictor(gray, face)
		left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
		right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
		blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
		font = cv2.FONT_HERSHEY_DUPLEX
		if blinking_ratio > 5.7:
			cv2.putText(frame, "DROWSINESS DETECTED", (10, 50), font, 1, (0, 0, 255))
	cv2.imshow("Accident detector", frame)
	for ((top, right, bottom, left)) in boxes:
		crop_img = rgb[top:bottom, left:right]
		crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
		roi_gray = cv2.resize(crop_img,(48,48),interpolation=cv2.INTER_AREA)
		if np.sum([roi_gray])!=0:
			roi = roi_gray.astype('float')/255.0
			roi = img_to_array(roi)
			roi = np.expand_dims(roi,axis=0)
		pred = model.predict(roi)[0]
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)
		cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		label=class_labels[pred.argmax()]
		label_position = (top,left)
		cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
		cv2.imshow("Frame", frame)
		cv2.imshow("Frame2", crop_img)
		print(pred)
	if (cv2.waitKey(1) == ord('q')):
		break
cv2.destroyAllWindows()
vs.stop()