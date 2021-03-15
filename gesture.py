from imutils.video import VideoStream
import imutils
import pickle
import time
import cv2
import pandas as pd
import numpy as np
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
class_labels = ['0','1','2','3','4', '5','Neutral','556']
model=tf.keras.models.load_model('gesture_model.h5')
while True:
	frame = vs.read()
	rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	roi = cv2.resize(frame,(128,128),interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
	pred = model.predict(gray)[0]
	print(pred)
	cv2.imshow('name',frame)
	if (cv2.waitKey(1) == ord('q')):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()