# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 12:46:45 2019

@author: vahid
"""
from keras.models import Sequential, Model
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
import keras.callbacks
from sklearn.model_selection import train_test_split
from  sklearn import model_selection
from sklearn.utils import shuffle
import pandas as pd
import tensorflow as tf
from keras import optimizers
from sklearn import svm
from keras.layers import BatchNormalization
from keras.models import load_model
from keras import backend as t
t.set_image_data_format('channels_last')

import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt

fname= "gender_model.h5"
test_model = load_model(fname)

#%%%%
img_rows=64
img_cols=64
num_channels=1
img_data_list=[]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
data_path="test1.jpg"

input_img=cv2.imread(data_path)
input_img_gray=cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
input_img_resize=cv2.resize(input_img_gray,(img_rows,img_cols))
img_data_list.append(input_img_resize)
        
img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
#img_data /= 255
print(img_data.shape)

#image dimension ordering
if num_channels==1:
    if t.image_data_format()=='channels_first':
        img_data=np.expand_dims(img_data,axis=1)
        print(img_data.shape)
    else:
        img_data=np.expand_dims(img_data,axis=3)    
else:
    if t.image_data_format()=='channels_first':
        img_data=np.rollaxis(img_data,3,1)
        print(img_data.shape)
    else:
        img_data=np.rollaxis(img_data,3,3)
                
print(img_data.shape)
#test_video=Normalize(test_video)
        
'''gender_prediction = (test_model.predict(img_data))
gender_label_arg = np.argmax(gender_prediction)
print(gender_label_arg)       
if gender_label_arg == 1:
    color = (0, 0, 255)
    text= 'woman'
else:
    color = (255, 0, 0)
    text = 'man'
        
faces = face_cascade.detectMultiScale(input_img, 1.3, 2)       
for (x,y,w,h) in faces:
    print('*************Thank You for uploading a real image of your face****************')
    cv2.rectangle(input_img,(x,y),(x+w,y+h),color,2)
    text_pos = (x + 10, y - 15)
    cv2.putText(input_img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    #roi_gray = in[y:y+h, x:x+w]
    cv2.imshow('img',input_img)
    
    cv2.waitKey(30) & 0xff'''














