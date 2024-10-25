""" A class for testing a SSD model on a video file or webcam """

import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image 
import pickle
import numpy as np
from random import shuffle
from scipy.misc import imread, imresize
from timeit import default_timer as timer
import time
import sys
sys.path.append("..")
from ssd_utils import BBoxUtility
import time
import csv 
import logging
#from keras.models import load_model
import sys
sys.path.append("..")
from ssd import SSD300 as SSD

#***********************************************************jadid

from keras import backend as t
t.set_image_data_format('channels_last')
from keras.models import load_model

import numpy as np
import os
from keras.layers import BatchNormalization
from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution3D,MaxPooling3D


#**********************************************************
input_shape = (300,300,3)

# Change this if you run with other classes than VOC
class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
NUM_CLASSES = len(class_names)

model = SSD(input_shape, num_classes=NUM_CLASSES)

# Change this path if you want to use your own trained weights
model.load_weights('weights_SSD300.hdf5') 
fname= "gender_model.h5"
test_model = load_model(fname)

img_rows=64
img_cols=64
num_channels=1
img_data_list=[]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image_path = "G:/vice_presidential/Iranian_gender_recognition/test/testm4.jpg"
#logging.basicConfig(filename='date', level = logging.DEBUG, format='%(asctime)s: %(levelname)s: %(message)s')
class_names = class_names
num_classes = len(class_names)
model = model
input_shape = input_shape
bbox_util = BBoxUtility(num_classes)
opt=[]
# Create unique and somewhat visually distinguishable bright
# colors for the different classes.
class_colors = []
start_frame = 0
conf_thresh = 0.6
for i in range(0, num_classes):
    # This can probably be written in a more elegant manner
    hue = 255*i/num_classes
    col = np.zeros((1,1,3)).astype("uint8")
    col[0][0][0] = hue
    col[0][0][1] = 128 # Saturation
    col[0][0][2] = 255 # Value
    cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
    col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
    class_colors.append(col) 
    

        
orig_image = cv2.imread(image_path)
orig_image = np.array(orig_image)
im_size = (input_shape[0], input_shape[1])    
resized = cv2.resize(orig_image, im_size)
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
to_draw = cv2.resize(resized, (int(input_shape[0]), input_shape[1]))
# Use model to predict 
inputs = [image.img_to_array(rgb)]
tmp_inp = np.array(inputs)
x = preprocess_input(tmp_inp)
y = model.predict(x)
results = bbox_util.detection_out(y)
if len(results) > 0 and len(results[0]) > 0:
    # Interpret output, only one frame is used 
                
    det_label = results[0][:, 0]
    det_conf = results[0][:, 1]
    det_xmin = results[0][:, 2]
    det_ymin = results[0][:, 3]
    det_xmax = results[0][:, 4]
    det_ymax = results[0][:, 5]

    top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * to_draw.shape[1]))
        ymin = int(round(top_ymin[i] * to_draw.shape[0]))
        xmax = int(round(top_xmax[i] * to_draw.shape[1]))
        ymax = int(round(top_ymax[i] * to_draw.shape[0]))

        # Draw the box on top of the to_draw image
        class_num = int(top_label_indices[i])
        if not class_names[class_num]=='person':
            print('Please upload a picture of your face')
            break                
        elif class_names[class_num]=='person':
            input_img_gray=cv2.cvtColor(orig_image,cv2.COLOR_BGR2GRAY)
            input_img_resize=cv2.resize(input_img_gray,(img_rows,img_cols))
            img_data_list.append(input_img_resize)
                    
            img_data = np.array(img_data_list)
            #img_data = img_data.astype('float32')
            #img_data /= 255
            print(img_data.shape)
            faces = face_cascade.detectMultiScale(orig_image, 1.3, 2)
            #if not faces():
            #    print('Please upload a picture of your face')
            #    pass
            
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
                
            gender_prediction = (test_model.predict(img_data))
            gender_label_arg = np.argmax(gender_prediction)
            print(gender_label_arg)
                
            if gender_label_arg == 1:
                color = (0, 0, 255)
                text= 'woman'
            else:
                color = (255, 0, 0)
                text = 'man'   
                
            for (x,y,w,h) in faces:
                cv2.rectangle(orig_image,(x,y),(x+w,y+h),color,2)
                text_pos = (x + 10, y - 15)
                cv2.putText(orig_image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                #roi_gray = in[y:y+h, x:x+w]
                if text=='woman':
                    print('Please upload a real your face image')
                cv2.imshow('img',orig_image)
                cv2.waitKey(30) & 0xff
                
                        
            
        
