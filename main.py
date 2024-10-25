# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 19:09:29 2019

@author: vahid
"""

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

from keras.layers import Activation,Conv2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Input
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
#from data_augmentation import ImageGenerator
import itertools
from sklearn.metrics import classification_report,confusion_matrix


def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm=cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix without normalization")
    print(cm)
    thresh=cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment="center",
        color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('Predicted label')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                            use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                            use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3),
            #kernel_regularizer=regularization,
            padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input, output)
    return model



#%%%%
img_rows=64
img_cols=64
num_channels=1
img_data_list=[]

data_dir_list=[('man'),('woman')]
data_path="/media/vahid/My Passport/vice_presidential/Iranian_gender_recognition/dataset/dataset/dataset/"
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+dataset)
    print('load the dataset'+'{}\n'.format(dataset))
    
    for img in img_list:
        input_img=cv2.imread(data_path+'/'+dataset+'/'+img)
        input_img=cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(img_rows,img_cols))
        img_data_list.append(input_img_resize)
        
img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
#img_data /= 255
print(img_data.shape)

#%%%%
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

#%%%%
#define the number o classes

num_classes=2
num_of_samples=img_data.shape[0]
print(num_of_samples)
labels=np.ones((num_of_samples),dtype='int64')
labels[0:2344]=0
labels[2344:4253]=1


#conver class labels to one _hot encoding
Y=np_utils.to_categorical(labels,num_classes)
batch_size = 32
input_shape = (img_rows, img_cols, 1)
num_epochs = 200
#shuffle data
x,y=shuffle(img_data,Y,random_state=2)
#image_generator = ImageGenerator(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=batch_size,epochs=num_epochs,verbose=1)
#model.save('gender_model.h5')
Y_pred =model.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1)
#y_pred = model.predict_classes(x_test)
target_names = ['(man)','(woman)']
print(classification_report(np.argmax(y_test,axis=1),y_pred,target_names=target_names))
cnf_matrix=(confusion_matrix(np.argmax(y_test,axis=1),y_pred))
add_cnf_matrix=(cnf_matrix)
plot_confusion_matrix(add_cnf_matrix,classes=target_names,title='Confusion matrix')











