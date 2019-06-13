# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 07:47:18 2019

@author: acer
"""
import os, cv2, random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img,array_to_img
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import load_model
import matplotlib.image as mpimg
import h5py

def prepare_list_and_dict(img_path):
    dict_to_save = {}
    img_list = []
    path = img_path
    files = os.listdir(path)
    
    for file in files:
        filepath = path+ "/" +file
        filenames = os.listdir(filepath)
        #print("filenames:",filenames)
        for filename in filenames:
            img_name = path + file + "/" +filename   # dict: {'1/a.jpg': 1}
            img_label = int(file)
            img_list.append(img_name)
            #print("filename:", img_name)
            #print("label:", img_label)
            dict_to_save[img_name] = img_label
    #print("dict:", dict_to_save)
    return img_list, dict_to_save

def prepare_data(img_list, dict_names):
    x = [] # images as arrays
    y = [] # labels
    img_width = 150
    img_height = 150    
    for image in img_list:
        '''cv2.imread(image)是用cv2讀取圖片, image是路徑, 返回一串data, 以list形式儲存在x中'''
        try:
            x.append(cv2.resize(cv2.imread(image), (img_width, img_height), interpolation=cv2.INTER_CUBIC))   
            label = int(dict_names[image]) 
            y.append(label)
        except:
            pass       
    return x, y   
 
path = 'signs_copy/'
img_list, dict_to_save = prepare_list_and_dict(path)
#print("img_list:", img_list)
#print("dict:", dict_to_save)
x, y = prepare_data(img_list, dict_to_save)
#print("y:", y)

x_mid, x_val, y_mid, y_val = train_test_split(x,y, test_size=0.2)#, random_state=1)
x_train,x_test,y_train,y_test = train_test_split(x_mid,y_mid,test_size=0.25)#,random_state=1)
y_train = np_utils.to_categorical(y_train,num_classes = 121)
y_val = np_utils.to_categorical(y_val,num_classes = 121)
y_test = np_utils.to_categorical(y_test,num_classes = 121)
#print('train samples:',len(x_train))
#print('val samples:',len(x_val))
#print('test samples:',len(x_test))

nb_train_samples = len(x_train)
nb_validation_samples = len(x_val)
batch_size = 5

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(1024))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(121))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow(np.array(x_train), y_train, batch_size=batch_size)
validation_generator = val_datagen.flow(np.array(x_val), y_val, batch_size=batch_size)


visualize_acc_loss = model.fit_generator(
    train_generator, 
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)

'''model.fit_generator會記錄record, 可通過.history / .epoch來查詢
類似地可用histroy['val_loss'], histroy['loss'], histroy['val_acc'], histroy['acc'])
'''
print("history:", (visualize_acc_loss.history['acc']))    #history['acc']記錄了10次結果
print("epoch:", visualize_acc_loss.epoch)    #epoch是從0-9共10次
print("accuracy: {:.2f}%".format(visualize_acc_loss.history['acc'][-1]*100))  
print("epoch:", visualize_acc_loss.epoch[-1]+1)    

model.save('sign2.h5')
