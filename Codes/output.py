# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 20:26:51 2019

@author: Areeb
"""
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
import glob
import os
from sklearn.model_selection import train_test_split
import shutil
import tensorflow as tf
import numpy as np
from keras.models import load_model



coarse = load_model('Inception_coarse_final')
aircraft = load_model('MobileNet_only_aircrafts_final')
dog = load_model('MobileNet_only_dogs_final')
bird = load_model('MobileNet_only_birds_final')
car = load_model('MobileNet_only_cars_final')
flower = load_model('MobileNet_only_flowers_final')


path='/content/drive/My Drive/CS783_A2/Assignment2 Test Dataset/'
length = len(path)
model={'0':aircraft,'1':bird,'2':car,'3':dog,'4':flower}
labels=["aircraft","bird","car","dog","flower"]
output=[]
obj = glob.glob(path+'*.jpg')
imagedict = [file for file in os.listdir(path) if os.path.splitext(file)[-1] == '.jpg']

for i in obj:
    output1=""
    output1=output1+i[length:]+" "
    img = image.load_img(i, target_size=(299,299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = coarse.predict(x)
    k=preds[0].tolist().index(max(preds[0]))
    #print(k)
    #find highest preds index
    output1=output1+labels[k]+" "+labels[k]+"@"
    img = image.load_img(i, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    fine_preds=model[str(k)].predict(x)
    #find highest preds index
    k=fine_preds[0].tolist().index(max(fine_preds[0]))
    output1=output1+str(k+1)
    output.append(output1)
    
with open('output.txt', "w") as file:
    for name in output:
        file.write(name+'\n')