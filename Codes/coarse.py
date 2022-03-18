# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:49:06 2019

@author: Areeb
"""

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2
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



##########################################################################    
#########################-->1.Dataset Preprocessing<--####################
######(creates Train and Validation Sets for Coarse Classification)#######
##########################################################################

os.mkdir('train_fine')
os.mkdir('test_fine')
lengths={"aircrafts":7,"birds_":11,"cars":8,"dogs_":5,"flowers_":5}
for i in ["aircrafts","birds_","cars","dogs_","flowers_"]:
    os.mkdir('train/'+i)
    os.mkdir('test/'+i)
    for j in range(1,lengths[i]+1):
        obj = glob.glob('dataset/'+i+'/'+str(j)+'/*.jpg')
        obj_train, obj_test = train_test_split(obj, test_size=0.20)
        for file in obj_train:
            shutil.copy(file,'train/'+i)
        for file in obj_test:
            shutil.copy(file,'test/'+i)

##########################################################################    
#########################-->2.Dataset Preprocessing<--####################
#######(creates Train and Validation Sets for Fine Classification)########
##########################################################################

for i in ["aircrafts","birds_","cars","dogs_","flowers_"]:
    os.mkdir('train_'+i)
    os.mkdir('test_'+i)
    for j in range(1,lengths[i]+1):
        os.mkdir('train_'+i+'/'+str(j))
        os.mkdir('test_'+i+'/'+str(j))
        obj = glob.glob('dataset/'+i+'/'+str(j)+'/*.jpg')
        obj_train, obj_test = train_test_split(obj, test_size=0.20)
        for file in obj_train:
            shutil.copy(file,'train_'+i+'/'+str(j))
        for file in obj_test:
            shutil.copy(file,'test_'+i+'/'+str(j))

##########################################################################    
#####################-->3.Initializing InceptionV3<--#####################
##########################################################################
        

base_model = InceptionV3(weights='imagenet', include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 5 classes
predictions = Dense(5, activation='softmax')(x)
# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
#compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

##########################################################################    
#########################-->3.Data Augmentation<--########################
############(helps improve training examples for small datasets)##########
##########################################################################

from keras.preprocessing.image import ImageDataGenerator
WIDTH = 299
HEIGHT = 299
BATCH_SIZE = 32
# data prep
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    "train_"+i,
    target_size=(HEIGHT, WIDTH),
		batch_size=BATCH_SIZE,
		class_mode='categorical')
    
validation_generator = validation_datagen.flow_from_directory(
    "test_"+i,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

##########################################################################    
#######################-->4.Compiling InceptionV3<--######################
##########################################################################

# train the model on the new data for a few epochs
EPOCHS = 5
BATCH_SIZE = 32
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 64

MODEL_FILE = 'Inception_coarse_final'

history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS)

#Saving the model
model.save(MODEL_FILE)

##########################################################################    
##################-->5.Validating on Validation Set<--####################
##########################################################################

k=0
for i in ["aircrafts","birds_","cars","dogs_","flowers_"]:
    count=0
    print(i)
    obj = glob.glob('test/'+str(i)+'/*.jpg')
    for j in obj:
        img = image.load_img(j, target_size=(HEIGHT, WIDTH))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        if(preds[0].tolist().index(max(preds[0]))==k):
            count+=1
    print(count, len(obj), count/len(obj)*100)
    k=k+1