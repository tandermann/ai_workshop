#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:25:51 2020

@author: Tobias Andermann (tobiasandermann88@gmail.com)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os, glob
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def make_convnet(instance,n_classes):
    input_shape = instance.shape
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(n_classes, activation='sigmoid'))
    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model



data_gen = ImageDataGenerator(rescale=1.0/255)

imgdir = '/Users/xhofmt/GitHub/dna_sequence_nn/data/images'
img_size = 16
batch_size = 5
classes = [os.path.basename(i) for i in glob.glob(os.path.join(imgdir,'train/*'))]
n_classes = len(classes)


train_generator = data_gen.flow_from_directory(
        imgdir + '/train',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=True)


Xbatch, Ybatch = train_generator.next()

np.shape(Xbatch)

plt.imshow(Xbatch[0])

    
my_cnn = make_convnet(Xbatch[0],n_classes)


validation_generator = data_gen.flow_from_directory(
        imgdir + '/validation',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=False)



history = my_cnn.fit_generator(train_generator, epochs=100,validation_data=validation_generator,steps_per_epoch=len(train_generator),validation_steps=len(validation_generator),verbose=2)
my_cnn.save_weights('/Users/xhofmt/GitHub/dna_sequence_nn/trained_models/my_cnn_weights')
#my_cnn.load_weights('/Users/xhofmt/GitHub/dna_sequence_nn/trained_models/my_cnn_weights')

plt.plot(history.history['loss']);
plt.plot(history.history['val_loss']);
plt.legend(['training loss', 'validation loss']);


plt.plot(history.history['accuracy']);
plt.plot(history.history['val_accuracy']);
plt.legend(['training accuracy','validation accuracy']);




prediction = np.argmax(my_cnn.predict(validation_generator),axis=1)
true_labels = validation_generator.classes

sum(prediction==true_labels)/len(true_labels)


