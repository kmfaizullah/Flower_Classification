from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pathlib
import random
import keras
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from os import listdir
from os.path import join
import cv2
import pandas
import os
import datetime
from keras import utils as np_utils
from keras.preprocessing.image import ImageDataGenerator




class ShowData:
    def __init__(self, train_images,train_label,test_images,test_lebel,index_to_label):
        self.train_images = train_images
        self.train_label= train_label
        self.test_images= test_images
        self.test_lebel=test_lebel
        self.index_to_label=index_to_label
        
    def checkTrainData(self):
        plt.figure(figsize=(20,20))
        for i in range(5):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i], cmap=plt.cm.binary)
            plt.xlabel(self.index_to_label[self.train_label[i]])
        plt.show()
        
    def TestingData(self,predictions,index_to_label):
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.test_images[i], cmap=plt.cm.binary)
            plt.xlabel(index_to_label[np.argmax(predictions[i])])
            plt.ylabel(index_to_label[self.test_lebel[i]])
        plt.show()
