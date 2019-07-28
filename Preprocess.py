import pathlib
import pandas as pd
import numpy as np
from os import listdir
from os.path import join
import os
import random
import cv2

class Process:
    def __init__(self, data_directory,image_height,image_width):
        self.directory = data_directory
        self.height= image_height
        self.width= image_width
        
    def ImageProcess(self):
        all_image = []
        size = self.height,self.width
        
        data_root = pathlib.Path(self.directory)
        folders = os.listdir(data_root)
        
        all_image_paths = list(data_root.glob('*/*'))
        all_image_paths = [str(path) for path in all_image_paths]
        random.shuffle(all_image_paths)
        
        for path in all_image_paths:
            img = cv2.imread(path)
            im = cv2.resize(img,size)
            im =im/ 255.0
            all_image.append(im)
        
        all_image=np.asarray(all_image)
            
        label_to_index = dict((name, index) for index,name in enumerate(folders))
        index_to_label= dict((index,name) for index,name in enumerate(folders))
        all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
        all_image_labels=np.asarray(all_image_labels)

        return all_image,all_image_labels,index_to_label


