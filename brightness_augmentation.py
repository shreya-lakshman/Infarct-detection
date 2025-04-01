from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
import numpy as np
import cv2
IMG_SIZE=100

CATEGORIES=[]
DATADIR=os.path.join(os.getcwd(),'Training_Data_Augmented')
for files in os.listdir(DATADIR):
    CATEGORIES.append(files)
for category in CATEGORIES:
    path=os.path.join(DATADIR,category)
    for img in os.listdir(path):
        datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
        image_path =os.path.join(path,img)
        image = np.expand_dims(cv2.imread(image_path), 0)
        save_here = path
        datagen.fit(image)
        for x, val in zip(datagen.flow(image,                   
                save_to_dir=save_here,     
                save_prefix= 'aug',
                save_format='jpg'),range(1)) :
                pass    

