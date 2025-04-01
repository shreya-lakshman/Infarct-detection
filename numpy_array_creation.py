import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
import pickle
from keras.utils import to_categorical

IMG_SIZE=100
CATEGORIES_train=[]
CATEGORIES_test=[]
training_data=[]

DATADIR_train=os.path.join(os.getcwd(),'Training_Data_Augmented')
DATADIR_test=os.path.join(os.getcwd(),'Test_Data')
for folder in os.listdir(DATADIR_train):
    CATEGORIES_train.append(folder)
for folder in os.listdir(DATADIR_test):
    CATEGORIES_test.append(folder)


def create_training_data_arrays(DATADIR,CATEGORIES):
    for category in CATEGORIES:
        path=os.path.join(DATADIR,category)
        class_num= CATEGORIES.index(category)

        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data_arrays(DATADIR_train,CATEGORIES_train)
x_train=[]
y_train=[]
for features, label in training_data:
    x_train.append(features)
    y_train.append(label)
X_train= np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y_train=np.array(y_train)



training_data=[]
create_training_data_arrays(DATADIR_test,CATEGORIES_test)
x_test=[]
y_test=[]
for features, label in training_data:
    x_test.append(features)
    y_test.append(label)
X_test= np.array(x_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y_test=np.array(y_test)


pickle_out=open("X_train.pickle","wb")
pickle.dump(X_train,pickle_out)
pickle_out.close()

pickle_out=open("Y_train.pickle","wb")
pickle.dump(Y_train,pickle_out)
pickle_out.close()

pickle_out=open("X_test.pickle","wb")
pickle.dump(X_test,pickle_out)
pickle_out.close()

pickle_out=open("Y_test.pickle","wb")
pickle.dump(Y_test,pickle_out)
pickle_out.close()

