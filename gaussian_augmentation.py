import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

f=1
count=0
def write_imgs(noisy_image):
    g=1
    for x in my_range(50, 255, 50):
        cv2.normalize(noisy_image, noisy_image, 0, x, cv2.NORM_MINMAX, dtype=-1)
        noisy_image= noisy_image.astype(np.uint8)
        cv2.imwrite(path+'\\'+'aug{}{}.jpg'.format(f,g), noisy_image)
        g+=1
        global count
        count+=1
def gaussian_noise(img,path):
    img = cv2.imread(os.path.join(path,img))
    mean = 0
    var = 1
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, img.shape[:-1] )
    noisy_image = np.zeros(img.shape, np.float32)
    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian
    write_imgs(noisy_image)

DATADIR=os.path.join(os.getcwd(),'Training_Data_Augmented')
noisy=[]
CATEGORIES = os.listdir(DATADIR)
for category in CATEGORIES:
        path=os.path.join(DATADIR,category)
        print(os.listdir(path))
        for imgs in os.listdir(path):
            if imgs.find("DWI")!=-1:
                gaussian_noise(imgs,path)
                f+=1

print(count)
                
