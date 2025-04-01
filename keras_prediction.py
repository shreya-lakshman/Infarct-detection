import cv2
import tensorflow as tf
import numpy as np
import os


def prepare(path):
    IMG_SIZE=100
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        new_array= cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

model= tf.keras.models.load_model("savedmodel")
CATEGORIES_label=[]
CATEGORIES_img=[]

DATADIR_label=os.path.join(os.getcwd(),'Training_Data_Augmented')
# DATADIR_img="C:\\Users\\Asus\\OneDrive\\Desktop\\DL_final_model\\Inference_data"

for files in os.listdir(DATADIR_label):
    CATEGORIES_label.append(files)

# for files in os.listdir(DATADIR_img):
#     CATEGORIES_img.append(files)
# print(CATEGORIES_label)
# print(CATEGORIES_img)

n=0
# category=['Bilateral cerebellar hemispheres', 'Bilateral frontal lobes', 'Bilateral occipital lobes', 'Brainstem', 'Lacunar infarct in dorsal aspect of pons', 'Lacunar infarct in left parietal lobe', 'Lacunar infarct in medulla oblongata on the left', 'Lacunar infarct in pons on the left', 'Lacunar infarct in posterior limb of left internal capsule', 'Lacunar infarct in right corona radiata', 'Lacunar infarct in right putamen', 'Lacunar infarcts in bilateral occipital lobes', 'Lacunar infarcts in left corona radiata', 'Lacunar infarcts in the right parietal lobe', 'Left centrum semi ovale and right parietal lobe', 'Left cerebellar hemisphere', 'Left cerebellar lacunar infarcts', 'Left frontal lobe', 'Left fronto-parietal lobe', 'Left insula', 'Left occipital and temporal lobes', 'Left parietal lobe', 'Left thalamic lacunar infarct', 'Medial part of right frontal and parietal lobes', 'Mid brain on right side', 'Pontine infarct on the right', 'Right anterior thalamic infarct', 'Right cerebellar hemisphere', 'Right corona radiata', 'Right fronto-parietal lobe', 'Right ganglio-capsular region', 'Right insula', 'Right lentiform nucleus', 'Right occipital lobe', 'Right parietal lacunar infarct', 'Right temporal lobe', 'Right thalamus', 'Splenium of the corpus callosum']
for category in CATEGORIES_label:
    #path=os.path.join(DATADIR_label,category)
    path=os.path.join(DATADIR_label,category)
    #print(path)
    prediction=model.predict(prepare(path))
    max_pos=np.argmax(prediction,axis=1)
    if(CATEGORIES_label[int(max_pos)]==category):
        print(CATEGORIES_label[int(max_pos)],"-", category)
        n=n+1
        print(n)


print("Training accuracy:",(n/38))


