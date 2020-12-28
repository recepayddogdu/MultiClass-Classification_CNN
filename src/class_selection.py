from keras.datasets import cifar100
from matplotlib import pyplot
import cv2 
import numpy as np
import os

# load data
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
train_dir='../data/train'
test_dir='../data/test'

def open_dir(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

open_dir(train_dir)
open_dir(test_dir)
    
def subsetdata(X_data,y_data,image_dir):
    if not os.path.exists(image_dir+"/bee"):
        list_name=['/bee','/couch','/girl','/lawn_mower','/whale','/wolf']
        list_num=[6, 25, 35, 41, 95, 97]
        for i,num in enumerate(list_num):
            index=np.where(y_data==num)
            subset_x_data=X_data[np.isin(y_data,[num]).flatten()]
            for a,x in enumerate(subset_x_data):
                image_path=(image_dir+list_name[i])
                open_dir(image_path)
                image_path=(image_path+"/"+str(a)+".png")
                cv2.imwrite(image_path,x)
    else:
        print(image_dir+" konumunda datalar zaten hazir. Kontrol edin.")

subsetdata(X_train,y_train,train_dir)
subsetdata(X_test,y_test,test_dir)