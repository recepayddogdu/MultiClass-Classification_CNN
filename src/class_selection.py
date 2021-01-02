from keras.datasets import cifar100
from matplotlib import pyplot
import cv2 
import numpy as np
import os

# load data
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
train_path='data/train'
test_path='data/test'

def open_path(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

open_path(train_path)
open_path(test_path)
    
def class_selection(X_data,y_data,image_dir):
    if not os.path.exists(image_dir+"/bee"):
        class_name=['/bee','/couch','/girl','/lawn_mower','/whale','/wolf']
        class_num=[6, 25, 35, 41, 95, 97]
        for i,num in enumerate(class_num):
            index=np.where(y_data==num) #class_num'daki dataların indexlerini belirlemek
            subset_x_data=X_data[np.isin(y_data,[num]).flatten()]
            for a,x in enumerate(subset_x_data):
                image_path=(image_dir+class_name[i])
                open_path(image_path)
                image_path=(image_path+"/"+str(a)+".png")
                cv2.imwrite(image_path,x)
    else:
        print(image_dir+" konumunda datalar zaten hazir. Kontrol edin.")

class_selection(X_train,y_train,train_path)
class_selection(X_test,y_test,test_path)
