import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt

train_path="data/train"
test_path="data/test"

def class_show(train_path):
    concate_list=[]
    
    list_name=['/bee','/couch','/girl','/lawn_mower','/whale','/wolf']
    list_num=[6, 25, 35, 41, 95, 97]

    for x,name in enumerate(list_name):
        img_list=[]
        img_first=name+"/0.png"
        img_concate=cv2.imread(train_path+img_first)
        for i in range(10):
            img_name=name+"/"+str(i)+".png"
            img=cv2.imread(train_path+img_name)
            img_list.append(img)
            if i>0:
                img_concate=np.concatenate((img_concate,img_list[i]),axis=1)
        concate_list.append(img_concate)
        ax3 =plt.subplot(10,1,x+1)
        ax3.set_yticks([])
        ax3.set_xticks([])
        ax3.set_ylabel(name[1:], rotation=0, labelpad=32)
        plt.imshow(concate_list[x])
        plt.axis('on')
    
    plt.show()

class_show(train_path)