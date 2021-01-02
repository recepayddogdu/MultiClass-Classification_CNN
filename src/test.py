from keras.preprocessing import image
from keras import models 
import matplotlib.pyplot as plt
import numpy as np
import cv2 

list_name=['/bee','/couch','/girl','/lawn_mower','/whale','/wolf']
model=models.load_model('src/models/best_model2.h5')
test_path='data/test'


for x,name in enumerate(list_name):
    predict_list=[]
    random=np.random.randint(1,100)
    path=(test_path+name+'/'+str(random)+".png")
    Data1=image.load_img(path, target_size=(32,32))

    Data=image.img_to_array(Data1)
    
    y=model.predict(Data.reshape(1,32,32,3))
    
    predict_ind=np.argmax(y)
    
    ax3 = plt.subplot(6,1,x+1)
    ax3.set_yticks([])
    ax3.set_xticks([])

    ax3.set_ylabel('Olmasi Gereken:{0}\nTahmin:{1}'.format(list_name[x][1:],
                                                    list_name[predict_ind][1:]),
                                                    rotation=0,
                                                    labelpad=75)
    plt.imshow(Data1)

plt.show()
    
    

