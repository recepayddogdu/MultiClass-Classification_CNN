from keras.preprocessing import image
from keras import models 
import matplotlib.pyplot as plt
import numpy as np
import cv2 

list_name=['/bee','/couch','/girl','/lawn_mower','/whale','/wolf']
model=models.load_model('src/models/best_model2.h5')
test_dir='data/test'


for x,name in enumerate(list_name):
    tahmin_list=[]
    random=np.random.randint(1,100)
    path=(test_dir+name+'/'+str(random)+".png")
    Giris1=image.load_img(path,
                          target_size=(32,32))

    #Numpy dizisine dönüştür
    Giris=image.img_to_array(Giris1)
    #Görüntüuü ağa uygula
    y=model.predict(Giris.reshape(1,32,32,3))
    #En yüksek tahmin sınıfını bul
    tahmin_indeks=np.argmax(y)
    tahmin_yuzde=y[0][tahmin_indeks]*100
    
    
    ax3 =plt.subplot(6,6,x+1)
    ax3.set_yticks([])
    ax3.set_xticks([])

    ax3.set_xlabel('Label:{0}\nTahmin:{1}'.format(list_name[x][1:],list_name[tahmin_indeks][1:]))
    plt.imshow(Giris1)

plt.show()
    
    

