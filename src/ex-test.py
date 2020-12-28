from keras.preprocessing import image
from keras import models 
import numpy as np

list_name=['/bee','/couch','/girl','/lawn_mower','/whale','/wolf']
model=models.load_model('best_model.h5')
test_dir='data/test'

count=0
for name in list_name:
    random=np.random.randint(1,100)
    Giris1=image.load_img((test_dir+name+'/'+str(random)+".png"),
                          target_size=(32,32))

    #Numpy dizisine dönüştür
    Giris=image.img_to_array(Giris1)
    #Görüntüuü ağa uygula
    y=model.predict(Giris.reshape(1,32,32,3))
    #En yüksek tahmin sınıfını bul
    tahmin_indeks=np.argmax(y)
    tahmin_yuzde=y[0][tahmin_indeks]*100

    if str(name[1:]) == str(list_name[tahmin_indeks][1:]):
        count = count+1

    print("seçilen görüntü=",name[1:])
    print("tahmin sonucu=",tahmin_indeks)
    print('tahmin class=',list_name[tahmin_indeks][1:])
    print("\n")
print("tahmin edilen görüntü sayısı:", str(count)+"/6\n")