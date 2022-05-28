import sys
import os
import shutil
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shutil import copyfile
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

commestible_dictionnary = {
    "Armillaria lutea" : False,
    "Coprinellus micaceus" : False,
    "Fomes formentarius" : False,
    "Fomitopsis pinicola" : False,
    "Ganoderma pferfferi" : True,
    "Mycena galericulata" : True,
    "Plicatura crispa" : False,
    "Pluteus cervinus" : True,
    "Tricholoma scalpturatum" : True,
    "Xerocomellus chrysenteron" : True,
}

chemin_photo = sys.argv[1]

#Prediction
model = tf.keras.models.load_model('cnn.model')

test = image.load_img(chemin_photo, target_size = (64, 64))
test = image.img_to_array(test)
test = np.expand_dims(test, axis = 0)
result = model.predict(test)
test.class_indices

if result[0][0] == 0:
    prediction = 0
elif result[0][0] == 1:
    prediction = 1
elif result[0][0] == 2:
    prediction = 2
elif result[0][0] == 3:
    prediction = 3
elif result[0][0] == 4:
    prediction = 4
elif result[0][0] == 5:
    prediction = 5
elif result[0][0] == 6:
    prediction = 6
elif result[0][0] == 7:
    prediction = 7
elif result[0][0] == 8:
    prediction = 8
else:
    prediction = 9

liste_espece= list(commestible_dictionnary)
espece = liste_espece[prediction]
print("Espece : " + espece)

commestible = commestible_dictionnary[espece]

if commestible == True:
    print("Comestible : Oui")
else :
    print("Comestible : NON!")

