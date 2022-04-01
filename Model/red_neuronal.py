import zipfile
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import numpy as np
import os

pathDataSet = './tmp'
isExist = os.path.exists(pathDataSet)

if not isExist:
    local_zip = './Zips/Dataset.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall(pathDataSet)
    zip_ref.close()
else:
    print('LA CARPETA EXISTE')


#DIMENSIONES DE LAS IMAGENES
width = 100
height = 100

#VALORES DE ENTRADAS
directorio_entrenamiento = "./tmp/Train/"
generador_de_imagenes = ImageDataGenerator(rescale = 1./255)
generador_entrenamiento = generador_de_imagenes.flow_from_directory(
    directorio_entrenamiento,
    target_size= (width,height),
    class_mode = 'categorical',
    batch_size=126
)

#VALORES DE SALIDA
directorio_entrenamiento = "./tmp/Validation/"
generador_de_imagenes = ImageDataGenerator(rescale =1./255)
generador_validaciones = generador_de_imagenes.flow_from_directory(
    directorio_entrenamiento,
    target_size= (width,height),
    class_mode = 'categorical',
    batch_size=126
)

def redNeuronalConvolucional():    
    model = tf.keras.models.Sequential([
    #INPUT_SHAPE SE DEFINE LAS DIMENSIONES DE LAS IMAGENES, RESPETANDO EL COLOR EN RGB (3 BYTES DE COLORES)
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(width,height, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        # #mejorar la eficiencia de la red neuronal, eliminando información basura
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),

        #aplicación de la capa DENSE
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    history = model.fit_generator(generador_entrenamiento, epochs=40, validation_data=generador_validaciones, verbose=True, validation_steps=3)

    return model, history, width, height