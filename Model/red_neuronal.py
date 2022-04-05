from gc import callbacks
import zipfile
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger

pathDataSet = '../tmp'
isExist = os.path.exists(pathDataSet)

if not isExist:
    local_zip = '../Zips/Dataset.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall(pathDataSet)
    zip_ref.close()
else:
    print('LA CARPETA EXISTE')


#DIMENSIONES DE LAS IMAGENES y aumento de datos(rotacion, zoom)
width = 100
height = 100

#VALORES DE ENTRADAS
directorio_entrenamiento = "../tmp/Train/"
generador_de_imagenes = ImageDataGenerator(
    rescale = 1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

generador_entrenamiento = generador_de_imagenes.flow_from_directory(
    directorio_entrenamiento,
    target_size= (width,height),
    batch_size=32,
    class_mode = 'categorical',
)

nClases = generador_entrenamiento.num_classes
# print(generador_entrenamiento.classes)
print(f"Número de clases {nClases}")
#VALORES DE SALIDA

generador_de_imagenes_validacion = ImageDataGenerator(
    rescale = 1./255,
)

directorio_validaciones = "../tmp/Validation/"

generador_validaciones = generador_de_imagenes_validacion.flow_from_directory(
    directorio_validaciones,
    target_size= (width,height),
    batch_size=32,
    class_mode = 'categorical',
)

# def redNeuronalConvolucional():
model = tf.keras.models.Sequential([
#INPUT_SHAPE SE DEFINE LAS DIMENSIONES DE LAS IMAGENES, RESPETANDO EL COLOR EN RGB (3 BYTES DE COLORES)
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width,height, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    # #mejorar la eficiencia de la red neuronal, eliminando información basura
    tf.keras.layers.Flatten(),

    #aplicación de la capa DENSE
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(nClases, activation='softmax')
])

csv_logger = CSVLogger('training.log', separator=',', append=False)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
history = model.fit(generador_entrenamiento, validation_data=generador_validaciones, epochs=50, verbose=True, callbacks=[csv_logger])
model.save("model.h5")
    # return model, history, width, height
