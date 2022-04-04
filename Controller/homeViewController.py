from PyQt5.QtWidgets import QMainWindow, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import Qt, QSize
from keras.models import load_model
from numpy import double
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from View.Ui_home_view import Ui_MainWindow

class HomeViewController(QMainWindow):
    def __init__(self):
        super(HomeViewController, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.modelo = load_model("./Model/model.h5")
        # self.modelo, self.history, self.widthImage, self.heightImage = redNeuronalConvolucional()
        self.label_img = QLabel()
        self.history = pd.read_csv('./Model/training.log',sep=',',engine='python')
        self.ui.label_file_sucessful.setVisible(False)
        self.ui.label_alert.setVisible(False)
        self.ui.pushButton_buscarArchivo.clicked.connect(self.openFileNameDialog)
        self.ui.pushButton_iniciarPrediccion.clicked.connect(self.cargarArchivo)
        self.ui.pushButton_limpiar.clicked.connect(self.limpiarImagen)
        self.ui.pushButton_verGrafica.clicked.connect(self.generarGrafica)
        self.addGif()


    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "",";Image Files (*.jpg *.png)", options=options)
        if fileName:
           self.textFile = fileName
           self.ui.lineEdit_pathFile.setText(fileName)
           self.ui.label_alert.setVisible(False)


    def cargarArchivo(self):
        if(len(self.ui.lineEdit_pathFile.text()) == 0):
            self.ui.label_alert.setVisible(True)
            self.ui.label_file_sucessful.setVisible(False)
        else:
            resultTrain = ''
            imageEvalute = open(self.textFile)
            img = tf.keras.preprocessing.image.load_img(imageEvalute.name, target_size=(100,100))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            result_1 = self.modelo.predict(x)
            answer = np.argmax(result_1[0])
            print(f'Resultado: {answer}')
            if(answer == 0):
                resultTrain = 'Broca'
            elif(answer == 1):
                resultTrain = 'Martillo'
            elif(answer == 2):
                resultTrain = 'Tornillo'
            elif(answer == 3):
                resultTrain = 'Llave'
            elif(answer == 4):
                resultTrain = 'Martillo'
            elif(answer == 5):
                resultTrain = 'Perica'
            elif(answer == 6):
                resultTrain = 'Remachadora'
            elif(answer == 7):
                resultTrain = 'Tornillo'

            self.label_img.setPixmap(QPixmap(self.textFile).scaled(500,400))
            self.ui.label_file_sucessful.setVisible(True)
            self.ui.verticalLayout_image.addWidget(self.label_img, alignment=Qt.AlignCenter)
            self.ui.label_resultadoPrediccion.setText(resultTrain)


    def limpiarImagen(self):
        self.ui.label_alert.setVisible(False)
        self.ui.label_file_sucessful.setVisible(False)
        self.ui.label_resultadoPrediccion.setText('')
        self.ui.lineEdit_pathFile.clear()
        self.addGif()


    def addGif(self):
        self.textFile = './Resources/loading-2.gif'
        size = QSize(300,300)
        movie = QMovie(self.textFile)
        movie.setScaledSize(size)
        self.label_img.setMovie(movie)
        self.label_img.movie().start()
        self.label_img.movie().scaledSize()
        self.ui.verticalLayout_image.addWidget(self.label_img, alignment=Qt.AlignCenter)


    def generarGrafica(self):
        loss = self.history['loss']
        val_loss = self.history['val_loss']
        acc = self.history['accuracy']
        val_acc = self.history['val_accuracy']

        plt.subplot(1,2,1)
        plt.plot(loss, label="Training loss")
        plt.plot(val_loss, label="Validation loss")
        plt.legend()
        plt.xlabel("Iteraciones")
        plt.ylabel("Errores")
        plt.title("LOSS")

        plt.subplot(1,2,2)
        plt.plot(acc, label="Training accuracy")
        plt.plot(val_acc, label="Validation accuracy")
        plt.title("Accuracy")

        plt.xlabel("Iteraciones")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.show()