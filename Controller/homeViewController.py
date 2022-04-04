from PyQt5.QtWidgets import QMainWindow, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import Qt, QSize
from numpy import double
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from Model.red_neuronal import getModelSaved, redNeuronalConvolucional
from View.Ui_home_view import Ui_MainWindow
import pandas as pd

class HomeViewController(QMainWindow):
    def __init__(self):
        super(HomeViewController, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.modelo, self.widthImage, self.heightImage = getModelSaved()
        self.label_img = QLabel()
        self.ui.label_file_sucessful.setVisible(False)
        self.ui.label_alert.setVisible(False)
        self.ui.pushButton_buscarArchivo.clicked.connect(self.openFileNameDialog)
        self.ui.pushButton_iniciarPrediccion.clicked.connect(self.cargarArchivo)
        self.ui.pushButton_limpiar.clicked.connect(self.limpiarImagen)
        self.ui.pushButton_graficaLoss.clicked.connect(self.generarGraficaLoss)
        self.ui.pushButton_graficaAccuracy.clicked.connect(self.generarGraficaAcurracy)
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
            img = tf.keras.preprocessing.image.load_img(imageEvalute.name, target_size=(self.widthImage,self.heightImage))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            result_1 = self.modelo.predict(images, batch_size=10)
            print(f'Resultados: {result_1[0]}')
            if(result_1[0][0] == 1):
                resultTrain = 'Broca'
            elif(result_1[0][1] == 1):
                resultTrain = 'Desarmador'
            elif(result_1[0][2] == 1):
                resultTrain = 'Llave Alen'
            elif(result_1[0][3] == 1):
                resultTrain = 'Llave'
            elif(result_1[0][4] == 1):
                resultTrain = 'Martillo'
            elif(result_1[0][5] == 1):
                resultTrain = 'Perica'
            elif(result_1[0][6] == 1):
                resultTrain = 'Remachadora'
            elif(result_1[0][7] == 1):
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


    def generarGraficaLoss(self):
        loss = pd.read_csv("./FileSaved/history.csv")
        loss = loss['loss']
        plt.plot(loss, label="LOSS RGB")
        plt.legend()
        plt.xlabel("iteraciones")
        plt.ylabel("errores")
        plt.show()
        

    def generarGraficaAcurracy(self):
        loss = pd.read_csv("./FileSaved/history.csv")
        loss = loss['accuracy']
        plt.plot(loss, label="accuracy")
        plt.legend()
        plt.xlabel("iteraciones")
        plt.ylabel("errores")
        plt.show()