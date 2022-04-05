# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Trabajos\8cuatrimestre\Inteligencia_Artificial\Corte_3\Actividad_4\red_convolucional\View\home_view.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(860, 571)
        MainWindow.setStyleSheet("background-color: rgb(65, 89, 141);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lineEdit_pathFile = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_pathFile.setGeometry(QtCore.QRect(300, 60, 311, 31))
        self.lineEdit_pathFile.setStyleSheet("border-radius: 15px;\n"
"background-color: rgb(255, 255, 255);")
        self.lineEdit_pathFile.setReadOnly(True)
        self.lineEdit_pathFile.setObjectName("lineEdit_pathFile")
        self.pushButton_buscarArchivo = QtWidgets.QToolButton(self.centralwidget)
        self.pushButton_buscarArchivo.setGeometry(QtCore.QRect(630, 70, 31, 21))
        self.pushButton_buscarArchivo.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.pushButton_buscarArchivo.setObjectName("pushButton_buscarArchivo")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(390, 30, 151, 21))
        self.label.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 10pt \"MS Shell Dlg 2\";")
        self.label.setObjectName("label")
        self.pushButton_iniciarPrediccion = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_iniciarPrediccion.setGeometry(QtCore.QRect(300, 130, 93, 28))
        self.pushButton_iniciarPrediccion.setStyleSheet("border-radius: 10px;\n"
"font: 75 8pt \"MS Shell Dlg 2\";\n"
"color: rgb(255, 255, 255);\n"
"background-color: rgb(217, 145, 0);")
        self.pushButton_iniciarPrediccion.setObjectName("pushButton_iniciarPrediccion")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(140, 170, 581, 301))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_image = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_image.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_image.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_image.setObjectName("verticalLayout_image")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(190, 510, 81, 16))
        self.label_2.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 8pt \"Myanmar Text\";")
        self.label_2.setObjectName("label_2")
        self.label_file_sucessful = QtWidgets.QLabel(self.centralwidget)
        self.label_file_sucessful.setGeometry(QtCore.QRect(420, 100, 101, 20))
        self.label_file_sucessful.setStyleSheet("color: rgb(0, 170, 0);")
        self.label_file_sucessful.setAlignment(QtCore.Qt.AlignCenter)
        self.label_file_sucessful.setObjectName("label_file_sucessful")
        self.label_alert = QtWidgets.QLabel(self.centralwidget)
        self.label_alert.setGeometry(QtCore.QRect(430, 100, 81, 16))
        self.label_alert.setStyleSheet("color: rgb(255, 0, 0);")
        self.label_alert.setObjectName("label_alert")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(170, 60, 121, 31))
        self.label_3.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 63 8pt \"Segoe UI Semibold\";")
        self.label_3.setObjectName("label_3")
        self.label_resultadoPrediccion = QtWidgets.QLabel(self.centralwidget)
        self.label_resultadoPrediccion.setGeometry(QtCore.QRect(280, 510, 271, 16))
        self.label_resultadoPrediccion.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 8pt \"Myanmar Text\";")
        self.label_resultadoPrediccion.setText("")
        self.label_resultadoPrediccion.setObjectName("label_resultadoPrediccion")
        self.pushButton_limpiar = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_limpiar.setGeometry(QtCore.QRect(490, 130, 93, 28))
        self.pushButton_limpiar.setStyleSheet("border-radius: 10px;\n"
"font: 75 8pt \"MS Shell Dlg 2\";\n"
"color: rgb(255, 255, 255);\n"
"background-color: rgb(255, 68, 43);")
        self.pushButton_limpiar.setObjectName("pushButton_limpiar")
        self.pushButton_verGrafica = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_verGrafica.setGeometry(QtCore.QRect(740, 270, 111, 28))
        self.pushButton_verGrafica.setStyleSheet("border-radius: 10px;\n"
"font: 75 8pt \"MS Shell Dlg 2\";\n"
"color: rgb(255, 255, 255);\n"
"background-color: rgb(170, 170, 0);")
        self.pushButton_verGrafica.setObjectName("pushButton_verGrafica")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_buscarArchivo.setText(_translate("MainWindow", "..."))
        self.label.setText(_translate("MainWindow", "Seleccionar imagen"))
        self.pushButton_iniciarPrediccion.setText(_translate("MainWindow", "INICIAR"))
        self.label_2.setText(_translate("MainWindow", "RESULTADO: "))
        self.label_file_sucessful.setText(_translate("MainWindow", "File Sucessful!"))
        self.label_alert.setText(_translate("MainWindow", "Import a file!"))
        self.label_3.setText(_translate("MainWindow", "Selecccionar archivo:"))
        self.pushButton_limpiar.setText(_translate("MainWindow", "LIMPIAR"))
        self.pushButton_verGrafica.setText(_translate("MainWindow", "VER GRÁFICA"))
