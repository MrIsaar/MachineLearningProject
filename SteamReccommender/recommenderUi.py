# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\SteamReccommender\RecommendUi.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from recommender import Recommender


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1500, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.MainLabel = QtWidgets.QLabel(self.centralwidget)
        self.MainLabel.setGeometry(QtCore.QRect(20, 10, 200, 100))
        self.MainLabel.setObjectName("MainLabel")
        self.reloadButton = QtWidgets.QPushButton(self.centralwidget)
        self.reloadButton.setGeometry(QtCore.QRect(80, 480, 151, 61))
        self.reloadButton.setObjectName("reloadButton")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(260, 20, 1200, 511))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.outputLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.outputLayout.setContentsMargins(0, 0, 0, 0)
        self.outputLayout.setObjectName("outputLayout")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.label0 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label0.setObjectName("label0")
        self.horizontalLayout_13.addWidget(self.label0)
        self.like0 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.like0.setObjectName("like0")
        self.horizontalLayout_13.addWidget(self.like0)
        self.dislike0 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        
        self.dislike0.setObjectName("dislike0")
        self.horizontalLayout_13.addWidget(self.dislike0)
        self.outputLayout.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.label1 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label1.setObjectName("label1")
        self.horizontalLayout_15.addWidget(self.label1)
        self.like1 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.like1.setObjectName("like1")
        self.horizontalLayout_15.addWidget(self.like1)
        self.dislike1 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.dislike1.setObjectName("dislike1")
        self.horizontalLayout_15.addWidget(self.dislike1)
        self.outputLayout.addLayout(self.horizontalLayout_15)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.label2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label2.setObjectName("label2")
        self.horizontalLayout_16.addWidget(self.label2)
        self.like2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.like2.setObjectName("like2")
        self.horizontalLayout_16.addWidget(self.like2)
        self.dislike2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.dislike2.setObjectName("dislike2")
        self.horizontalLayout_16.addWidget(self.dislike2)
        self.outputLayout.addLayout(self.horizontalLayout_16)
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.label3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label3.setObjectName("label3")
        self.horizontalLayout_17.addWidget(self.label3)
        self.like3 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.like3.setObjectName("like3")
        self.horizontalLayout_17.addWidget(self.like3)
        self.dislike3 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.dislike3.setObjectName("dislike3")
        self.horizontalLayout_17.addWidget(self.dislike3)
        self.outputLayout.addLayout(self.horizontalLayout_17)
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.label4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label4.setObjectName("label4")
        self.horizontalLayout_18.addWidget(self.label4)
        self.like4 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.like4.setObjectName("like4")
        self.horizontalLayout_18.addWidget(self.like4)
        self.dislike4 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.dislike4.setObjectName("dislike4")
        self.horizontalLayout_18.addWidget(self.dislike4)
        self.outputLayout.addLayout(self.horizontalLayout_18)
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.label5 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label5.setObjectName("label5")
        self.horizontalLayout_19.addWidget(self.label5)
        self.like5 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.like5.setObjectName("like5")
        self.horizontalLayout_19.addWidget(self.like5)
        self.dislike5 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.dislike5.setObjectName("dislike5")
        self.horizontalLayout_19.addWidget(self.dislike5)
        self.outputLayout.addLayout(self.horizontalLayout_19)
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.label6 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label6.setObjectName("label6")
        self.horizontalLayout_21.addWidget(self.label6)
        self.like6 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.like6.setObjectName("like6")
        self.horizontalLayout_21.addWidget(self.like6)
        self.dislike6 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.dislike6.setObjectName("dislike6")
        self.horizontalLayout_21.addWidget(self.dislike6)
        self.outputLayout.addLayout(self.horizontalLayout_21)
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.label7 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label7.setObjectName("label7")
        self.horizontalLayout_20.addWidget(self.label7)
        self.like7 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.like7.setObjectName("like7")
        self.horizontalLayout_20.addWidget(self.like7)
        self.dislike7 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.dislike7.setObjectName("dislike7")
        self.horizontalLayout_20.addWidget(self.dislike7)
        self.outputLayout.addLayout(self.horizontalLayout_20)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.label8 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label8.setObjectName("label8")
        self.horizontalLayout_14.addWidget(self.label8)
        self.like8 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.like8.setObjectName("like8")
        self.horizontalLayout_14.addWidget(self.like8)
        self.dislike8 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.dislike8.setObjectName("dislike8")
        self.horizontalLayout_14.addWidget(self.dislike8)
        self.outputLayout.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label9 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label9.setObjectName("label9")
        self.horizontalLayout_11.addWidget(self.label9)
        self.like9 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.like9.setObjectName("like9")
        self.horizontalLayout_11.addWidget(self.like9)
        self.dislike9 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.dislike9.setObjectName("dislike9")
        self.horizontalLayout_11.addWidget(self.dislike9)
        self.outputLayout.addLayout(self.horizontalLayout_11)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 846, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.setSizeRestrains()
        
    def setSizeRestrains(self):
        
        self.dislike0.setMaximumSize(QtCore.QSize(100, 16777215))
        self.dislike1.setMaximumSize(QtCore.QSize(100, 16777215))
        self.dislike2.setMaximumSize(QtCore.QSize(100, 16777215))
        self.dislike3.setMaximumSize(QtCore.QSize(100, 16777215))
        self.dislike4.setMaximumSize(QtCore.QSize(100, 16777215))
        self.dislike5.setMaximumSize(QtCore.QSize(100, 16777215))
        self.dislike6.setMaximumSize(QtCore.QSize(100, 16777215))
        self.dislike7.setMaximumSize(QtCore.QSize(100, 16777215))
        self.dislike8.setMaximumSize(QtCore.QSize(100, 16777215))
        self.dislike9.setMaximumSize(QtCore.QSize(100, 16777215))
        self.like0.setMaximumSize   (QtCore.QSize(100, 16777215))
        self.like1.setMaximumSize   (QtCore.QSize(100, 16777215))
        self.like2.setMaximumSize   (QtCore.QSize(100, 16777215))
        self.like3.setMaximumSize   (QtCore.QSize(100, 16777215))
        self.like4.setMaximumSize   (QtCore.QSize(100, 16777215))
        self.like5.setMaximumSize   (QtCore.QSize(100, 16777215))
        self.like6.setMaximumSize   (QtCore.QSize(100, 16777215))
        self.like7.setMaximumSize   (QtCore.QSize(100, 16777215))
        self.like8.setMaximumSize   (QtCore.QSize(100, 16777215))
        self.like9.setMaximumSize   (QtCore.QSize(100, 16777215))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Steam Recommender"))
        self.MainLabel.setText(_translate("MainWindow", "Steam Recommender\nwait a few seconds\nwhen loading suggestions"))
        self.reloadButton.setText(_translate("MainWindow", "New Suggestions"))
        self.label0.setText(_translate("MainWindow", "GameData"))
        self.like0.setText(_translate("MainWindow", "like"))
        self.dislike0.setText(_translate("MainWindow", "dislike"))
        self.label1.setText(_translate("MainWindow", "GameData"))
        self.like1.setText(_translate("MainWindow", "like"))
        self.dislike1.setText(_translate("MainWindow", "dislike"))
        self.label2.setText(_translate("MainWindow", "GameData"))
        self.like2.setText(_translate("MainWindow", "like"))
        self.dislike2.setText(_translate("MainWindow", "dislike"))
        self.label3.setText(_translate("MainWindow", "GameData"))
        self.like3.setText(_translate("MainWindow", "like"))
        self.dislike3.setText(_translate("MainWindow", "dislike"))
        self.label4.setText(_translate("MainWindow", "GameData"))
        self.like4.setText(_translate("MainWindow", "like"))
        self.dislike4.setText(_translate("MainWindow", "dislike"))
        self.label5.setText(_translate("MainWindow", "GameData"))
        self.like5.setText(_translate("MainWindow", "like"))
        self.dislike5.setText(_translate("MainWindow", "dislike"))
        self.label6.setText(_translate("MainWindow", "GameData"))
        self.like6.setText(_translate("MainWindow", "like"))
        self.dislike6.setText(_translate("MainWindow", "dislike"))
        self.label7.setText(_translate("MainWindow", "GameData"))
        self.like7.setText(_translate("MainWindow", "like"))
        self.dislike7.setText(_translate("MainWindow", "dislike"))
        self.label8.setText(_translate("MainWindow", "GameData"))
        self.like8.setText(_translate("MainWindow", "like"))
        self.dislike8.setText(_translate("MainWindow", "dislike"))
        self.label9.setText(_translate("MainWindow", "GameData"))
        self.like9.setText(_translate("MainWindow", "like"))
        self.dislike9.setText(_translate("MainWindow", "dislike"))
        
        self.like0.clicked.connect(lambda:self.clicked(self.like0))
        self.like1.clicked.connect(lambda:self.clicked(self.like1))
        self.like2.clicked.connect(lambda:self.clicked(self.like2))
        self.like3.clicked.connect(lambda:self.clicked(self.like3))
        self.like4.clicked.connect(lambda:self.clicked(self.like4))
        self.like5.clicked.connect(lambda:self.clicked(self.like5))
        self.like6.clicked.connect(lambda:self.clicked(self.like6))
        self.like7.clicked.connect(lambda:self.clicked(self.like7))
        self.like8.clicked.connect(lambda:self.clicked(self.like8))
        self.like9.clicked.connect(lambda:self.clicked(self.like9))
        
        self.dislike0.clicked.connect(lambda:self.clicked(self.dislike0))
        self.dislike1.clicked.connect(lambda:self.clicked(self.dislike1))
        self.dislike2.clicked.connect(lambda:self.clicked(self.dislike2))
        self.dislike3.clicked.connect(lambda:self.clicked(self.dislike3))
        self.dislike4.clicked.connect(lambda:self.clicked(self.dislike4))
        self.dislike5.clicked.connect(lambda:self.clicked(self.dislike5))
        self.dislike6.clicked.connect(lambda:self.clicked(self.dislike6))
        self.dislike7.clicked.connect(lambda:self.clicked(self.dislike7))
        self.dislike8.clicked.connect(lambda:self.clicked(self.dislike8))
        self.dislike9.clicked.connect(lambda:self.clicked(self.dislike9))
        
        self.recommender = Recommender()
        self.loadData()
        self.reloadButton.clicked.connect(self.loadData)
        
    def loadData(self):
        data = self.recommender.recommended(maxcount=10)
        
        self.label0.setText(data[0])
        self.label1.setText(data[1])
        self.label2.setText(data[2])
        self.label3.setText(data[3])
        self.label4.setText(data[4])
        self.label5.setText(data[5])
        self.label6.setText(data[6])
        self.label7.setText(data[7])
        self.label8.setText(data[8])
        self.label9.setText(data[9])
        
        self.recommender.updatemodel()
        
        
    def clicked(self,b):
        label = None
        try:
            if b.objectName()[0] == 'l':
                if b.objectName()[-1] == '0': self.recommender.rate(0,1)
                if b.objectName()[-1] == '1': self.recommender.rate(1,1)
                if b.objectName()[-1] == '2': self.recommender.rate(2,1)
                if b.objectName()[-1] == '3': self.recommender.rate(3,1)
                if b.objectName()[-1] == '4': self.recommender.rate(4,1)
                if b.objectName()[-1] == '5': self.recommender.rate(5,1)    
                if b.objectName()[-1] == '6': self.recommender.rate(6,1)
                if b.objectName()[-1] == '7': self.recommender.rate(7,1)
                if b.objectName()[-1] == '8': self.recommender.rate(8,1)
                if b.objectName()[-1] == '9': self.recommender.rate(9,1)
            else:
                if b.objectName()[-1] == '0': self.recommender.rate(0,-1)
                if b.objectName()[-1] == '1': self.recommender.rate(1,-1)
                if b.objectName()[-1] == '2': self.recommender.rate(2,-1)
                if b.objectName()[-1] == '3': self.recommender.rate(3,-1)
                if b.objectName()[-1] == '4': self.recommender.rate(4,-1)
                if b.objectName()[-1] == '5': self.recommender.rate(5,-1)    
                if b.objectName()[-1] == '6': self.recommender.rate(6,-1)
                if b.objectName()[-1] == '7': self.recommender.rate(7,-1)
                if b.objectName()[-1] == '8': self.recommender.rate(8,-1)
                if b.objectName()[-1] == '9': self.recommender.rate(9,-1)
        except:
            pass
        
   


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
