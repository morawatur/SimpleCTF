# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ransac.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

import Constants as const
import Dm3Reader3 as dm3
import ImageSupport as imsup

import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets

# -------------------------------------------------------------------

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app=None):
        super(Ui_MainWindow, self).__init__()
        self.ellipses = []
        self.app = app
        self.setupUi()

    def setupUi(self):
        self.setObjectName('MainWindow')
        self.resize(750, 580)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName('centralwidget')

        img_path = QtWidgets.QFileDialog.getOpenFileName()
        img = dm3.ReadDm3File(img_path)
        self.img_view = GraphicsLabel(self, img)
        self.img_view.setGeometry(QtCore.QRect(20, 10, const.ccWidgetDim, const.ccWidgetDim))
        self.img_view.setObjectName('img_view')

        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(560, 20, 160, 310))
        self.verticalLayoutWidget.setObjectName('verticalLayoutWidget')
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName('verticalLayout')
        self.n_iter_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.n_iter_label.setEnabled(True)
        self.n_iter_label.setObjectName('n_iter_label')
        self.verticalLayout.addWidget(self.n_iter_label)
        self.n_iter_edit = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.n_iter_edit.setMinimum(1)
        self.n_iter_edit.setMaximum(2000)
        self.n_iter_edit.setSingleStep(10)
        self.n_iter_edit.setProperty('value', 100)
        self.n_iter_edit.setObjectName('n_iter_edit')
        self.verticalLayout.addWidget(self.n_iter_edit)
        self.n_inl_threshold_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.n_inl_threshold_label.setObjectName('n_inl_threshold_label')
        self.verticalLayout.addWidget(self.n_inl_threshold_label)
        self.n_inl_threshold_edit = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.n_inl_threshold_edit.setMinimum(1000)
        self.n_inl_threshold_edit.setMaximum(20000)
        self.n_inl_threshold_edit.setSingleStep(100)
        self.n_inl_threshold_edit.setProperty("value", 7000)
        self.n_inl_threshold_edit.setObjectName(_fromUtf8("n_inl_threshold_edit"))
        self.verticalLayout.addWidget(self.n_inl_threshold_edit)

        self.n_ell_to_find_label = QtGui.QLabel(self.verticalLayoutWidget)
        self.n_ell_to_find_label.setObjectName(_fromUtf8("n_ell_to_find_label"))
        self.verticalLayout.addWidget(self.n_ell_to_find_label)

        self.n_ell_to_find_edit = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.n_ell_to_find_edit.setMinimum(0)       # if val == 0 then finish after n_iterations
        self.n_ell_to_find_edit.setMaximum(10)
        self.n_ell_to_find_edit.setSingleStep(1)
        self.n_ell_to_find_edit.setProperty("value", 3)
        self.n_ell_to_find_edit.setObjectName(_fromUtf8("n_ell_to_find_edit"))
        self.verticalLayout.addWidget(self.n_ell_to_find_edit)

        self.try_again_threshold_label = QtGui.QLabel(self.verticalLayoutWidget)
        self.try_again_threshold_label.setObjectName(_fromUtf8("try_again_threshold_label"))
        self.verticalLayout.addWidget(self.try_again_threshold_label)
        self.try_again_threshold_edit = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.try_again_threshold_edit.setMinimum(100)
        self.try_again_threshold_edit.setMaximum(50000)
        self.try_again_threshold_edit.setSingleStep(100)
        self.try_again_threshold_edit.setProperty("value", 3000)
        self.try_again_threshold_edit.setObjectName(_fromUtf8("try_again_threshold_edit"))
        self.verticalLayout.addWidget(self.try_again_threshold_edit)
        self.min_dist_label = QtGui.QLabel(self.verticalLayoutWidget)
        self.min_dist_label.setObjectName(_fromUtf8("min_dist_label"))
        self.verticalLayout.addWidget(self.min_dist_label)
        self.min_dist_edit = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.min_dist_edit.setMinimum(1)
        self.min_dist_edit.setMaximum(100)
        self.min_dist_edit.setProperty("value", 5)
        self.min_dist_edit.setObjectName(_fromUtf8("min_dist_edit"))
        self.verticalLayout.addWidget(self.min_dist_edit)
        self.max_n_tries_label = QtGui.QLabel(self.verticalLayoutWidget)
        self.max_n_tries_label.setObjectName(_fromUtf8("max_n_tries_label"))
        self.verticalLayout.addWidget(self.max_n_tries_label)
        self.max_n_tries_edit = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.max_n_tries_edit.setMinimum(1)
        self.max_n_tries_edit.setMaximum(100)
        self.max_n_tries_edit.setProperty("value", 20)
        self.max_n_tries_edit.setObjectName(_fromUtf8("max_n_tries_edit"))
        self.verticalLayout.addWidget(self.max_n_tries_edit)
        self.min_ab_ratio_label = QtGui.QLabel(self.verticalLayoutWidget)
        self.min_ab_ratio_label.setObjectName(_fromUtf8("min_ab_ratio_label"))
        self.verticalLayout.addWidget(self.min_ab_ratio_label)
        self.min_ab_ratio_edit = QtGui.QDoubleSpinBox(self.verticalLayoutWidget)
        self.min_ab_ratio_edit.setMaximum(1.0)
        self.min_ab_ratio_edit.setSingleStep(0.05)
        self.min_ab_ratio_edit.setProperty("value", 0.3)
        self.min_ab_ratio_edit.setObjectName(_fromUtf8("min_ab_ratio_edit"))
        self.verticalLayout.addWidget(self.min_ab_ratio_edit)
        self.verticalLayoutWidget_2 = QtGui.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(560, 330, 160, 181))
        self.verticalLayoutWidget_2.setObjectName(_fromUtf8("verticalLayoutWidget_2"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.find_model_button = QtGui.QPushButton(self.verticalLayoutWidget_2)
        self.find_model_button.setObjectName(_fromUtf8("find_model_button"))
        self.verticalLayout_2.addWidget(self.find_model_button)
        self.export_button = QtGui.QPushButton(self.verticalLayoutWidget_2)
        self.export_button.setObjectName(_fromUtf8("export_button"))
        self.verticalLayout_2.addWidget(self.export_button)
        self.crop_button = QtGui.QPushButton(self.verticalLayoutWidget_2)
        self.crop_button.setObjectName(_fromUtf8("crop_button"))
        self.verticalLayout_2.addWidget(self.crop_button)
        self.img_view.raise_()
        self.verticalLayoutWidget.raise_()
        self.n_iter_label.raise_()
        self.verticalLayoutWidget_2.raise_()
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 750, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(self)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        self.setStatusBar(self.statusbar)

        self.find_model_button.clicked.connect(self.run_ransac)
        # self.export_button.clicked.connect(self.export_image)
        # self.crop_button.clicked.connect(self.crop_image)

        self.statusbar.showMessage('Ready')
        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "RANSAC Ellipse Fitting", None))
        self.n_iter_label.setText(_translate("MainWindow", "Num. of iterations", None))
        self.n_inl_threshold_label.setText(_translate("MainWindow", "Num. of inliers (threshold)", None))
        self.n_ell_to_find_label.setText(_translate("MainWindow", "Num. of ellipses to find", None))
        self.try_again_threshold_label.setText(_translate("MainWindow", "Try-again threshold", None))
        self.min_dist_label.setText(_translate("MainWindow", "Min. dist. from model [px]", None))
        self.max_n_tries_label.setText(_translate("MainWindow", "Max. num. of tries", None))
        self.min_ab_ratio_label.setText(_translate("MainWindow", "Min. a-b ratio", None))
        self.find_model_button.setText(_translate("MainWindow", "Start", None))
        self.export_button.setText(_translate("MainWindow", "Export", None))
        self.crop_button.setText(_translate("MainWindow", "Crop", None))

    def run_ransac(self):
        self.statusbar.showMessage('Found ellipses: 0')
        self.ellipses = []
        self.ellipses = ran.fit_model_to_image(self.img_view.image,
                                               self.n_iter_edit.value(),
                                               self.n_inl_threshold_edit.value(),
                                               self.n_ell_to_find_edit.value(),
                                               self.try_again_threshold_edit.value(),
                                               self.min_dist_edit.value(),
                                               self.max_n_tries_edit.value(),
                                               self.min_ab_ratio_edit.value(),
                                               self)
        self.img_view.update()

# -------------------------------------------------------------------

class GraphicsLabel(QtGui.QLabel):
    def __init__(self, parent, image=None):
        super(GraphicsLabel, self).__init__(parent)

        self.image = image
        q_image = QtGui.QImage(imsup.ScaleImage(self.image, 0.0, 255.0).astype(np.uint8),
                               self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap(q_image)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)

        self.view = QtGui.QGraphicsView(self)
        self.scene = QtGui.QGraphicsScene()
        self.view.setScene(self.scene)

        self.scene.addPixmap(pixmap)
        self.view.show()
        self.update()

    def update(self):
        super(GraphicsLabel, self).update()

        mid_x = const.ccWidgetDim // 2
        mid_y = const.ccWidgetDim // 2

        line_pen = QtGui.QPen(QtCore.Qt.yellow)
        line_pen.setCapStyle(QtCore.Qt.RoundCap)
        line_pen.setWidth(2)

        coeff = const.ccWidgetDim / self.image.shape[0]
        print('coeff = {0}'.format(coeff))

        for item in self.scene.items()[:-1]:
            self.scene.removeItem(item)

        for e in self.parent().ellipses:
            aa = e.a * coeff
            bb = e.b * coeff
            b = min(aa, bb)
            a = max(aa, bb)
            e_item = QtGui.QGraphicsEllipseItem(mid_x - a, mid_y - b, 2 * a, 2 * b)
            e_item.setTransformOriginPoint(mid_x, mid_y)
            print(e.tau, ran.rad2deg(e.tau))
            e_item.setRotation(ran.rad2deg(e.tau))
            e_item.setPen(line_pen)
            self.scene.addItem(e_item)

# -------------------------------------------------------------------

def run_ransac_window():
    import sys
    app = QtGui.QApplication(sys.argv)
    ransac_win = Ui_MainWindow(app)
    sys.exit(app.exec_())
