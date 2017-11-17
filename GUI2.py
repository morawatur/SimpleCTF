import Constants as const
import Dm3Reader3 as dm3
import ImageSupport as imsup
import ctf_calc

import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets

# -------------------------------------------------------------------

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app=None):
        super(Ui_MainWindow, self).__init__()
        self.rings = []
        self.app = app
        self.setupUi()

    def setupUi(self):
        self.setObjectName('MainWindow')
        self.resize(750, 580)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName('centralwidget')

        fileDialog = QtWidgets.QFileDialog()
        img_path = QtWidgets.QFileDialog.getOpenFileName()[0]
        img_data, px_dims = dm3.ReadDm3File(img_path)
        img = imsup.ImageWithBuffer(img_data.shape[0], img_data.shape[1], imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'], px_dim_sz=px_dims[0])
        img.LoadAmpData(img_data)

        self.img_view = GraphicsLabel(self, img)
        self.img_view.setGeometry(QtCore.QRect(20, 10, const.ccWidgetDim, const.ccWidgetDim))
        self.img_view.setObjectName('img_view')

        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(560, 20, 160, 340))
        self.verticalLayoutWidget.setObjectName('verticalLayoutWidget')
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName('verticalLayout')

        self.df_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.df_label.setEnabled(True)
        self.df_label.setObjectName('df_label')
        self.verticalLayout.addWidget(self.df_label)

        self.df_edit = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.df_edit.setMinimum(-1e4)
        self.df_edit.setMaximum(1e4)
        self.df_edit.setSingleStep(10)
        self.df_edit.setProperty('value', 100)
        self.df_edit.setObjectName('df_edit')
        self.verticalLayout.addWidget(self.df_edit)

        self.A1_amp_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.A1_amp_label.setObjectName('A1_amp_label')
        self.verticalLayout.addWidget(self.A1_amp_label)

        self.A1_amp_edit = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.A1_amp_edit.setMinimum(0)
        self.A1_amp_edit.setMaximum(1e4)
        self.A1_amp_edit.setSingleStep(10)
        self.A1_amp_edit.setProperty('value', 0)
        self.A1_amp_edit.setObjectName('A1_amp_edit')
        self.verticalLayout.addWidget(self.A1_amp_edit)

        self.A1_phs_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.A1_phs_label.setObjectName('A1_phs_label')
        self.verticalLayout.addWidget(self.A1_phs_label)

        self.A1_phs_edit = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.A1_phs_edit.setMinimum(0)
        self.A1_phs_edit.setMaximum(359)
        self.A1_phs_edit.setSingleStep(1)
        self.A1_phs_edit.setProperty('value', 0)
        self.A1_phs_edit.setObjectName('A1_phs_edit')
        self.verticalLayout.addWidget(self.A1_phs_edit)

        self.Cs_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.Cs_label.setObjectName('Cs_label')
        self.verticalLayout.addWidget(self.Cs_label)

        self.Cs_edit = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.Cs_edit.setMinimum(0)
        self.Cs_edit.setMaximum(1000)
        self.Cs_edit.setSingleStep(10)
        self.Cs_edit.setProperty('value', 0)
        self.Cs_edit.setObjectName('Cs_edit')
        self.verticalLayout.addWidget(self.Cs_edit)

        self.df_spread_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.df_spread_label.setObjectName('df_spread_label')
        self.verticalLayout.addWidget(self.df_spread_label)

        self.df_spread_edit = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.df_spread_edit.setMinimum(0)
        self.df_spread_edit.setMaximum(100)
        self.df_spread_edit.setSingleStep(1)
        self.df_spread_edit.setProperty('value', 0)
        self.df_spread_edit.setObjectName('df_spread_edit')
        self.verticalLayout.addWidget(self.df_spread_edit)

        self.conv_angle_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.conv_angle_label.setObjectName('conv_angle_label')
        self.verticalLayout.addWidget(self.conv_angle_label)

        self.conv_angle_edit = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.conv_angle_edit.setMinimum(0)
        self.conv_angle_edit.setMaximum(359)
        self.conv_angle_edit.setSingleStep(1)
        self.conv_angle_edit.setProperty('value', 0)
        self.conv_angle_edit.setObjectName('conv_angle_edit')
        self.verticalLayout.addWidget(self.conv_angle_edit)

        self.img_view.raise_()
        self.verticalLayoutWidget.raise_()

        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 750, 21))
        self.menubar.setObjectName('menubar')
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName('statusbar')
        self.setStatusBar(self.statusbar)

        # self.find_model_button.clicked.connect(self.run_ransac)
        # self.export_button.clicked.connect(self.export_image)
        # self.crop_button.clicked.connect(self.crop_image)

        self.statusbar.showMessage('Ready')
        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle('RANSAC Ellipse Fitting')
        self.df_label.setText('Defocus [nm]')
        self.A1_amp_label.setText('A1 amplitude [nm]')
        self.A1_phs_label.setText('A1 angle [deg]')
        self.Cs_label.setText('Cs [nm]')
        self.df_spread_label.setText('Defocus spread [nm]')
        self.conv_angle_label.setText('Conv. angle [mrad]')

    def draw_rings(self):
        self.img_view.update()

# -------------------------------------------------------------------

class GraphicsLabel(QtWidgets.QLabel):
    def __init__(self, parent, image=None):
        super(GraphicsLabel, self).__init__(parent)

        self.image = image
        # self.image.UpdateImageFromBuffer()
        # padded_image = imsup.PadImageBufferToNx512(self.image, np.max(self.image.buffer))
        p_image = np.copy(self.image.buffer)
        q_image = QtGui.QImage(imsup.ScaleImage(p_image, 0.0, 255.0).astype(np.uint8),
                               p_image.shape[1], p_image.shape[0], QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap(q_image)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)

        self.view = QtWidgets.QGraphicsView(self)
        self.scene = QtWidgets.QGraphicsScene()
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

        coeff = const.ccWidgetDim / self.image.width
        print('coeff = {0}'.format(coeff))

        for item in self.scene.items()[:-1]:
            self.scene.removeItem(item)

        for e in self.parent().ellipses:
            aa = e.a * coeff
            bb = e.b * coeff
            b = min(aa, bb)
            a = max(aa, bb)
            e_item = QtWidgets.QGraphicsEllipseItem(mid_x - a, mid_y - b, 2 * a, 2 * b)
            e_item.setTransformOriginPoint(mid_x, mid_y)
            e_item.setPen(line_pen)
            self.scene.addItem(e_item)

# -------------------------------------------------------------------

def run_aberr_window():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    aberr_win = Ui_MainWindow(app)
    sys.exit(app.exec_())

run_aberr_window()
