import Constants as const
import CrossCorr as cc
import Dm3Reader3 as dm3
import ImageSupport as imsup
import ctf_calc
import aberrations as ab
import simulation as sim

import numpy as np
import copy
from functools import partial
from PyQt5 import QtGui, QtCore, QtWidgets

# -------------------------------------------------------------------

class Thon_ring:
    def __init__(self, mid=(0, 0), a=0, b=0, tau=0.0):
        self.mid = mid
        self.r1 = a
        self.r2 = b
        self.tilt = tau

    def copy(self):
        return copy.deepcopy(self)

    def set_mid(self, x, y):
        self.mid = x, y

    def set_r1(self, a):
        self.r1 = a

    def set_r2(self, b):
        self.r2 = b

    def set_tilt(self, tau):
        self.tilt = tau

# ------------------------------------------------------------

def deg2rad(deg):
    return np.pi * deg / 180.0

# ------------------------------------------------------------

def rad2deg(rad):
    return 180.0 * rad / np.pi

# -------------------------------------------------------------------

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app=None):
        super(Ui_MainWindow, self).__init__()
        self.rings = []
        self.aberrs = ab.Aberrations()
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
        # px_dims[0] = 40e-12
        fft = np.fft.fft2(img_data)
        fft_amp, fft_phs = ab.complex2polar(fft)
        diff_amp = sim.fft2diff(fft_amp)
        diff_amp = np.log10(diff_amp)

        img = imsup.ImageWithBuffer(img_data.shape[0], img_data.shape[1], imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'], px_dim_sz=px_dims[0])
        img.LoadAmpData(diff_amp)

        self.img_view = GraphicsLabel(self, img)
        self.img_view.setGeometry(QtCore.QRect(20, 10, const.ccWidgetDim, const.ccWidgetDim))
        self.img_view.setObjectName('img_view')

        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(560, 20, 160, 340))
        self.verticalLayoutWidget.setObjectName('verticalLayoutWidget')
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName('verticalLayout')

        self.n_rings_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.n_rings_label.setEnabled(True)
        self.n_rings_label.setObjectName('n_rings_label')
        self.verticalLayout.addWidget(self.n_rings_label)

        self.n_rings_edit = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.n_rings_edit.setMinimum(1)
        self.n_rings_edit.setMaximum(5)
        self.n_rings_edit.setSingleStep(1)
        self.n_rings_edit.setProperty('value', 1)
        self.n_rings_edit.setObjectName('n_rings_edit')
        self.verticalLayout.addWidget(self.n_rings_edit)

        self.df_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.df_label.setEnabled(True)
        self.df_label.setObjectName('df_label')
        self.verticalLayout.addWidget(self.df_label)

        self.df_edit = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.df_edit.setMinimum(-1e4)
        self.df_edit.setMaximum(1e4)
        self.df_edit.setSingleStep(10)
        self.df_edit.setProperty('value', 20)
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

        self.bright_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bright_slider.setRange(0.0, 100.0)
        self.bright_slider.setValue(50.0)

        self.cont_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.cont_slider.setRange(1.0, 100.0)
        self.cont_slider.setValue(50.0)

        self.bright_slider.valueChanged.connect(self.img_view.change_gain_and_bias)
        self.cont_slider.valueChanged.connect(self.img_view.change_gain_and_bias)

        self.verticalLayout.addWidget(self.bright_slider)
        self.verticalLayout.addWidget(self.cont_slider)

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

        self.n_rings_edit.valueChanged.connect(self.update_rings)
        self.df_edit.valueChanged.connect(self.update_df)
        self.A1_amp_edit.valueChanged.connect(self.update_A1)
        self.A1_phs_edit.valueChanged.connect(self.update_A1)
        self.Cs_edit.valueChanged.connect(self.update_Cs)
        self.df_spread_edit.valueChanged.connect(self.update_df_sp)
        self.conv_angle_edit.valueChanged.connect(self.update_conv_ang)

        # self.find_model_button.clicked.connect(self.run_ransac)
        # self.export_button.clicked.connect(self.export_image)
        # self.crop_button.clicked.connect(self.crop_image)

        self.statusbar.showMessage('Ready')
        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle('Aberration fitter')
        self.n_rings_label.setText('Num. of rings')
        self.df_label.setText('Defocus [nm]')
        self.A1_amp_label.setText('A1 amplitude [nm]')
        self.A1_phs_label.setText('A1 angle [deg]')
        self.Cs_label.setText('Cs [mm]')
        self.df_spread_label.setText('Defocus spread [nm]')
        self.conv_angle_label.setText('Conv. angle [mrad]')

    def update_rings(self):
        ctf = ctf_calc.calc_ctf_1d_dev(self.img_view.image.width, self.img_view.image.px_dim, self.aberrs)
        ring_positions = ctf_calc.get_pctf_zero_crossings(ctf)
        if len(ring_positions) == 0:
            return
        mid = (const.ccWidgetDim // 2, const.ccWidgetDim // 2)
        tilt = self.A1_phs_edit.value()
        self.rings = [ Thon_ring(mid, ring_positions[i], ring_positions[i], tilt) for i in range(self.n_rings_edit.value()) ]
        self.img_view.update()

    def update_df(self):
        self.aberrs.set_C1(self.df_edit.value() * 1e-9)
        self.update_rings()

    def update_A1(self):
        self.aberrs.set_A1(self.A1_amp_edit.value() * 1e-9, deg2rad(self.A1_phs_edit.value()))  # zmienic na mrad
        self.update_rings()

    def update_Cs(self):
        self.aberrs.set_Cs(self.Cs_edit.value() * 1e-3)
        self.update_rings()

    def update_df_sp(self):
        self.aberrs.set_df_spread(self.df_spread_edit.value() * 1e-9)
        self.update_rings()

    def update_conv_ang(self):
        self.aberrs.set_conv_angle(self.conv_angle_edit.value() * 1e-3)
        self.update_rings()

# -------------------------------------------------------------------

class GraphicsLabel(QtWidgets.QLabel):
    def __init__(self, parent, image=None):
        super(GraphicsLabel, self).__init__(parent)

        self.image = image
        self.scaled_image = ctf_calc.scale_image(self.image.buffer, np.min(self.image.buffer), np.max(self.image.buffer))
        self.image_to_disp = np.copy(self.scaled_image)

        self.view = QtWidgets.QGraphicsView(self)
        self.scene = QtWidgets.QGraphicsScene()
        self.view.setScene(self.scene)

        self.gain = 1.0
        self.bias = 0.0

        self.repaint_pixmap()
        self.view.show()

    def repaint_pixmap(self):
        # self.image.UpdateImageFromBuffer()
        # padded_image = imsup.PadImageBufferToNx512(self.image, np.max(self.image.buffer))
        s_image = ctf_calc.scale_image(self.image_to_disp, 0.0, 255.0)

        q_image = QtGui.QImage(s_image.astype(np.uint8), s_image.shape[1], s_image.shape[0], QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap(q_image)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)

        if len(self.scene.items()) > 0:
            self.scene.removeItem(self.scene.items()[-1])

        self.scene.addPixmap(pixmap)
        self.update()

    def change_gain_and_bias(self):
        self.gain = self.parent().cont_slider.value() * 0.02
        self.bias = self.parent().bright_slider.value() * 1.5 - 75
        print(self.gain, self.bias)
        self.image_to_disp = self.gain * self.scaled_image + self.bias
        self.repaint_pixmap()

    def update(self):
        super(GraphicsLabel, self).update()

        mid_x = const.ccWidgetDim // 2
        mid_y = const.ccWidgetDim // 2

        line_pen = QtGui.QPen(QtCore.Qt.yellow)
        line_pen.setCapStyle(QtCore.Qt.RoundCap)
        line_pen.setWidth(2)

        coeff = const.ccWidgetDim / self.image.width

        for item in self.scene.items()[:-1]:
            self.scene.removeItem(item)

        for r in self.parent().rings:
            a = r.r1 * coeff
            b = r.r2 * coeff
            e_item = QtWidgets.QGraphicsEllipseItem(mid_x - a, mid_y - b, 2 * a, 2 * b)
            e_item.setTransformOriginPoint(mid_x, mid_y)
            e_item.setRotation(r.tilt)
            e_item.setPen(line_pen)
            self.scene.addItem(e_item)

# -------------------------------------------------------------------

def run_aberr_window():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    aberr_win = Ui_MainWindow(app)
    sys.exit(app.exec_())

run_aberr_window()
