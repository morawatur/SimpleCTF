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

    def set_mid(self, mid=(0, 0)):
        self.mid = mid

    def set_r1(self, a):
        self.r1 = a

    def set_r2(self, b):
        self.r2 = b

    def set_tilt(self, tau):
        self.tilt = tau

# ------------------------------------------------------------

def kx_2_px(kx, width, px_dim):
    rec_px_dim = 1.0 / (width * px_dim)
    px = kx / rec_px_dim
    return px

# ------------------------------------------------------------

def create_Thon_ring_from_x0_and_y0(x, y):
    tau = np.arctan2(x, y)
    ring = Thon_ring(mid=(0, 0), tau = tau)
    ring.r1 = np.sqrt(np.abs(x * y * np.cos(2 * tau) / (y * y * np.cos(tau) ** 2 - x * x * np.sin(tau) ** 2)))
    ring.r2 = np.sqrt(np.abs(x * y * np.cos(2 * tau) / (x * x * np.cos(tau) ** 2 - y * y * np.sin(tau) ** 2)))
    # print(x, y, rad2deg(tau))
    print(ring.r1, ring.r2)
    return ring

# ------------------------------------------------------------

def create_Thon_ring_from_pctf_zeros(ctf, n):
    C1 = ctf.abset.get_C1_cf()
    Cs = ctf.abset.get_Cs_cf()
    A1 = ctf.abset.get_A1_cf().real
    # print(C1, Cs, A1)

    if Cs > 0:
        kx2_delta = np.sqrt((C1 + A1) ** 2 + 4 * n * np.pi * Cs)
        ky2_delta = np.sqrt((C1 - A1) ** 2 + 4 * n * np.pi * Cs)

        kx2_12 = [(-A1 - C1 + kx2_delta) / (2.0 * Cs), (-A1 - C1 - kx2_delta) / (2.0 * Cs)]
        ky2_12 = [(A1 - C1 + ky2_delta) / (2.0 * Cs), (A1 - C1 - ky2_delta) / (2.0 * Cs)]

        kx2 = np.max(kx2_12)
        ky2 = np.max(ky2_12)

        kx = np.sqrt(kx2)
        ky = np.sqrt(ky2)
    else:
        kx = np.sqrt((n * np.pi) / (C1 + A1))
        ky = np.sqrt(np.abs((n * np.pi) / (C1 - A1)))

    px_x = int(kx_2_px(kx, ctf.w, ctf.px))
    px_y = int(kx_2_px(ky, ctf.w, ctf.px))
    ring = create_Thon_ring_from_x0_and_y0(px_x, px_y)

    return ring

# ------------------------------------------------------------

def get_Thon_rings(pctf, threshold=0.1, ap=0):

    if ap > 0:
        n = pctf.shape[0]
        c = n // 2
        y, x = np.ogrid[-c:n-c, -c:n-c]
        mask = x * x + y * y > ap * ap
        pctf[mask] = 2

    pctf[abs(pctf) < threshold] = 1
    pctf[pctf != 1] = 0

    return ctf_calc.scale_image(pctf, 0, 1)

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
        self.A1_amp_edit.setProperty('value', 1)
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
        self.df_spread_edit.setProperty('value', 1)
        self.df_spread_edit.setObjectName('df_spread_edit')
        self.verticalLayout.addWidget(self.df_spread_edit)

        self.conv_angle_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.conv_angle_label.setObjectName('conv_angle_label')
        self.verticalLayout.addWidget(self.conv_angle_label)

        self.conv_angle_edit = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.conv_angle_edit.setMinimum(0)
        self.conv_angle_edit.setMaximum(359)
        self.conv_angle_edit.setSingleStep(1)
        self.conv_angle_edit.setProperty('value', 1)
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

        self.update_aberrs()
        ctf_data = ctf_calc.calc_ctf_2d_dev(img.width, img.px_dim, self.aberrs)
        pctf_data = ctf_data.get_ctf_sine()
        thon_rings = get_Thon_rings(pctf_data, threshold=0.1, ap=400)
        pctf_img = imsup.ImageWithBuffer(pctf_data.shape[0], pctf_data.shape[1])
        pctf_img.LoadAmpData(thon_rings)

        self.ctf_view = GraphicsLabel(self, pctf_img)
        self.ctf_view.setGeometry(QtCore.QRect(20, 10, const.ccWidgetDim, const.ccWidgetDim))
        self.ctf_view.setObjectName('ctf_view')
        self.ctf_view.opacity.setOpacity(1.0)

        self.img_view.show()
        self.ctf_view.show()
        self.verticalLayoutWidget.show()

        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 750, 21))
        self.menubar.setObjectName('menubar')
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName('statusbar')
        self.setStatusBar(self.statusbar)

        self.n_rings_edit.valueChanged.connect(self.update_rings)
        self.df_edit.valueChanged.connect(self.update_aberrs)
        self.A1_amp_edit.valueChanged.connect(self.update_aberrs)
        self.A1_phs_edit.valueChanged.connect(self.update_aberrs)
        self.Cs_edit.valueChanged.connect(self.update_aberrs)
        self.df_spread_edit.valueChanged.connect(self.update_aberrs)
        self.conv_angle_edit.valueChanged.connect(self.update_aberrs)

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

    def update_aberrs(self):
        self.aberrs.set_C1(self.df_edit.value() * 1e-9)
        self.aberrs.set_A1(self.A1_amp_edit.value() * 1e-9, deg2rad(self.A1_phs_edit.value()))
        self.aberrs.set_Cs(self.Cs_edit.value() * 1e-3)
        self.aberrs.set_df_spread(self.df_spread_edit.value() * 1e-9)
        self.aberrs.set_conv_angle(self.conv_angle_edit.value() * 1e-3)
        # self.update_rings()

    def update_rings(self):
        self.update_aberrs()
        ctf_data = ctf_calc.calc_ctf_2d_dev(self.img_view.image.width, self.img_view.image.px_dim, self.aberrs)
        pctf_data = ctf_data.get_ctf_sine()
        thon_rings = get_Thon_rings(pctf_data, threshold=0.1, ap=400)
        self.ctf_view.image_to_disp = np.copy(thon_rings)
        self.ctf_view.repaint_pixmap()

    # def update_rings(self):
    #     ctf = ctf_calc.calc_ctf_2d_dev(self.img_view.image.width, self.img_view.image.px_dim, self.aberrs)
    #     self.rings = [ create_Thon_ring_from_pctf_zeros(ctf, i) for i in range(1, self.n_rings_edit.value() + 1) ]
    #     self.img_view.update()

    # def update_df(self):
    #     self.aberrs.set_C1(self.df_edit.value() * 1e-9)
    #     self.update_rings()
    #
    # def update_A1(self):
    #     self.aberrs.set_A1(self.A1_amp_edit.value() * 1e-9, deg2rad(self.A1_phs_edit.value()))  # zmienic na mrad
    #     self.update_rings()
    #
    # def update_Cs(self):
    #     self.aberrs.set_Cs(self.Cs_edit.value() * 1e-3)
    #     self.update_rings()
    #
    # def update_df_sp(self):
    #     self.aberrs.set_df_spread(self.df_spread_edit.value() * 1e-9)
    #     self.update_rings()
    #
    # def update_conv_ang(self):
    #     self.aberrs.set_conv_angle(self.conv_angle_edit.value() * 1e-3)
    #     self.update_rings()

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
        self.view.setStyleSheet('background: transparent')

        self.gain = 1.0
        self.bias = 0.0

        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.view.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1.0)

        self.repaint_pixmap()
        self.view.show()

    def repaint_pixmap(self):
        # self.image.UpdateImageFromBuffer()
        # padded_image = imsup.PadImageBufferToNx512(self.image, np.max(self.image.buffer))
        s_image = ctf_calc.scale_image(self.image_to_disp, 0.0, 255.0)
        q_image = QtGui.QImage(s_image.astype(np.uint8), s_image.shape[1], s_image.shape[0], QtGui.QImage.Format_Indexed8)
        q_image = q_image.convertToFormat(QtGui.QImage.Format_ARGB32)

        # jak to przyspieszyc?
        for x in range(q_image.height()):
            for y in range(q_image.width()):
                color = QtGui.QColor(q_image.pixel(x, y))
                # print(color.black(), color.alpha())
                if color.black() == 255:
                    q_image.setPixel(x, y, QtGui.QColor(0,0,0,0).rgba())

        pixmap = QtGui.QPixmap(q_image)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)

        # if len(self.scene.items()) > 0:
        #     self.scene.removeItem(self.scene.items()[-1])

        for item in self.scene.items():
            self.scene.removeItem(item)

        self.scene.addPixmap(pixmap)
        # self.update()

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
