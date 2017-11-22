import Constants as const
import Dm3Reader3 as dm3
import ImageSupport as imsup
import ctf_calc
import aberrations as ab
import simulation as sim

import numpy as np
import copy
# from functools import partial
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

def get_Thon_rings(pctf, ap=0, threshold=0.1):

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
        self.resize(750, 640)
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


        # ----- Image view -----

        self.img_view = GraphicsLabel(self, img)
        self.img_view.setGeometry(QtCore.QRect(20, 10, const.ccWidgetDim, const.ccWidgetDim))

        # ----- Horizontal layout 1 (navigation) -----

        self.nav_hbox_widget = QtWidgets.QWidget(self.centralwidget)
        self.nav_hbox_widget.setGeometry(QtCore.QRect(20, 520, 512, 50))

        self.prev_button = QtWidgets.QPushButton(QtGui.QIcon('gui/prev.png'), '', self.nav_hbox_widget)
        # self.prev_button.clicked.connect(self.prev_image)

        self.next_button = QtWidgets.QPushButton(QtGui.QIcon('gui/next.png'), '', self.nav_hbox_widget)
        # self.next_button.clicked.connect(self.next_image)

        self.nav_hbox = QtWidgets.QHBoxLayout(self.nav_hbox_widget)
        self.nav_hbox.addWidget(self.prev_button)
        self.nav_hbox.addWidget(self.next_button)

        # ----- Vertical layout 1 (aberrations) -----

        self.aber_vbox_widget = QtWidgets.QWidget(self.centralwidget)
        self.aber_vbox_widget.setGeometry(QtCore.QRect(560, 10, 160, 600))
        self.aber_vbox = QtWidgets.QVBoxLayout(self.aber_vbox_widget)

        self.df_label = QtWidgets.QLabel('Defocus [nm]', self.aber_vbox_widget)
        self.df_label.setEnabled(True)
        self.df_label.setObjectName('df_label')
        self.aber_vbox.addWidget(self.df_label)

        self.df_input = QtWidgets.QLineEdit(self.aber_vbox_widget)
        self.df_input.setText(str(40.0))
        self.df_input.setObjectName('df_input')
        self.aber_vbox.addWidget(self.df_input)

        self.A1_amp_label = QtWidgets.QLabel('A1 amplitude [nm]', self.aber_vbox_widget)
        self.A1_amp_label.setObjectName('A1_amp_label')
        self.aber_vbox.addWidget(self.A1_amp_label)

        self.A1_amp_input = QtWidgets.QLineEdit(self.aber_vbox_widget)
        self.A1_amp_input.setText(str(0.0))
        self.A1_amp_input.setObjectName('A1_amp_input')
        self.aber_vbox.addWidget(self.A1_amp_input)

        self.A1_phs_label = QtWidgets.QLabel('A1 angle [deg]', self.aber_vbox_widget)
        self.A1_phs_label.setObjectName('A1_phs_label')
        self.aber_vbox.addWidget(self.A1_phs_label)

        self.A1_phs_input = QtWidgets.QLineEdit(self.aber_vbox_widget)
        self.A1_phs_input.setText(str(0.0))
        self.A1_phs_input.setObjectName('A1_phs_input')
        self.aber_vbox.addWidget(self.A1_phs_input)

        self.Cs_label = QtWidgets.QLabel('Cs [mm]', self.aber_vbox_widget)
        self.Cs_label.setObjectName('Cs_label')
        self.aber_vbox.addWidget(self.Cs_label)

        self.Cs_input = QtWidgets.QLineEdit(self.aber_vbox_widget)
        self.Cs_input.setText(str(0.0))
        self.Cs_input.setObjectName('Cs_input')
        self.aber_vbox.addWidget(self.Cs_input)

        self.df_spread_label = QtWidgets.QLabel('Defocus spread [nm]', self.aber_vbox_widget)
        self.df_spread_label.setObjectName('df_spread_label')
        self.aber_vbox.addWidget(self.df_spread_label)

        self.df_spread_input = QtWidgets.QLineEdit(self.aber_vbox_widget)
        self.df_spread_input.setText(str(0.0))
        self.df_spread_input.setObjectName('df_spread_input')
        self.aber_vbox.addWidget(self.df_spread_input)

        self.conv_angle_label = QtWidgets.QLabel('Conv. angle [mrad]', self.aber_vbox_widget)
        self.conv_angle_label.setObjectName('conv_angle_label')
        self.aber_vbox.addWidget(self.conv_angle_label)

        self.conv_angle_input = QtWidgets.QLineEdit(self.aber_vbox_widget)
        self.conv_angle_input.setText(str(0.0))
        self.conv_angle_input.setObjectName('conv_angle_input')
        self.aber_vbox.addWidget(self.conv_angle_input)

        self.aperture_label = QtWidgets.QLabel('Aperture [px]', self.aber_vbox_widget)
        self.aperture_label.setObjectName('aperture_label')
        self.aber_vbox.addWidget(self.aperture_label)

        self.aperture_input = QtWidgets.QLineEdit(self.aber_vbox_widget)
        self.aperture_input.setText(str(0))
        self.aber_vbox.addWidget(self.aperture_input)

        self.threshold_label = QtWidgets.QLabel('Ring width [au]', self.aber_vbox_widget)
        self.threshold_label.setObjectName('threshold_label')
        self.aber_vbox.addWidget(self.threshold_label)

        self.threshold_input = QtWidgets.QLineEdit(self.aber_vbox_widget)
        self.threshold_input.setText(str(0.01))
        self.aber_vbox.addWidget(self.threshold_input)

        self.aber_vbox.addStretch(1)

        self.update_button = QtWidgets.QPushButton('Update', self.aber_vbox_widget)
        self.update_button.clicked.connect(self.update_rings)
        self.aber_vbox.addWidget(self.update_button)

        self.bright_label = QtWidgets.QLabel('Brightness', self.aber_vbox_widget)
        self.cont_label = QtWidgets.QLabel('Contrast', self.aber_vbox_widget)
        self.gamma_label = QtWidgets.QLabel('Gamma', self.aber_vbox_widget)

        self.bright_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bright_slider.setFixedHeight(14)
        self.bright_slider.setRange(0.0, 100.0)
        self.bright_slider.setValue(50.0)

        self.cont_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.cont_slider.setFixedHeight(14)
        self.cont_slider.setRange(1.0, 100.0)
        self.cont_slider.setValue(50.0)

        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma_slider.setFixedHeight(14)
        self.gamma_slider.setRange(50.0, 150.0)
        self.gamma_slider.setValue(100.0)

        self.bright_slider.valueChanged.connect(self.img_view.correct_image)
        self.cont_slider.valueChanged.connect(self.img_view.correct_image)
        self.gamma_slider.valueChanged.connect(self.img_view.correct_image)

        self.aber_vbox.addWidget(self.bright_label)
        self.aber_vbox.addWidget(self.bright_slider)
        self.aber_vbox.addWidget(self.cont_label)
        self.aber_vbox.addWidget(self.cont_slider)
        self.aber_vbox.addWidget(self.gamma_label)
        self.aber_vbox.addWidget(self.gamma_slider)

        log_hbox = QtWidgets.QHBoxLayout(self.aber_vbox_widget)
        self.log_label = QtWidgets.QLabel('Log.', self.aber_vbox_widget)
        self.log_input = QtWidgets.QLineEdit(self.aber_vbox_widget)
        self.log_input.setText(str(-1))
        self.log_button = QtWidgets.QPushButton('OK', self.aber_vbox_widget)
        self.log_button.clicked.connect(self.img_view.set_log_base)
        log_hbox.addWidget(self.log_label)
        log_hbox.addWidget(self.log_input)
        log_hbox.addWidget(self.log_button)
        self.aber_vbox.addLayout(log_hbox)

        # ----- CTF overlay -----

        self.update_aberrs()
        ctf_data = ctf_calc.calc_ctf_2d_dev(img.width, img.px_dim, self.aberrs)
        pctf_data = ctf_data.get_ctf_sine()
        thon_rings = get_Thon_rings(pctf_data, ap=int(self.aperture_input.text()), threshold=float(self.threshold_input.text()))
        pctf_img = imsup.ImageWithBuffer(pctf_data.shape[0], pctf_data.shape[1])
        pctf_img.LoadAmpData(thon_rings)

        self.ctf_view = GraphicsLabel(self, pctf_img, mzt=True)
        # self.ctf_view.setGeometry(QtCore.QRect(20, 10, const.ccWidgetDim, const.ccWidgetDim))
        self.ctf_view.setGeometry(self.img_view.geometry())
        self.ctf_view.opacity.setOpacity(0.7)

        self.img_view.show()
        self.ctf_view.show()
        self.nav_hbox_widget.show()
        self.aber_vbox_widget.show()

        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 750, 21))
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)

        self.statusbar.showMessage('Ready')
        self.setWindowTitle('Aberration fitter')
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()

    def update_aberrs(self):
        self.aberrs.set_C1(float(self.df_input.text()) * 1e-9)
        self.aberrs.set_A1(float(self.A1_amp_input.text()) * 1e-9, deg2rad(float(self.A1_phs_input.text())))
        self.aberrs.set_Cs(float(self.Cs_input.text()) * 1e-3)
        self.aberrs.set_df_spread(float(self.df_spread_input.text()) * 1e-9)
        self.aberrs.set_conv_angle(float(self.conv_angle_input.text()) * 1e-3)

    def update_rings(self):
        self.update_aberrs()
        ctf_data = ctf_calc.calc_ctf_2d_dev(self.img_view.image.width, self.img_view.image.px_dim, self.aberrs)
        pctf_data = ctf_data.get_ctf_sine()
        thon_rings = get_Thon_rings(pctf_data, ap=int(self.aperture_input.text()), threshold=float(self.threshold_input.text()))
        self.ctf_view.image_to_disp = np.copy(thon_rings)
        self.ctf_view.repaint_pixmap()

    # def update_rings(self):
    #     ctf = ctf_calc.calc_ctf_2d_dev(self.img_view.image.width, self.img_view.image.px_dim, self.aberrs)
    #     self.rings = [ create_Thon_ring_from_pctf_zeros(ctf, i) for i in range(1, self.n_rings_edit.value() + 1) ]
    #     self.img_view.update()

# -------------------------------------------------------------------

class GraphicsLabel(QtWidgets.QLabel):
    def __init__(self, parent, image=None, mzt=False):
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
        self.gamma = 1.0
        self.log_base = -1

        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.view.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1.0)

        self.make_zero_transparent = mzt

        self.repaint_pixmap()
        self.view.show()

    def repaint_pixmap(self):
        # self.image.UpdateImageFromBuffer()
        # padded_image = imsup.PadImageBufferToNx512(self.image, np.max(self.image.buffer))

        if self.make_zero_transparent:
            s_image = ctf_calc.scale_image(self.image_to_disp, 0.0, 255.0).astype(np.uint32)
            alphas = np.ones(s_image.shape, dtype=np.uint32)
            alphas *= s_image
            s_image_argb32 = (alphas << 24 | s_image[:, :] << 16 | s_image[:, :] << 8 | s_image[:, :])
            q_image = QtGui.QImage(s_image_argb32, s_image_argb32.shape[1], s_image_argb32.shape[0], QtGui.QImage.Format_ARGB32)
        else:
            s_image = ctf_calc.scale_image(self.image_to_disp, 0.0, 255.0)
            q_image = QtGui.QImage(s_image.astype(np.uint8), s_image.shape[1], s_image.shape[0], QtGui.QImage.Format_Indexed8)

        pixmap = QtGui.QPixmap(q_image)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)

        # if len(self.scene.items()) > 0:
        #     self.scene.removeItem(self.scene.items()[-1])

        for item in self.scene.items():
            self.scene.removeItem(item)

        self.scene.addPixmap(pixmap)
        # self.update()

    def set_log_base(self):
        self.log_base = float(self.parent().log_input.text())
        self.correct_image()

    def correct_image(self):
        self.gain = int(self.parent().cont_slider.value()) * 0.02
        self.bias = int(self.parent().bright_slider.value()) * 1.5 - 75
        self.gamma = int(self.parent().gamma_slider.value()) * 0.01
        # print(self.gain, self.bias, self.gamma, self.log_base)

        if self.log_base < 0:
            self.image_to_disp = self.gain * (self.scaled_image ** self.gamma) + self.bias
        else:
            img_to_log = np.copy(self.scaled_image)
            img_to_log[img_to_log < 1e-7] = 1
            log_scaled_image = np.log(img_to_log) / np.log(self.log_base)
            log_scaled_image = ctf_calc.scale_image(log_scaled_image, np.min(log_scaled_image), np.max(log_scaled_image))
            self.image_to_disp = self.gain * (log_scaled_image ** self.gamma) + self.bias

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
