import math
import re
import sys
from os import path
from functools import partial

import numpy as np
from PyQt5 import QtGui, QtWidgets

import Dm3Reader3 as dm3
import Constants as const
import CrossCorr as cc
import ImageSupport as imsup
import Propagation as prop

# --------------------------------------------------------

class CheckButton(QtWidgets.QPushButton):
    def __init__(self, text, width, height):
        super(CheckButton, self).__init__(text)
        self.defaultStyle = 'background-color:transparent; color:transparent; width:{0}; height:{1}; border:1px solid rgb(0, 0, 0); padding:-1px;'.format(width, height)
        self.clickedStyle = 'background-color:rgba(255, 255, 255, 100); color:transparent; width:{0}; height:{1}; border:1px solid rgb(0, 0, 0); padding:-1px;'.format(width, height)
        self.wasClicked = False
        self.initUI()

    def initUI(self):
        self.setStyleSheet(self.defaultStyle)
        self.clicked.connect(self.handleButton)

    def handleButton(self):
        self.wasClicked = not self.wasClicked
        if self.wasClicked:
            self.setStyleSheet(self.clickedStyle)
        else:
            self.setStyleSheet(self.defaultStyle)

# --------------------------------------------------------

class ButtonGridOnLabel(QtWidgets.QLabel):
    def __init__(self, image, gridDim, parent):
        super(ButtonGridOnLabel, self).__init__(parent)
        self.grid = QtWidgets.QGridLayout()
        self.gridDim = gridDim
        self.image = image               # scaling and other changes will be executed on image buffer (not on image itself)
        self.initUI()

    def initUI(self):
        self.image.ReIm2AmPh()
        self.image.UpdateBuffer()
        self.createPixmap()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(0)
        self.setLayout(self.grid)
        self.createGrid()

    def createPixmap(self):
        qImg = QtGui.QImage(imsup.ScaleImage(self.image.buffer, 0.0, 255.0).astype(np.uint8),
                           self.image.width, self.image.height, QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap(qImg)
        # pixmap.convertFromImage(qImg)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)
        self.setPixmap(pixmap)
        imgNumLabel, defocusLabel = self.parent().accessLabels()
        imgNumLabel.setText('Image {0}'.format(self.image.numInSeries))
        defocusLabel.setText('df = {0:.1e} nm'.format(self.image.defocus * 1e9))

    def changePixmap(self, toNext=True):
        newImage = self.image.next if toNext else self.image.prev
        if newImage is not None:
            newImage.ReIm2AmPh()
            self.image = newImage
            self.createPixmap()

    def createGrid(self):
        if self.grid.count() > 0:
            rowCount = int(math.sqrt(self.grid.count()))
            colCount = rowCount
            old_positions = [(i, j) for i in range(rowCount) for j in range(colCount)]
            for pos in old_positions:
                button = self.grid.itemAtPosition(pos[0], pos[1]).widget()
                button.deleteLater()
                # self.grid.removeWidget(button)
                # button.setParent(None)

        positions = [(i, j) for i in range(self.gridDim) for j in range(self.gridDim)]
        btnWidth = math.ceil(const.ccWidgetDim / self.gridDim)

        for pos in positions:
            button = CheckButton('{0}'.format(pos), btnWidth, btnWidth)
            self.grid.addWidget(button, *pos)
        # print(self.grid.rowCount(), self.grid.columnCount())  # rowCount increases, but does not decrease

    def changeGrid(self, more=True):
        newGridDim = self.gridDim + 1 if more else self.gridDim - 1
        if 0 < newGridDim < 10:
            self.gridDim = newGridDim
            self.createGrid()

    def applyChangesToImage(self, image):
        image.UpdateImageFromBuffer()
        cropCoords = imsup.DetermineCropCoords(image.width, image.height, image.shift)
        self.parent().commonCoords = imsup.GetCommonArea(self.parent().commonCoords, cropCoords)

    def applyChangesToAll(self):
        # self.applyChangesToImage(self.image)
        # tmpNext = self.image
        # tmpPrev = self.image
        # while tmpNext.next is not None:
        #     tmpNext = tmpNext.next
        #     self.applyChangesToImage(tmpNext)
        # while tmpPrev.prev is not None:
        #     tmpPrev = tmpPrev.prev
        #     self.applyChangesToImage(tmpPrev)
        first = imsup.GetFirstImage(self.image)
        imgList = imsup.CreateImageListFromFirstImage(first)
        for img in imgList:
            self.applyChangesToImage(img)
            # img.MoveToCPU()       # ???

        self.createPixmap()
        # super() instead of parent()?
        self.parent().parent().parent().statusBar().showMessage('All changes applied'.format(self.image.numInSeries))
        # self.parent().parent().parent().close()

    def resetImage(self):
        self.image.UpdateBuffer()
        self.image.shift = [0, 0]
        self.image.defocus = 0.0
        self.createPixmap()
        self.parent().parent().parent().statusBar().showMessage('Image no {0} was reset'.format(self.image.numInSeries))

# --------------------------------------------------------

class LineEditWithLabel(QtWidgets.QWidget):
    def __init__(self, parent, labText='df', defaultValue=''):
        super(LineEditWithLabel, self).__init__(parent)
        self.label = QtWidgets.QLabel(labText)
        self.input = QtWidgets.QLineEdit(defaultValue)
        self.initUI()

    def initUI(self):
        # self.label.setFixedWidth(50)
        # self.input.setFixedWidth(50)
        self.setFixedWidth(50)
        self.input.setMaxLength(10)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(self.label)
        vbox.addWidget(self.input)
        self.setLayout(vbox)

# --------------------------------------------------------

class CrossCorrWidget(QtWidgets.QWidget):
    def __init__(self, image, gridDim, parent):
        super(CrossCorrWidget, self).__init__(parent)
        self.imgNumLabel = QtWidgets.QLabel('no data', self)
        self.defocusLabel = QtWidgets.QLabel('no data', self)
        self.btnGrid = ButtonGridOnLabel(image, gridDim, self)
        self.commonCoords = [0, 0, image.height, image.width]
        self.initUI()

    def initUI(self):
        prevButton = QtWidgets.QPushButton(QtGui.QIcon('gui/prev.png'), '', self)
        prevButton.clicked.connect(partial(self.btnGrid.changePixmap, False))

        nextButton = QtWidgets.QPushButton(QtGui.QIcon('gui/next.png'), '', self)
        nextButton.clicked.connect(partial(self.btnGrid.changePixmap, True))

        lessButton = QtWidgets.QPushButton(QtGui.QIcon('gui/less.png'), '', self)
        lessButton.clicked.connect(partial(self.btnGrid.changeGrid, False))

        moreButton = QtWidgets.QPushButton(QtGui.QIcon('gui/more.png'), '', self)
        moreButton.clicked.connect(partial(self.btnGrid.changeGrid, True))

        hbox_tl = QtWidgets.QHBoxLayout()
        hbox_tl.addWidget(prevButton)
        hbox_tl.addWidget(nextButton)

        hbox_ml = QtWidgets.QHBoxLayout()
        hbox_ml.addWidget(lessButton)
        hbox_ml.addWidget(moreButton)

        vbox_switch = QtWidgets.QVBoxLayout()
        vbox_switch.addLayout(hbox_tl)
        vbox_switch.addLayout(hbox_ml)
        vbox_switch.addWidget(self.imgNumLabel)
        vbox_switch.addWidget(self.defocusLabel)

        self.corrAllAtOnceRadioButton = QtWidgets.QRadioButton('Corr. all at once', self)
        self.corrAllAtOnceRadioButton.setChecked(False)

        correlateWithPrevButton = QtWidgets.QPushButton('Corr. with prev.')
        # correlateWithPrevButton.clicked.connect(self.correlateWithPrev)
        correlateWithPrevButton.clicked.connect(self.correlateAll)
        correlateWithSimButton = QtWidgets.QPushButton('Corr. with sim.')
        correlateWithSimButton.clicked.connect(self.correlateWithSim)
        setDefocusButton = QtWidgets.QPushButton('Set defocus')
        setDefocusButton.clicked.connect(self.setDefocus)

        vbox_corr = QtWidgets.QVBoxLayout()
        vbox_corr.addWidget(self.corrAllAtOnceRadioButton)
        vbox_corr.addWidget(correlateWithPrevButton)
        vbox_corr.addWidget(correlateWithSimButton)
        vbox_corr.addWidget(setDefocusButton)

        self.shiftStepEdit = QtWidgets.QLineEdit('5', self)
        self.shiftStepEdit.setFixedWidth(20)
        self.shiftStepEdit.setMaxLength(3)

        upButton = QtWidgets.QPushButton(QtGui.QIcon('gui/up.png'), '', self)
        upButton.clicked.connect(self.movePixmapUp)

        downButton = QtWidgets.QPushButton(QtGui.QIcon('gui/down.png'), '', self)
        downButton.clicked.connect(self.movePixmapDown)

        leftButton = QtWidgets.QPushButton(QtGui.QIcon('gui/left.png'), '', self)
        leftButton.clicked.connect(self.movePixmapLeft)

        rightButton = QtWidgets.QPushButton(QtGui.QIcon('gui/right.png'), '', self)
        rightButton.clicked.connect(self.movePixmapRight)

        hbox_mm = QtWidgets.QHBoxLayout()
        hbox_mm.addWidget(leftButton)
        hbox_mm.addWidget(self.shiftStepEdit)
        hbox_mm.addWidget(rightButton)

        vbox_move = QtWidgets.QVBoxLayout()
        vbox_move.addWidget(upButton)
        vbox_move.addLayout(hbox_mm)
        vbox_move.addWidget(downButton)

        self.dfMinEdit = LineEditWithLabel(self, labText='df min [nm]', defaultValue=str(const.dfStepMin))
        self.dfMaxEdit = LineEditWithLabel(self, labText='df max [nm]', defaultValue=str(const.dfStepMax))
        self.dfStepEdit = LineEditWithLabel(self, labText='step [nm]', defaultValue=str(const.dfStepChange))
        self.dfMinEdit.setFixedWidth(60)
        self.dfMaxEdit.setFixedWidth(60)

        hbox_mr = QtWidgets.QHBoxLayout()
        hbox_mr.addWidget(self.dfMinEdit)
        hbox_mr.addWidget(self.dfMaxEdit)
        hbox_mr.addWidget(self.dfStepEdit)

        applyChangesButton = QtWidgets.QPushButton('Apply changes', self)
        applyChangesButton.clicked.connect(self.btnGrid.applyChangesToAll)

        resetButton = QtWidgets.QPushButton('Reset image', self)
        resetButton.clicked.connect(self.btnGrid.resetImage)

        vbox_right = QtWidgets.QVBoxLayout()
        vbox_right.addLayout(hbox_mr)
        vbox_right.addWidget(applyChangesButton)
        vbox_right.addWidget(resetButton)

        hbox_main = QtWidgets.QHBoxLayout()
        hbox_main.addLayout(vbox_switch)
        hbox_main.addLayout(vbox_corr)
        hbox_main.addLayout(vbox_move)
        hbox_main.addLayout(vbox_right)

        vbox_main = QtWidgets.QVBoxLayout()
        vbox_main.addWidget(self.btnGrid)
        vbox_main.addLayout(hbox_main)
        self.setLayout(vbox_main)

    def movePixmapUp(self):
        cc.MoveImageUp(self.btnGrid.image, int(self.shiftStepEdit.text()))
        self.btnGrid.createPixmap()

    def movePixmapDown(self):
        cc.MoveImageDown(self.btnGrid.image, int(self.shiftStepEdit.text()))
        self.btnGrid.createPixmap()

    def movePixmapLeft(self):
        cc.MoveImageLeft(self.btnGrid.image, int(self.shiftStepEdit.text()))
        self.btnGrid.createPixmap()

    def movePixmapRight(self):
        cc.MoveImageRight(self.btnGrid.image, int(self.shiftStepEdit.text()))
        self.btnGrid.createPixmap()

    def getFragCoords(self):
        fragCoords = []
        for pos in range(self.btnGrid.grid.count()):
            btn = self.btnGrid.grid.itemAt(pos).widget()
            if btn.wasClicked:
                btn.handleButton()
                values = re.search('([0-9]), ([0-9])', btn.text())
                fragPos = (int(values.group(1)), int(values.group(2)))
                fragCoords.append(fragPos)
        return fragCoords

    def correlateAll(self):
        fragCoords = self.getFragCoords()
        if self.corrAllAtOnceRadioButton.isChecked():
            first = self.btnGrid.image
            while first.prev is not None:
                first = first.prev
            while first.next is not None:
                self.btnGrid.image = first.next
                self.correlateWithPrev(fragCoords)
                first = first.next
        else:
            self.correlateWithPrev(fragCoords)

    # move this method to ButtonGridOnLabel?
    def correlateWithPrev(self, fragCoords):
        image = self.btnGrid.image
        if image.prev is None:
            self.parent().parent().statusBar().showMessage("Can't correlate. This is the reference image.")
            return
        self.parent().parent().statusBar().showMessage('Correlating...')
        self.correlateWithImage(image.prev, fragCoords, True)
        self.parent().parent().statusBar().showMessage('Done! Image no {0} was shifted to image no {1}'.format(image.numInSeries, image.prev.numInSeries))

    # self.btnGrid.image powinien byc atrybutem CcWidget, a nie ButtonGridOnLabel
    # (ten drugi moglby sie do niego odnosic przez parent)
    def correlateWithSim(self):
        fragCoords = self.getFragCoords()
        exitWave = self.parent().parent().getIwfrWidgetRef().exitWave
        imageSim = SimulateImageForDefocus(exitWave, self.btnGrid.image.defocus)
        roiCoords = imsup.DetermineEqualCropCoords(self.btnGrid.image.width, imageSim.width)
        roiImage = imsup.CropImageROICoords(self.btnGrid.image, roiCoords)
        self.btnGrid.image = imsup.CreateImageWithBufferFromImage(roiImage)
        self.btnGrid.image.MoveToCPU()
        self.correlateWithImage(imageSim, fragCoords, False)

    def correlateWithImage(self, imageToCorr, fragCoords, wat):
        image = self.btnGrid.image
        mcfBest = cc.MaximizeMCFCore(imageToCorr, image, self.btnGrid.gridDim, fragCoords,
                                     float(self.dfMinEdit.input.text()),
                                     float(self.dfMaxEdit.input.text()),
                                     float(self.dfStepEdit.input.text()))
        shift = cc.GetShift(mcfBest)
        image.shift = (image.prev.shift if wat else [0, 0])
        cc.ShiftImageAmpBuffer(image, shift)
        self.btnGrid.image.defocus = mcfBest.defocus
        self.btnGrid.createPixmap()
        ccfPath = const.ccfResultsDir + const.ccfName + str(image.numInSeries) + '.png'
        imsup.SaveAmpImage(mcfBest, ccfPath)

    def setDefocus(self):
        dfnm = float(self.dfMinEdit.input.text())
        self.btnGrid.image.defocus = dfnm * 1e-9
        self.defocusLabel.setText('df = {0:.1e} nm'.format(dfnm))

    def accessLabels(self):
        return self.imgNumLabel, self.defocusLabel

# --------------------------------------------------------

# odroznic parent() (klasa dziedziczona) od parent, czyli instancji, ktorej atrybutem jest IwfrWidget
class IwfrWidget(QtWidgets.QWidget):
    def __init__(self, image, parent):
        super(IwfrWidget, self).__init__(parent)
        self.display = QtWidgets.QLabel()
        self.imageSim = image
        self.exitWave = image
        # self.owner = parent
        # self.commonCoords = commonCoords
        self.createPixmap()
        self.initUI()

    def initUI(self):
        n_images = count_linked_images(self.imageSim)
        self.numOfFirstEdit = LineEditWithLabel(self, labText='First to EWR', defaultValue='1')
        self.numInFocusEdit = LineEditWithLabel(self, labText='In focus', defaultValue=str(n_images // 2))
        self.numOfImagesEdit = LineEditWithLabel(self, labText='Num. of images', defaultValue=str(n_images))
        self.numOfItersEdit = LineEditWithLabel(self, labText='Num. of iterations', defaultValue=str(const.nIterations))
        self.dfToSimEdit = LineEditWithLabel(self, labText='Defocus [nm]', defaultValue='0.0')

        self.numOfFirstEdit.setFixedWidth(100)
        self.numInFocusEdit.setFixedWidth(100)
        self.numOfImagesEdit.setFixedWidth(100)
        self.numOfItersEdit.setFixedWidth(100)
        self.dfToSimEdit.setFixedWidth(100)

        self.amplitudeRadioButton = QtWidgets.QRadioButton('Amplitude', self)
        self.amplitudeRadioButton.setChecked(True)
        self.phaseRadioButton = QtWidgets.QRadioButton('Phase', self)
        self.fftRadioButton = QtWidgets.QRadioButton('FFT', self)

        self.phase_unwrap_rbutton = QtWidgets.QRadioButton('Phase unwrapped', self)
        # self.phase_unwrap_rbutton.setChecked(True)
        self.phase_wrap_rbutton = QtWidgets.QRadioButton('Phase wrapped', self)
        self.phase_unwrap_rbutton.setDisabled(True)
        self.phase_wrap_rbutton.setDisabled(True)

        simulateButton = QtWidgets.QPushButton('Simulate image', self)
        simulateButton.clicked.connect(self.simulateImage)

        runEwrButton = QtWidgets.QPushButton('Run EWR', self)
        runEwrButton.clicked.connect(self.runEwr)

        self.export_path_edit = LineEditWithLabel(self, labText='Export path', defaultValue='img')
        self.export_path_edit.setFixedWidth(100)
        export_button = QtWidgets.QPushButton('Export', self)
        export_button.clicked.connect(self.export)

        vbox_edit1 = QtWidgets.QVBoxLayout()
        vbox_edit1.addWidget(self.numOfFirstEdit)
        vbox_edit1.addWidget(self.numInFocusEdit)

        vbox_edit2 = QtWidgets.QVBoxLayout()
        vbox_edit2.addWidget(self.numOfImagesEdit)
        vbox_edit2.addWidget(self.numOfItersEdit)

        disp_radio_group = QtWidgets.QButtonGroup(self)
        disp_radio_group.addButton(self.amplitudeRadioButton)
        disp_radio_group.addButton(self.phaseRadioButton)
        disp_radio_group.addButton(self.fftRadioButton)

        vbox_disp_radio = QtWidgets.QVBoxLayout()
        vbox_disp_radio.addWidget(self.amplitudeRadioButton)
        vbox_disp_radio.addWidget(self.phaseRadioButton)
        vbox_disp_radio.addWidget(self.fftRadioButton)

        phase_radio_group = QtWidgets.QButtonGroup(self)
        phase_radio_group.addButton(self.phase_unwrap_rbutton)
        phase_radio_group.addButton(self.phase_wrap_rbutton)

        vbox_phase_radio = QtWidgets.QVBoxLayout()
        vbox_phase_radio.addWidget(self.phase_unwrap_rbutton)
        vbox_phase_radio.addWidget(self.phase_wrap_rbutton)

        vbox_sim = QtWidgets.QVBoxLayout()
        vbox_sim.addWidget(self.dfToSimEdit)
        vbox_sim.addWidget(simulateButton)

        vbox_export = QtWidgets.QVBoxLayout()
        vbox_export.addWidget(self.export_path_edit)
        vbox_export.addWidget(export_button)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addLayout(vbox_edit1)
        hbox.addLayout(vbox_edit2)
        hbox.addLayout(vbox_disp_radio)
        hbox.addLayout(vbox_phase_radio)
        hbox.addLayout(vbox_sim)
        hbox.addLayout(vbox_export)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(runEwrButton)

        vbox_main = QtWidgets.QVBoxLayout()
        vbox_main.addWidget(self.display)
        vbox_main.addLayout(vbox)

        self.setLayout(vbox_main)

        self.amplitudeRadioButton.toggled.connect(self.displayAmplitude)
        self.phaseRadioButton.toggled.connect(self.displayPhase)
        self.fftRadioButton.toggled.connect(self.displayFFT)

        self.phase_unwrap_rbutton.toggled.connect(self.unwrap_phase)
        self.phase_wrap_rbutton.toggled.connect(self.wrap_phase)

    def displayAmplitude(self):
        self.imageSim.buffer = np.copy(self.imageSim.amPh.am)
        # self.imageSim.UpdateBuffer()
        self.createPixmap()
        self.phase_unwrap_rbutton.setDisabled(True)
        self.phase_wrap_rbutton.setDisabled(True)

    def displayPhase(self):
        # self.imageSim.buffer = np.copy(self.imageSim.amPh.ph)
        self.imageSim.UpdateBufferFromPhase()
        self.createPixmap()
        # self.imageSim.UpdateBuffer()
        self.phase_unwrap_rbutton.setEnabled(True)
        self.phase_unwrap_rbutton.setChecked(True)
        self.phase_wrap_rbutton.setEnabled(True)

    def displayFFT(self):
        fft = cc.FFT(self.imageSim)
        diff = cc.FFT2Diff(fft)
        diff.ReIm2AmPh()
        diff.MoveToCPU()
        diffToDisp = np.log10(diff.amPh.am)
        self.imageSim.buffer = imsup.ScaleImage(diffToDisp, 0.0, 255.0)
        self.createPixmap()
        self.imageSim.UpdateBuffer()
        self.phase_unwrap_rbutton.setDisabled(True)
        self.phase_wrap_rbutton.setDisabled(True)

    def unwrap_phase(self):
        self.imageSim.UpdateBufferFromPhase()
        self.createPixmap()
        self.imageSim.UpdateBuffer()

    def wrap_phase(self):
        self.imageSim.buffer = self.imageSim.amPh.ph % (2 * np.pi)
        self.createPixmap()

    # why exported absolute-phase image looks like normal phase image?
    def export(self):
        export_path = self.export_path_edit.input.text() + '.png'
        img_to_save = imsup.CopyImage(self.imageSim)
        img_to_save.UpdateImageFromBuffer()
        imsup.SaveAmpImage(img_to_save, export_path)
        self.parent().parent().statusBar().showMessage('Image saved as "{0}.png"'.format(self.export_path_edit.input.text()))

    def createPixmap(self):
        paddedExitWave = imsup.PadImageBufferToNx512(self.imageSim, np.max(self.imageSim.buffer))

        qImg = QtGui.QImage(imsup.ScaleImage(paddedExitWave.buffer, 0.0, 255.0).astype(np.uint8),
                            paddedExitWave.width, paddedExitWave.height, QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap(qImg)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)
        self.display.setPixmap(pixmap)

    def changePixmap(self, toNext=True):
        newImage = self.image.next if toNext else self.image.prev
        if newImage is not None:
            newImage.ReIm2AmPh()
            self.exitWave = newImage
            self.createPixmap()

    def runEwr(self):
        currentImage = self.parent().parent().getCcWidgetRef().btnGrid.image
        firstImage = imsup.GetFirstImage(currentImage)
        imgList = imsup.CreateImageListFromFirstImage(firstImage)
        idxInFocus = int(self.numInFocusEdit.input.text()) - 1
        cc.DetermineAbsoluteDefocus(imgList, idxInFocus)

        squareCoords = imsup.MakeSquareCoords(self.parent().parent().getCcWidgetRef().commonCoords)
        # print(squareCoords)
        imgListAll = imsup.CreateImageListFromFirstImage(self.imageSim)
        firstIdx = int(self.numOfFirstEdit.input.text()) - 1
        howMany = int(self.numOfImagesEdit.input.text())
        imgListToEWR = imsup.CreateImageListFromImage(imgListAll[firstIdx], howMany)

        for img, idx in zip(imgListToEWR, range(len(imgListToEWR))):
            print('df = {0:.1f} nm'.format(img.defocus * 1e9))
            imgListToEWR[idx] = imsup.CropImageROICoords(img, squareCoords)
            cropPath = const.cropResultsDir + const.cropName + str(idx + 1) + '.png'
            imsup.SaveAmpImage(imgListToEWR[idx], cropPath)
            imgListToEWR[idx].MoveToCPU()       # !!!

        # imgListToEWR = uw.UnwarpImageList(imgListToEWR, const.nDivForUnwarp)      # unwarp off
        exitWave = prop.PerformIWFR(imgListToEWR, int(self.numOfItersEdit.input.text()))
        exitWave.ChangeComplexRepr(imsup.Image.cmp['CAP'])  # !!!
        exitWave.MoveToCPU()                                # !!!
        # exitWave.amPh.ph = np.abs(exitWave.amPh.ph)         # !!!
        self.exitWave = imsup.CreateImageWithBufferFromImage(exitWave)
        self.imageSim = imsup.CreateImageWithBufferFromImage(self.exitWave)         # copy self.exitWave to self.imageSim
        # self.imageSim.MoveToCPU()
        self.createPixmap()

        self.imageSim.UpdateBufferFromPhase()       # !!!

    def simulateImage(self):
        dfProp = float(self.dfToSimEdit.input.text()) * 1e-9
        imageSim = prop.PropagateBackToDefocus(self.exitWave, dfProp)
        self.imageSim = imsup.CreateImageWithBufferFromImage(imageSim)
        self.imageSim.UpdateBuffer()
        self.imageSim.MoveToCPU()
        self.amplitudeRadioButton.setChecked(True)
        self.createPixmap()

    def accessInputs(self):
        return self.numInFocusEdit

# --------------------------------------------------------

class EwrWindow(QtWidgets.QMainWindow):
    def __init__(self, gridDim):
        super(EwrWindow, self).__init__(None)
        self.centralWidget = QtWidgets.QWidget(self)
        fileDialog = QtWidgets.QFileDialog()
        imagePath = fileDialog.getOpenFileName()[0]
        firstImage = LoadImageSeriesFromFirstFile(imagePath)
        self.ccWidget = CrossCorrWidget(firstImage, gridDim, self)
        self.iwfrWidget = IwfrWidget(firstImage, self)
        self.initUI()

    def initUI(self):
        self.statusBar().showMessage('Ready')

        hbox_main = QtWidgets.QHBoxLayout()
        hbox_main.addWidget(self.ccWidget)
        hbox_main.addWidget(self.iwfrWidget)
        self.centralWidget.setLayout(hbox_main)
        self.setCentralWidget(self.centralWidget)

        self.move(300, 300)
        self.setWindowTitle('Exit wave reconstruction window')
        self.setWindowIcon(QtGui.QIcon('gui/world.png'))
        self.show()
        self.setFixedSize(self.width(), self.height())     # disable window resizing

    def getCcWidgetRef(self):
        return self.ccWidget

    def getIwfrWidgetRef(self):
        return self.iwfrWidget

# --------------------------------------------------------

def LoadImageSeriesFromFirstFile(imgPath):
    imgList = imsup.ImageList()
    imgNumMatch = re.search('([0-9]+).dm3', imgPath)
    imgNumText = imgNumMatch.group(1)
    imgNum = int(imgNumText)

    while path.isfile(imgPath):
        print('Reading file "' + imgPath + '"')
        imgData, pxDims = dm3.ReadDm3File(imgPath)
        imsup.Image.px_dim_default = pxDims[0]
        imgData = np.abs(imgData)
        img = imsup.ImageWithBuffer(imgData.shape[0], imgData.shape[1], imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'],
                                    num=imgNum, px_dim_sz=pxDims[0])
        img.LoadAmpData(np.sqrt(imgData).astype(np.float32))
        imgList.append(img)

        imgNum += 1
        imgNumTextNew = imgNumText.replace(str(imgNum-1), str(imgNum))
        if imgNum == 10:
            imgNumTextNew = imgNumTextNew[1:]
        imgPath = RReplace(imgPath, imgNumText, imgNumTextNew, 1)
        imgNumText = imgNumTextNew

    imgList.UpdateLinks()
    return imgList[0]

# --------------------------------------------------------

def SimulateImageForDefocus(exitWave, dfProp):
    imageSim = prop.PropagateBackToDefocus(exitWave, dfProp)
    imageSim = imsup.CreateImageWithBufferFromImage(imageSim)
    imageSim.UpdateBuffer()
    imageSim.MoveToCPU()
    return imageSim

# --------------------------------------------------------

def count_linked_images(img):
    n_images = 0
    while img is not None:
        n_images += 1
        img = img.next
    return n_images

# --------------------------------------------------------

def RReplace(text, old, new, occurence):
    rest = text.rsplit(old, occurence)
    return new.join(rest)

# --------------------------------------------------------

def RunEwrWindow(gridDim):
    app = QtWidgets.QApplication(sys.argv)
    ccWindow = EwrWindow(gridDim)
    sys.exit(app.exec_())

# do zrobienia:
# zmienna image powinna byc atrybutem klasy EwrWindow; w ten sposob oba widgety mialyby do niego rowny dostep (?)

# mozliwosc zoomowania wynikow
# polaczenie korelacji z rekonstrukcja
# po kazdym etapie rekonstrukcji/korelacji zapisywanie pliku log na temat danego etapu
# (liczba obrazow, in_focus, wielkosci obrazow itd.)

# problem zauwazony 30.01.2017:
# jezeli numer pierwszego obrazu do rekonstrukcji jest > 1, to cos zlego dzieje sie z wyznaczaniem defocusow zsunietych obrazow
# (przyklad: df1 = -12, df2 = -6, df3 = -2, df4 = 0, df5 = 0, ...)