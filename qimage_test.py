import numpy as np
from PyQt5 import QtGui

a = np.array([[[10, 20, 30, 0]]]).astype(np.uint32)
print(a.shape)
print(a[:,:,0])
print(a[:,:,1])
print(a[:,:,2])
print(a[:,:,3])
b = (0 << 24 | a[:,:,0] << 16 | a[:,:,1] << 8 | a[:,:,2])

image = QtGui.QImage(b, 1, 1, QtGui.QImage.Format_ARGB32)
print(image.hasAlphaChannel())
imagee = image.convertToFormat(QtGui.QImage.Format_ARGB32)
color = QtGui.QColor(image.pixel(0, 0))
print(color.alpha(), color.red(), color.green(), color.blue())