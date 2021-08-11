import time
import numba
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple, Union, TYPE_CHECKING, Set, List

if TYPE_CHECKING:
    from HSIEvaluator import MainWindow


class GraphView(QtWidgets.QGraphicsView):
    SelectionChanged: QtCore.pyqtSignal = QtCore.pyqtSignal()
    NewSelection: QtCore.pyqtSignal = QtCore.pyqtSignal(str, set)

    def __init__(self):
        super(GraphView, self).__init__()
        scene = self._setUpScene()
        self.setMinimumSize(500, 500)
        self._mainWin: Union[None, 'MainWindow'] = None
        self._origCube: Union[None, np.ndarray] = None
        self._item = QtWidgets.QGraphicsPixmapItem()
        self._selectionOverlay: SelectionOverlay = SelectionOverlay()
        self._classOverlay: ClassificationOverlay = ClassificationOverlay()
        scene.addItem(self._item)
        scene.addItem(self._selectionOverlay)
        scene.addItem(self._classOverlay)
        self._startDrag = None
        self._selecting: bool = False

        self.setMouseTracking(True)

    def setMainWindowReference(self, mainWinRef: 'MainWindow') -> None:
        self._mainWin = mainWinRef

    def setCube(self, cube: np.ndarray) -> None:
        self._origCube = cube
        img = cube2RGB(cube)
        self._selectionOverlay.initOverlay(img.shape)
        self._item.setPixmap(npy2Pixmap(img))

    def getCurrentViewBounds(self) -> QtCore.QRectF:
        return self.mapToScene(self.rect()).boundingRect()

    def setClassOverlay(self, img: np.ndarray) -> None:
        self._classOverlay.updateImage(img)

    def updateImage(self, maxBrightness: float, newZero: int, newContrast: float) -> None:
        """
        Updating the previewed image with new zero value and contrast factor
        :param maxBrightness: Highest brightness to clip to
        :param newZero: integer value of the new zero value
        :param newContrast: float factor for contrast adjustment (1.0 = unchanged)
        :return:
        """
        newImg: np.ndarray = cube2RGB(self._origCube, maxBrightness)
        if newZero != 0:
            newImg = newImg.astype(np.float)
            newImg = np.clip(newImg + newZero, 0, 255)
            newImg = newImg.astype(np.uint8)

        if newContrast != 1.0:
            img: Image = Image.fromarray(newImg)
            contrastObj = ImageEnhance.Contrast(img)
            newImg = np.array(contrastObj.enhance(newContrast))

        self._item.setPixmap(npy2Pixmap(newImg))
        self.scene().update()

    def updateClassImage(self, classImage: np.ndarray) -> None:
        self._classOverlay.updateImage(classImage)

    def updateClassImgTransp(self, newAlpha: float) -> None:
        self._classOverlay.setAlpha(np.clip(newAlpha, 0.0, 1.0))
        self.scene().update()

    def showClassImage(self) -> None:
        self._classOverlay.show()
        self._selectionOverlay.hide()

    def hideClassImage(self) -> None:
        self._classOverlay.hide()
        self._selectionOverlay.show()

    def setSelectionPixelsToColor(self, px: Tuple[np.ndarray, np.ndarray], color: Tuple[int, int, int]) -> None:
        self._selectionOverlay.setPixelColors(px, color)

    def getPixelsOfColor(self, rgb: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param rgb:
        :return: Tuple[np.array(y-indices), np.array(x-indices)]
        """
        return self._selectionOverlay.getPixelsOfColor(rgb)

    def removeColor(self, rgb: Tuple[int, int, int]) -> None:
        self._selectionOverlay.removeColor(rgb)

    @QtCore.pyqtSlot(str)
    def removeColorOfClass(self, className: str) -> None:
        rgb: Tuple[int, int, int] = self._mainWin.getColorOfClass(className)
        self.removeColor(rgb)
        self.SelectionChanged.emit()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MiddleButton:
            self._startDrag = event.pos()
        elif event.button() == QtCore.Qt.LeftButton:
            self._selectionOverlay.startNewSelection(self.mapToScene(event.pos()), self._mainWin.getCurrentColor())
            self._selecting = True

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._startDrag is not None:
            p0 = event.pos()
            move = self._startDrag - p0
            self.horizontalScrollBar().setValue(move.x() + self.horizontalScrollBar().value())
            self.verticalScrollBar().setValue(move.y() + self.verticalScrollBar().value())
            self._startDrag = p0
        elif self._selecting:
            self._selectionOverlay.updateSelection(self.mapToScene(event.pos()), self._mainWin.getCurrentColor())
        else:
            pos: QtCore.QPointF = self.mapToScene(event.pos())
            x, y = int(round(pos.x())), int(round(pos.y()))
            self._mainWin.getSpecView().updateCursorSpectrum(x, y)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MiddleButton:
            self._startDrag = None
        elif event.button() == QtCore.Qt.LeftButton:
            pos: QtCore.QPoint = self.mapToScene(event.pos())
            self._selectionOverlay.updateSelection(pos, self._mainWin.getCurrentColor())
            self._selecting = False
            pixelIndices: Set[int] = self._selectionOverlay.finishSelection(pos)
            self.NewSelection.emit(self._mainWin.getCurrentClass(), pixelIndices)
            self.SelectionChanged.emit()
            self.scene().update()

    def wheelEvent(self, event: QtGui.QWheelEvent):
        factor = 1.01**(event.angleDelta().y()/8)
        self.scale(factor, factor)
        event.accept()

    def _setUpScene(self) -> QtWidgets.QGraphicsScene:
        scene = QtWidgets.QGraphicsScene(self)
        scene.setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)
        scene.setBackgroundBrush(QtCore.Qt.darkGray)
        self.setScene(scene)
        self.setCacheMode(QtWidgets.QGraphicsView.CacheBackground)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.BoundingRectViewportUpdate)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        return scene


class SelectionOverlay(QtWidgets.QGraphicsObject):
    def __init__(self):
        super(SelectionOverlay, self).__init__()
        self._startDrag: Union[None, Tuple[int, int]] = None  # x, y
        self._overlayArr: Union[None, np.ndarray] = None
        self._overlayPix: Union[None, QtGui.QPixmap] = None
        self._alpha: float = 0.8
        self.setZValue(1)

    def boundingRect(self) -> QtCore.QRectF:
        brect: QtCore.QRectF = QtCore.QRectF(0, 0, 1, 1)
        if self._overlayArr is not None:
            brect = QtCore.QRectF(0, 0, self._overlayArr.shape[0], self._overlayArr.shape[1])
        return brect

    def initOverlay(self, shape: Tuple[int, int, int]) -> None:
        self._overlayArr = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
        self._updatePixmap()

    def setPixelColors(self, pixCoords: Tuple[np.ndarray, np.ndarray], color: Tuple[int, int, int]) -> None:
        for i in range(len(pixCoords[0])):
            y, x = pixCoords[0][i], pixCoords[1][i]
            self._overlayArr[y, x, :3] = color
            self._overlayArr[y, x, 3] = 255
        self._updatePixmap()

    def startNewSelection(self, pos: QtCore.QPoint, colorRGB: Tuple[int, int, int]) -> None:
        assert self._overlayArr is not None
        x, y = int(pos.x()), int(pos.y())
        self._overlayArr[y, x, :3] = colorRGB
        self._overlayArr[y, x, 3] = 255
        self._updatePixmap()
        self._startDrag = x, y

    def updateSelection(self, pos: QtCore.QPoint, colorRGB: Tuple[int, int, int]) -> None:
        if self._startDrag is not None:
            x0, x1, y0, y1 = self._getStartStopCoords(pos)
            self._overlayArr[y0:y1, x0:x1, :3] = colorRGB
            self._overlayArr[y0:y1, x0:x1, 3] = 255
            self._updatePixmap()

    def finishSelection(self, pos: QtCore.QPoint) -> Set[int]:
        x0, x1, y0, y1 = self._getStartStopCoords(pos)
        self._startDrag = None
        return self._getCurrentSelectionIndices(x0, x1, y0, y1)

    def getPixelsOfColor(self, rgb: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets indices of the specified colors
        :param rgb:
        :return: Tuple[np.array(y-indices), np.array(x-indices)]
        """
        if self._overlayArr is not None:
            px = np.where(np.all(self._overlayArr[:, :, :3] == rgb, axis=-1))
        else:
            px = np.array([])
        return px

    def removeColor(self, rgb: Tuple[int, int, int]):
        yCoords, xCoords = self.getPixelsOfColor(rgb)
        for y, x in zip(yCoords, xCoords):
            self._overlayArr[y, x, :] = (0, 0, 0, 0)
        self._updatePixmap()

    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        painter.setOpacity(self._alpha)
        if self._overlayPix is not None:
            painter.drawPixmap(0, 0, self._overlayPix)

    def _updatePixmap(self) -> None:
        self._overlayPix = npy2Pixmap(self._overlayArr)
        self.update()

    def _getStartStopCoords(self, pos: QtCore.QPoint) -> Tuple[int, int, int, int]:
        """
        Takes a point and calculates start and stop x and y with respect to the started selection
        :param pos:
        :return:
        """
        x1, y1 = int(pos.x()), int(pos.y())
        x0, y0 = self._startDrag[0], self._startDrag[1]
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        return x0, x1, y0, y1

    def _getCurrentSelectionIndices(self, x0: int, x1: int, y0: int, y1: int) -> Set[int]:
        shape: np.ndarray = self._overlayArr.shape[:2]
        indices = getIndices(x0, x1, y0, y1, shape)
        return set(indices)


@numba.njit()
def getIndices(x0: int, x1: int, y0: int, y1: int, shape: np.ndarray) -> List[int]:
    """
    Numba optimized function to get indices in selected rectangle.
    :param x0:
    :param x1:
    :param y0:
    :param y1:
    :param shape:
    :return:
    """
    indices: List[int] = []
    i: int = 0
    for x in range(shape[1]):
        for y in range(shape[0]):
            curY = i // shape[1]
            curX = i % shape[1]
            if x0 <= curX <= x1 and y0 <= curY <= y1:
                indices.append(i)

            i += 1
    return indices


class ClassificationOverlay(QtWidgets.QGraphicsObject):
    def __init__(self):
        super(ClassificationOverlay, self).__init__()
        self._overlayArr: np.ndarray = None
        self._overlayPix: QtGui.QPixmap = None
        self._alpha: float = 2.0
        self.setZValue(1)

    def boundingRect(self) -> QtCore.QRectF:
        brect: QtCore.QRectF = QtCore.QRectF(0, 0, 1, 1)
        if self._overlayArr is not None:
            brect = QtCore.QRectF(0, 0, self._overlayArr.shape[0], self._overlayArr.shape[1])
        return brect

    def setAlpha(self, newAlpha: float) -> None:
        self._alpha = newAlpha
        self.update()

    def updateImage(self, img: np.ndarray) -> None:
        self._overlayArr = img
        self._overlayPix = npy2Pixmap(img)
        self.update()

    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        painter.setOpacity(self._alpha)
        if self._overlayPix is not None:
            painter.drawPixmap(0, 0, self._overlayPix)


def npy2Pixmap(img: np.ndarray) -> QtGui.QPixmap:
    height, width, channel = img.shape
    pix = QtGui.QPixmap()
    if channel == 3:
        bytesPerLine = 3 * width
        pix.convertFromImage(QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888))
    elif channel == 4:
        bytesPerLine = 4 * width
        pix.convertFromImage(QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGBA8888))
    else:
        raise ValueError('Only 3 or 4 channels supported')
    return pix


def cube2RGB(cube: np.ndarray, maxVal: float = 1.5) -> np.ndarray:
    """
    Converts HSI cube to rgb preview image
    :param cube: Array shape (NWavelength, Height, Width)
    :param maxVal: The maximum reflectance value to clip to
    :return: np.uint8 array shape (Height, Width, 3)
    """
    avg = np.mean(cube, axis=0)
    avg = np.clip(avg, 0.0, maxVal)
    avg -= avg.min()
    if avg.max() != 0:
        avg /= avg.max()
    avg *= 255.0
    img = np.stack((avg, avg, avg), axis=2)
    return img.astype(np.uint8)
