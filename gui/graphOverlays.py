"""
HSI Classifier
Copyright (C) 2021 Josef Brandt, University of Gothenburg <josef.brandt@gu.se>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program, see COPYING.
If not, see <https://www.gnu.org/licenses/>.
"""
import cv2
import numpy as np
import numba
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from typing import Tuple, Union, TYPE_CHECKING, Set, List, Dict, Optional

from gui.particleUI import getContourItemForParticle

if TYPE_CHECKING:
    from HSIEvaluator import MainWindow
    from gui.sampleview import SampleView
    from particles import Particle, ParticleHandler
    from gui.particleUI import ParticleContour
    from gui.classUI import ClassInterpretationParams


class GraphOverlays(QtCore.QObject):
    SelectionChanged: QtCore.pyqtSignal = QtCore.pyqtSignal()
    NewSelection: QtCore.pyqtSignal = QtCore.pyqtSignal(str, set)  # Tuple[class_name, set of pixelIndices]
    ParticlesChanged: QtCore.pyqtSignal = QtCore.pyqtSignal(str)  # name of sampleview

    def __init__(self):
        super(GraphOverlays, self).__init__()
        self._mainWin: Union[None, 'MainWindow'] = None
        self._sampleView: Union[None, 'SampleView'] = None
        self._origCube: Union[None, np.ndarray] = None
        self._selectionOverlay: SelectionOverlay = SelectionOverlay(self)
        self._classOverlay: ClassificationOverlay = ClassificationOverlay()
        self._particleItems: List['ParticleContour'] = []

    def setParentReferences(self, _sampleViewRef: 'SampleView', mainWinRef: 'MainWindow') -> None:
        self._sampleView = _sampleViewRef
        self._mainWin = mainWinRef
        self._selectionOverlay.setParentItem(_sampleViewRef)
        self._selectionOverlay.setMainWinRef(mainWinRef)
        self._classOverlay.setParentItem(_sampleViewRef)

    def setUpToCube(self, cube: np.ndarray) -> QtGui.QPixmap:
        """
        Sets references to the cube and initiates the selection overlay to the correct shape.
        """
        self._origCube = cube
        img = cube2RGB(cube)
        self._selectionOverlay.initOverlay(img.shape)
        return npy2Pixmap(img)

    def setCurrentlyPresentSelection(self, classes2Ind: Dict[str, Set[int]]) -> None:
        """
        Sets the current selection according to the provided classes2Ind dictionary.
        :param classes2Ind: Dict[classname: PixelIndices]
        """
        self._selectionOverlay.deselectAll()
        for cls, indices in classes2Ind.items():
            color: Tuple[int, int, int] = self._mainWin.getColorOfClass(cls)
            self._selectionOverlay.addPixelsToSelection(indices, color)
        self.SelectionChanged.emit()

    def deselectAll(self) -> None:
        """Removes all selection"""
        self._selectionOverlay.deselectAll()
        self.SelectionChanged.emit()

    def hideSelections(self) -> None:
        """
        Hides current selections, without commiting a new selection.
        """
        self._selectionOverlay.deselectAll()

    def setParticles(self, particles: List['Particle'], graphScene: QtWidgets.QGraphicsScene) -> None:
        """
        Takes a list of particles and creates visual objects accordingly. Previously present particles are removed.
        """
        # remove old particles
        for particle in self._particleItems:
            graphScene.removeItem(particle)

        self._particleItems = []
        for particle in particles:
            newContour: 'ParticleContour' = getContourItemForParticle(particle)
            newContour.setParentItem(self._sampleView)
            graphScene.addItem(newContour)
            self._particleItems.append(newContour)

    def toggleParticleVisibility(self) -> None:
        """
        Sets visibility of the particle contour items.
        """
        for item in self._particleItems:
            item.setVisible(not item.isVisible())

    def updateParticleColors(self, particleHandler: 'ParticleHandler', interpretationParams: 'ClassInterpretationParams'):
        """
        Updates the colors of the particle items in the graphics scene.
        :param particleHandler: Reference to particle handler
        :param interpretationParams: The params for interpreting the spec results
        """
        for particleItem in self._particleItems:
            assignment: str = particleHandler.getAssigmentOfParticleOfID(particleItem.getParticleID(), interpretationParams)
            color: Tuple[int, int, int] = self._mainWin.getColorOfClass(assignment)
            particleItem.setColor(color)

    def updateClassImage(self, classImage: np.ndarray) -> None:
        """
        Updates the graph representation of the current classification to the given RGBA image.
        """
        self._classOverlay.updateImage(classImage)

    def resetClassImage(self) -> None:
        """
        Resets the current graph representation of a previous classfication
        """
        self._classOverlay.resetOverlay()

    def updateClassImgTransp(self, newAlpha: float) -> None:
        self._classOverlay.setAlpha(np.clip(newAlpha, 0.0, 1.0))

    def getPixelsOfColor(self, rgb: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param rgb:
        :return: Tuple[np.array(y-indices), np.array(x-indices)]
        """
        return self._selectionOverlay.getPixelsOfColor(rgb)

    def getAveragedImage(self) -> np.ndarray:
        """
        Returns the averaged image of the original cube.
        """
        avgImg: np.ndarray = np.mean(cube2RGB(self._origCube, self._sampleView.getSelectedMaxBrightness()), axis=2)
        return avgImg

    def getCube(self) -> np.ndarray:
        return self._origCube

    def getGraphicItems(self) -> List[QtWidgets.QGraphicsItem]:
        """
        Returns a list of graphicsItems to be displayed from this sample.
        """
        return [self._selectionOverlay, self._classOverlay] + self._particleItems

    def removeColor(self, rgb: Tuple[int, int, int]) -> None:
        self._selectionOverlay.removeColor(rgb)

    @QtCore.pyqtSlot(str)
    def removeColorOfClass(self, className: str) -> None:
        rgb: Tuple[int, int, int] = self._mainWin.getColorOfClass(className)
        self.removeColor(rgb)
        self.SelectionChanged.emit()

    @QtCore.pyqtSlot(int, bool)
    def previewPixelsAccordingThreshold(self, threshold: int, bright: bool) -> None:
        """
        Selects pixels according a given brightnes threshold and previews with a default color.
        :param threshold: The threshold (0...255)
        :param bright: If True, the pixels brighter than the threshold are selected, otherwise the darker ones.
        """
        indices: Set[int] = getBrightOrDarkIndices(self._origCube, self._sampleView.getSelectedMaxBrightness(),
                                                   threshold, bright=bright)
        self._selectionOverlay.deselectAll()
        self._selectionOverlay.addPixelsToSelection(indices, (200, 200, 200))

    @QtCore.pyqtSlot(int, bool)
    def selectPixelsAccordingThreshold(self, threshold: int, bright: bool) -> None:
        """
        Selects pixels according a given brightnes threshold with the color of the currently selected class.
        :param threshold: The threshold (0...255)
        :param bright: If True, the pixels brighter than the threshold are selected, otherwise the darker ones.
        """
        self.deselectAll()
        indices: Set[int] = getBrightOrDarkIndices(self._origCube, self._sampleView.getSelectedMaxBrightness(),
                                                   threshold, bright=bright)
        self._selectionOverlay.addPixelsToSelection(indices, self._mainWin.getCurrentColor(), emitnewSelection=True)


class SelectionOverlay(QtWidgets.QGraphicsObject):
    def __init__(self, graphOverlayParent: 'GraphOverlays'):
        super(SelectionOverlay, self).__init__()
        self._selecting: bool = False
        self._graphParent: 'GraphOverlays' = graphOverlayParent
        self._mainWin: Optional['MainWindow'] = None
        self._startDrag: Optional[Tuple[int, int]] = None  # x, y
        self._overlayArr: Optional[np.ndarray] = None
        self._overlayPix: Optional[QtGui.QPixmap] = None
        self._alpha: float = 0.8
        self.setZValue(1)

    def boundingRect(self) -> QtCore.QRectF:
        brect: QtCore.QRectF = QtCore.QRectF(0, 0, 1, 1)
        if self._overlayArr is not None:
            brect = QtCore.QRectF(0, 0, self._overlayArr.shape[1], self._overlayArr.shape[0])
        return brect

    def setMainWinRef(self, mainWinRef: 'MainWindow') -> None:
        self._mainWin = mainWinRef

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton and event.modifiers() == QtCore.Qt.NoModifier and self._mainWin is not None:
            pos: QtCore.QPoint = self.mapToItem(self, event.pos())
            cubeShape = self._graphParent.getCube().shape
            if pos.x() < cubeShape[2] and pos.y() < cubeShape[1]:
                self.startNewSelection(pos, self._mainWin.getCurrentColor())
                self._selecting = True
        else:
            super(SelectionOverlay, self).mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._selecting and self._mainWin is not None:
            self.updateSelection(self.mapToItem(self, event.pos()), self._mainWin.getCurrentColor())
        else:
            super(SelectionOverlay, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton and self._mainWin is not None and self._selecting:
            pos: QtCore.QPoint = self.mapToItem(self, event.pos())
            self.updateSelection(pos, self._mainWin.getCurrentColor())
            self._selecting = False
            pixelIndices: Set[int] = self.finishSelection(pos)
            self._emitNewSelection(pixelIndices)
        else:
            super(SelectionOverlay, self).mouseReleaseEvent(event)

    def initOverlay(self, shape: Tuple[int, int, int]) -> None:
        self._overlayArr = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
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

    def _emitNewSelection(self, pixelIndices: Set[int]) -> None:
        self._graphParent.NewSelection.emit(self._mainWin.getCurrentClass(), pixelIndices)
        self._graphParent.SelectionChanged.emit()

    def deselectAll(self) -> None:
        self.initOverlay(self._overlayArr.shape)

    def addPixelsToSelection(self, pixelIndices: Set[int], rgb: Tuple[int, int, int], emitNewSelection: bool = False) -> None:
        """
        Adds the given pixel indices to the currently selected class.
        :param pixelIndices: The pixel indices to add
        :param rgb: The color to use
        :param emitNewSelection: If True, a new selection is emitted
        """
        for ind in pixelIndices:
            y: int = ind // self._overlayArr.shape[1]
            x: int = ind % self._overlayArr.shape[1]
            self._overlayArr[y, x, :3] = rgb
            self._overlayArr[y, x, 3] = 255
        self._updatePixmap()

        if emitNewSelection:
            self._emitNewSelection(pixelIndices)

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


def getBrightOrDarkIndices(cube: np.ndarray, maxBrightness: float, threshold: int, bright: bool = True) -> Set[int]:
    """
    Returns bright or dark pixel indices accoriding to the given threshold.
    :param cube: shape (K, M, N) cube of MxN spectra with K wavelenghts
    :param maxBrightness: Max brightness value to use for clipping the cube while converting to grayscale image
    :param threshold: The threshold value (0 - 255) to use for thresholding
    :param bright: If True, the bright pixel indices are returned, otherwise the dark ones
    :return Set of Pixel Indices
    """
    avgImg: np.ndarray = np.mean(cube2RGB(cube, maxBrightness), axis=2)
    if bright:
        thresh, binImg = cv2.threshold(avgImg, threshold, 255, cv2.THRESH_BINARY)
    else:
        thresh, binImg = cv2.threshold(avgImg, threshold, 255, cv2.THRESH_BINARY_INV)
    return set(np.where(binImg.flatten())[0])


def getThresholdedImage(cube: np.ndarray, maxBrightness: float, threshold: Union[int, None], bright: bool = True) -> np.ndarray:
    """
    Returns a thresholded image of the given spec cube.
    :param cube: shape (K, M, N) cube of MxN spectra with K wavelenghts
    :param maxBrightness: Max brightness value to use for clipping the cube while converting to grayscale image
    :param threshold: The threshold value (0 - 255) to use for thresholding. If None, OTSU's method is applied to find a threshold
    :param bright: If True, the bright pixel indices are returned, otherwise the dark ones
    :return Set of Pixel Indices
    """
    avgImg: np.ndarray = np.mean(cube2RGB(cube, maxBrightness), axis=2).astype(np.uint8)
    if type(threshold) == int:
        if bright:
            thresh, binImg = cv2.threshold(avgImg, threshold, 255, cv2.THRESH_BINARY)
        else:
            thresh, binImg = cv2.threshold(avgImg, threshold, 255, cv2.THRESH_BINARY_INV)
    elif threshold is None:
        if bright:
            thresh, binImg = cv2.threshold(avgImg, threshold, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            thresh, binImg = cv2.threshold(avgImg, threshold, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        raise TypeError
    return binImg


class ClassificationOverlay(QtWidgets.QGraphicsObject):
    def __init__(self):
        super(ClassificationOverlay, self).__init__()
        self._overlayArr: Union[None, np.ndarray] = None
        self._overlayPix: Union[None, QtGui.QPixmap] = None
        self._alpha: float = 2.0
        self.setZValue(1)

    def boundingRect(self) -> QtCore.QRectF:
        brect: QtCore.QRectF = QtCore.QRectF(0, 0, 1, 1)
        if self._overlayArr is not None:
            brect = QtCore.QRectF(0, 0, self._overlayArr.shape[1], self._overlayArr.shape[0])
        return brect

    def resetOverlay(self) -> None:
        """Sets a blank overlay"""
        if self._overlayArr is not None:
            blank: np.ndarray = np.zeros_like(self._overlayArr)
            self.updateImage(blank)

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

            
class ThresholdSelector(QtWidgets.QWidget):
    """
    Widget to determine a threshold for selecting bright or dark areas of the image.
    The thresholded image can also be used for particle detection.
    """
    ThresholdChanged: QtCore.pyqtSignal = QtCore.pyqtSignal(int, bool)
    ThresholdSelected: QtCore.pyqtSignal = QtCore.pyqtSignal(int, bool)
    ParticleDetectionRequired: QtCore.pyqtSignal = QtCore.pyqtSignal(int, bool)
    SelectionCancelled: QtCore.pyqtSignal = QtCore.pyqtSignal()

    def __init__(self, grayImage: np.ndarray):
        super(ThresholdSelector, self).__init__()
        self.setWindowTitle("Threshold Selector")
        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        fig: plt.Figure = plt.Figure(figsize=(3, 1.5))
        self._canvas: FigureCanvas = FigureCanvas(fig)
        axes: plt.Axes = fig.add_subplot()
        abundancies, binLimits = np.histogram(grayImage, bins=64)
        axes.plot(binLimits[0:-1], abundancies)

        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_xlim(0, 255)
        axes.axis("off")
        fig.tight_layout()
        self._canvas.draw()
        self._slider: QtWidgets.QSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setMaximum(255)
        self._slider.valueChanged.connect(self._emitChangedSignal)

        self._radioBright: QtWidgets.QRadioButton = QtWidgets.QRadioButton("Bright")
        self._radioBright.setChecked(True)
        self._radioBright.toggled.connect(self._emitChangedSignal)
        self._radioDark: QtWidgets.QRadioButton = QtWidgets.QRadioButton("Dark")
        self._radioDark.toggled.connect(self._emitChangedSignal)

        radioLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        radioLayout.addWidget(self._radioDark)
        radioLayout.addWidget(self._radioBright)
        radioLayout.addStretch()

        acceptBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Use as Selection")
        acceptBtn.released.connect(self._accept)
        acceptBtn.setMaximumWidth(120)

        particlesBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Use for Particle Detection")
        particlesBtn.released.connect(self._emitForParticleSelection)

        cancelBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Cancel")
        cancelBtn.released.connect(lambda: self.SelectionCancelled.emit())
        cancelBtn.setMaximumWidth(120)

        btnLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        btnLayout.addWidget(acceptBtn)
        btnLayout.addWidget(particlesBtn)
        btnLayout.addStretch()
        btnLayout.addWidget(cancelBtn)

        layout.addWidget(QtWidgets.QLabel("Use the slider to select the threshold."))
        layout.addWidget(self._slider)
        layout.addWidget(self._canvas)
        layout.addWidget(QtWidgets.QLabel("Select Dark Or Bright Image Parts"))
        layout.addLayout(radioLayout)
        layout.addStretch()
        layout.addLayout(btnLayout)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.SelectionCancelled.emit()
        a0.accept()

    def _emitChangedSignal(self) -> None:
        self.ThresholdChanged.emit(self._slider.value(), self._radioBright.isChecked())

    def _emitForParticleSelection(self) -> None:
        self.ParticleDetectionRequired.emit(self._slider.value(), self._radioBright.isChecked())

    def _accept(self) -> None:
        self.ThresholdSelected.emit(self._slider.value(), self._radioBright.isChecked())


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


def cube2RGB(cube: np.ndarray, maxVal: float = 1.5, defectThreshold: float = 1000.0) -> np.ndarray:
    """
    Converts HSI cube to rgb preview image
    :param cube: Array shape (NWavelength, Height, Width)
    :param maxVal: The maximum reflectance value to clip to
    :param defectThreshold: Values higher than that indicate defect (over saturated) pixels, which will be set to 0
    :return: np.uint8 array shape (Height, Width, 3)
    """
    avg = np.mean(cube, axis=0)
    avg[avg > defectThreshold] = 0
    avg = np.clip(avg, 0.0, maxVal)
    avg -= avg.min()
    if avg.max() != 0:
        avg /= avg.max()
    avg *= 255.0
    img = np.stack((avg, avg, avg), axis=2)
    return img.astype(np.uint8)
