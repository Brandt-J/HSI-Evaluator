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
from copy import deepcopy
from typing import *
import numpy as np
from PIL import Image, ImageEnhance
from PyQt5 import QtWidgets, QtCore, QtGui

from classification.classifyProcedures import createClassImg
from dataObjects import Sample
from gui.dbQueryWin import DatabaseQueryWindow
from gui.dbWin import DBUploadWin
from gui.graphOverlays import GraphOverlays, ThresholdSelector, cube2RGB, getThresholdedImage
from gui.sampleInfo import SampleInfo
from loadCube import loadCube
from logger import getLogger
from spectraObject import SpectraObject, getSpectraFromIndices

if TYPE_CHECKING:
    from logging import Logger
    from particles import ParticleHandler
    from gui.HSIEvaluator import MainWindow
    from gui.classUI import ClassInterpretationParams


class SampleView(QtWidgets.QGraphicsObject):
    """
    Grphical element for displaying a sample.
    """
    Activated: QtCore.pyqtSignal = QtCore.pyqtSignal(str)
    Renamed: QtCore.pyqtSignal = QtCore.pyqtSignal()
    Closed: QtCore.pyqtSignal = QtCore.pyqtSignal(str)
    ClassDeleted: QtCore.pyqtSignal = QtCore.pyqtSignal(str)
    BackgroundSelectionChanged: QtCore.pyqtSignal = QtCore.pyqtSignal()
    WavelenghtsChanged: QtCore.pyqtSignal = QtCore.pyqtSignal()
    NewClassificationResult: QtCore.pyqtSignal = QtCore.pyqtSignal()

    def __init__(self):
        super(SampleView, self).__init__()
        self.setAcceptHoverEvents(True)

        self._sampleData: Sample = Sample()
        self._isActive: bool = False

        self._mainWindow: Union[None, 'MainWindow'] = None
        self._graphOverlays: 'GraphOverlays' = GraphOverlays()
        self._threshSelector: Union[None, ThresholdSelector] = None
        self._dbQueryWin: Union[None, DatabaseQueryWindow] = None
        self._dbWin: Union[None, DBUploadWin] = None
        self._logger: 'Logger' = getLogger('SampleView')

        self._sampleInfo: SampleInfo = SampleInfo(self._sampleData.name)

        self._contextMenu: QtWidgets.QMenu = QtWidgets.QMenu()
        self._sampleMenu: QtWidgets.QMenu = QtWidgets.QMenu("Sample")
        self._dbMenu: QtWidgets.QMenu = QtWidgets.QMenu("Database")
        self._particlesMenu: QtWidgets.QMenu = QtWidgets.QMenu("Particles")

        self._imgAdjustWidget: ImageAdjustWidget = ImageAdjustWidget()
        self._imgAdjustWidget.ValuesChanged.connect(self.updateImage)

        self._closeAct: QtWidgets.QAction = QtWidgets.QAction("Close")
        self._uploadAct: QtWidgets.QAction = QtWidgets.QAction("Upload to Database")
        self._downloadAct: QtWidgets.QAction = QtWidgets.QAction("Download from Database")
        self._selectBrightnessAct: QtWidgets.QAction = QtWidgets.QAction("Brightness Select/\nParticleDetection")
        self._adjustBrightnessAct: QtWidgets.QAction = QtWidgets.QAction("Adjust Brightness/Contrast")

        self._pixmap: Optional[QtGui.QPixmap] = None

        self._establish_connections()
        self._configureWidgets()
        self._createContextMenu()

        self._setupWidgetsFromSampleData()

    def boundingRect(self) -> QtCore.QRectF:
        rect: QtCore.QRectF = QtCore.QRectF()
        if self._pixmap is not None:
            rect.setWidth(float(self._pixmap.size().width()))
            rect.setHeight(float(self._pixmap.size().height()))
        return rect

    @property
    def _classes2Indices(self) -> Dict[str, Set[int]]:
        """Shorthand for retrieving data from the sampleData object"""
        return self._sampleData.classes2Indices

    @_classes2Indices.setter
    def _classes2Indices(self, newData: Dict[str, Set[int]]) -> None:
        """Shorthand for setting data on the sampleData object"""
        self._sampleData.classes2Indices = newData

    @property
    def _name(self) -> str:
        """Shorthand for retrieveing name from the sampleData object"""
        return self._sampleData.name

    @_name.setter
    def _name(self, newName: str) -> None:
        """Shorthand for setting name on sampleData object"""
        self._sampleData.name = newName

    def setMainWindowReferences(self, parent: 'MainWindow') -> None:
        self._graphOverlays.setParentReferences(self, parent)
        self._mainWindow = parent

    def setUp(self, filePath: str, cube: np.ndarray, wavelengths: np.ndarray) -> None:
        self._sampleData.filePath = filePath
        self._sampleData.setDefaultName()
        self.setCube(cube, wavelengths)
        self._setupWidgetsFromSampleData()
        self.WavelenghtsChanged.emit()

    def mousePressEvent(self, event) -> None:
        if event.modifiers() == QtCore.Qt.ControlModifier:
            self.activate()
        elif event.button() == QtCore.Qt.RightButton:
            screenpos: QtCore.QPointF = event.screenPos()
            screenpos: QtCore.QPoint = QtCore.QPoint(int(screenpos.x()), int(screenpos.y()))
            self._contextMenu.exec_(screenpos)

    def hoverMoveEvent(self, event) -> None:
        pos: QtCore.QPointF = self.mapToItem(self, event.pos())
        x, y = int(round(pos.x())), int(round(pos.y()))
        cube: np.ndarray = self.getSpecObj().getCube()
        if cube is not None:
            if 0 <= x < cube.shape[2] and 0 <= y < cube.shape[1]:
                cursorSpec: np.ndarray = cube[:, y, x][np.newaxis, :]
                for proc in self._mainWindow.getPreprocessorsForSpecPreview():
                    cursorSpec = proc.applyToSpectra(cursorSpec)

                self._mainWindow.getresultPlots().updateCursorSpectrum(cursorSpec[0])

    def toggleToolarVisibility(self) -> None:
        self._sampleInfo.setVisible(not self._sampleInfo.isVisible())

    def setupFromSampleData(self) -> None:
        cube, wavelengths = loadCube(self._sampleData.filePath)
        self.setCube(cube, wavelengths)
        self._graphOverlays.setCurrentlyPresentSelection(self._classes2Indices)
        self._graphOverlays.setParticles(self._sampleData.getAllParticles(), self.scene())
        self._setupWidgetsFromSampleData()
        self._mainWindow.updateClassCreatorClasses()

    def saveCoordinatesToSampleData(self) -> None:
        self._sampleData.viewCoordinates = self.pos().x(), self.pos().y()

    def _setupWidgetsFromSampleData(self) -> None:
        self._sampleInfo.setName(self._sampleData.name)

    def setCube(self, cube: np.ndarray, wavelengths: np.ndarray) -> None:
        """
        Sets references to the spec cube.
        :param cube: Shape (KxMxN) cube with MxN spectra of K wavelenghts
        :param wavelengths: The corresponding K wavelengths.
        """
        if self._sampleData.flippedHorizontally:
            cube = np.flip(cube, axis=2)
        if self._sampleData.flippedVertically:
            cube = np.flip(cube, axis=1)

        self._sampleData.specObj.setCube(cube, wavelengths)
        self._pixmap = self._graphOverlays.setUpToCube(cube)

    def getSpecObj(self) -> 'SpectraObject':
        """
        Returns the spectra object.
        """
        return self._sampleData.specObj

    def getName(self) -> str:
        return self._name

    def getGraphOverlayObj(self) -> 'GraphOverlays':
        return self._graphOverlays

    def getWavelengths(self) -> np.ndarray:
        return self._sampleData.specObj.getWavelengths()

    def getSampleData(self) -> 'Sample':
        return self._sampleData

    def getSampleDataToSave(self) -> 'Sample':
        """
        Returns the sample data that is required for saving the sample to file.
        """
        saveSample: Sample = deepcopy(self._sampleData)
        saveSample.specObj = SpectraObject()
        saveSample.classOverlay = None
        saveSample.batchResult = None
        saveSample.particleHandler.resetParticleResults()
        return saveSample

    def getVisibleLabelledSpectra(self) -> Dict[str, np.ndarray]:
        """
        Gets the labelled Spectra that are currently set as visible, in form of a dictionary.
        :return: Dictionary [className, NxM array of N spectra with M wavelengths]
        """
        spectra: Dict[str, np.ndarray] = {}
        for name, indices in self._classes2Indices.items():
            if self._mainWindow.classIsVisible(name):
                spectra[name] = getSpectraFromIndices(np.array(list(indices)), self._sampleData.getSpecCube())

        return spectra

    def getClassNames(self) -> List[str]:
        """
        Returns the used class names.
        """
        return list(self._classes2Indices.keys())

    def getAllLabelledSpectra(self) -> Dict[str, np.ndarray]:
        """
        Gets all the labelled spectra in form of a dictionary. The spectra are NOT preprocessed
        :return: Dictionary [className, NxM array of N spectra with M wavelengths]
        """
        spectra: Dict[str, np.ndarray] = {}
        for name, indices in self._classes2Indices.items():
            spectra[name] = getSpectraFromIndices(np.array(list(indices)), self._sampleData.getSpecCube())
        return spectra

    def getSelectedMaxBrightness(self) -> float:
        """
        Get's the user selected max brightness value.
        """
        return self._imgAdjustWidget.getSelectedMaxBrightness()

    def setSampleData(self, data: 'Sample') -> None:
        """
        Sets the sample data.
        """
        self._sampleData = data

    def getSaveFileName(self) -> str:
        return self._sampleData.getFileHash() + '.pkl'

    def updateParticlesInGraphUI(self) -> None:
        """
        Forces an update of particles in the graph ui from the currently set sample data.
        """
        interpretationParams: 'ClassInterpretationParams' = self._mainWindow.getClassInterprationParams()
        self._graphOverlays.updateParticleColors(self._sampleData.getParticleHandler(), interpretationParams)
        self.NewClassificationResult.emit()

    def updateClassImageInGraphView(self) -> None:
        """
        Called after updating sample data. Creates a new class image and sets the graph display accordingly.
        """
        specConfCutoff: float = self._mainWindow.getClassInterprationParams().specConfThreshold
        if self._sampleData.batchResult is not None:
            specObj: 'SpectraObject' = self.getSpecObj()
            cubeShape = specObj.getCube().shape
            assignments: np.ndarray = self._sampleData.getBatchResults(specConfCutoff)
            colorDict: Dict[str, Tuple[int, int, int]] = self._mainWindow.getClassColorDict()

            clfImg: np.ndarray = createClassImg(cubeShape, assignments, colorDict)
            self._graphOverlays.updateClassImage(clfImg)
        self.NewClassificationResult.emit()

    def isActive(self) -> bool:
        return self._isActive

    def isSelectedForTraining(self) -> bool:
        return self._sampleInfo.isCheckedForTraining()

    def isSelectedForInference(self) -> bool:
        return self._sampleInfo.isCheckedForInference()

    def setSelectedForInference(self, selected: bool) -> None:
        self._sampleInfo.setCheckedForInference(selected)

    def setSelectedForTraining(self, selected: bool) -> None:
        self._sampleInfo.setCheckedForTraining(selected)

    def activate(self) -> None:
        self._isActive = True
        self.Activated.emit(self._name)

    def deactivate(self) -> None:
        self._isActive = False

    @QtCore.pyqtSlot(str)
    def removeClass(self, className: str) -> None:
        if className in self._classes2Indices.keys():
            del self._sampleData.classes2Indices[className]
            self._logger.info(f"Sample {self._name}: Deleted Selecion of class {className}")
            self.ClassDeleted.emit(className)

    @QtCore.pyqtSlot(str, str)
    def renameClass(self, oldName: str, newName: str) -> None:
        if oldName in self._classes2Indices.keys():
            self._classes2Indices[newName] = self._classes2Indices[oldName]
            del self._sampleData.classes2Indices[oldName]

    def _createContextMenu(self) -> None:
        self._sampleMenu.addAction(self._adjustBrightnessAct)
        self._sampleMenu.addSeparator()
        self._sampleMenu.addAction(self._closeAct)

        self._dbMenu.addAction(self._uploadAct)
        self._dbMenu.addAction(self._downloadAct)

        self._particlesMenu.addAction(self._selectBrightnessAct)

        self._contextMenu.addMenu(self._sampleMenu)
        self._contextMenu.addMenu(self._dbMenu)
        self._contextMenu.addMenu(self._particlesMenu)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        for subwin in [self._dbWin, self._threshSelector, self._imgAdjustWidget]:
            if subwin is not None:
                subwin.close()
        a0.accept()

    @QtCore.pyqtSlot(float, int, float)
    def updateImage(self, maxBrightness: float, newZero: int, newContrast: float) -> None:
        """
        Updating the previewed image with new zero value and contrast factor
        :param maxBrightness: Highest brightness to clip to
        :param newZero: integer value of the new zero value
        :param newContrast: float factor for contrast adjustment (1.0 = unchanged)
        :return:
        """
        QtWidgets.QMessageBox.warning(self, "Sorry", "At the moment not yet reimplemented...")
        # newImg: np.ndarray = cube2RGB(self._sampleData.getSpecCube(), maxBrightness)
        # if newZero != 0:
        #     newImg = newImg.astype(np.float)
        #     newImg = np.clip(newImg + newZero, 0, 255)
        #     newImg = newImg.astype(np.uint8)
        #
        # if newContrast != 1.0:
        #     img: Image = Image.fromarray(newImg)
        #     contrastObj = ImageEnhance.Contrast(img)
        #     newImg = np.array(contrastObj.enhance(newContrast))
        #
        # raise NotImplementedError  # Needs to be properly implemented!!!

    def _renameSample(self) -> None:
        newName, ok = QtWidgets.QInputDialog.getText(self._mainWindow, "Please enter a new name", "", text=self._name)
        if ok and newName != '':
            self._logger.info(f"Renaming {self._name} into {newName}")
            self._name = newName
            self._sampleInfo.setName(newName)
            self.Renamed.emit()

    def _configureWidgets(self) -> None:
        self._sampleInfo.getChangeNameBtn().ButtonClicked.connect(self._renameSample)

        self._closeAct.triggered.connect(lambda: self.Closed.emit(self._name))
        self._closeAct.setToolTip("Close Sample.")

        self._uploadAct.triggered.connect(self._uploadToSQL)
        self._uploadAct.setToolTip("Upload Spectra to SQL Database.")

        self._downloadAct.triggered.connect(self._downloadFromSQL)
        self._downloadAct.setToolTip("Download Spectra from SQL Database.")

        self._selectBrightnessAct.triggered.connect(self._selectByBrightness)
        self._adjustBrightnessAct.triggered.connect(self._adjustBrightness)

        self._sampleInfo.setParentItem(self)
        infoHeight: int = self._sampleInfo.boundingRect().height()
        self._sampleInfo.setPos(-(infoHeight + 5), -50)

    def _establish_connections(self) -> None:
        self._graphOverlays.NewSelection.connect(self._addNewSelection)

    @QtCore.pyqtSlot(str, set)
    def _addNewSelection(self, selectedClass: str,  selectedIndices: Set[int]) -> None:
        if selectedClass in self._classes2Indices:
            self._sampleData.classes2Indices[selectedClass].update(selectedIndices)
        else:
            self._sampleData.classes2Indices[selectedClass] = selectedIndices
        if selectedClass.lower() == 'background':
            self.BackgroundSelectionChanged.emit()

    def _selectByBrightness(self) -> None:
        """
        Opens the Threshold Selector to select bright or dark areas of the image.
        """
        self._mainWindow.disableWidgets()
        self._threshSelector = ThresholdSelector(self._graphOverlays.getAveragedImage())
        self._threshSelector.ThresholdChanged.connect(self._graphOverlays.previewPixelsAccordingThreshold)
        self._threshSelector.ThresholdSelected.connect(self._finishThresholdSelection)
        self._threshSelector.SelectionCancelled.connect(self._cancelThresholdSelection)
        self._threshSelector.ParticleDetectionRequired.connect(self.runParticleDetection)

        self._threshSelector.show()

    def _adjustBrightness(self) -> None:
        """
        Shows the brightness and contrast adjust widget.
        """
        self._imgAdjustWidget.show()

    @QtCore.pyqtSlot(int, bool)
    def _finishThresholdSelection(self, thresh: int, bright: bool) -> None:
        """
        Called, when a threshold for image selection is defined and should be applied to the currently selected class.
        :param thresh: Int Threshold (0...255)
        :param bright: If True, the pixels brighter than the threshold are selected, otherwise the darker ones.
        """
        self._graphOverlays.selectPixelsAccordingThreshold(thresh, bright)
        self._finishParticleDetection()

    @QtCore.pyqtSlot()
    def _cancelThresholdSelection(self) -> None:
        """
        Triggered when the Threshold Selector is closed or the cancel btn is pressed.
        """
        self._graphOverlays.hideSelections()
        self._finishParticleDetection()

    def _finishParticleDetection(self):
        """
        Closes and disconnects the threshold selector, re-enables widgets in main window and reset the selection
        widget to the actual selected classes.
        """
        if self._threshSelector is not None:
            self._threshSelector.ThresholdSelected.disconnect()
            self._threshSelector.ThresholdChanged.disconnect()
            self._threshSelector.SelectionCancelled.disconnect()
            self._threshSelector.close()
            self._threshSelector = None
            
        self._graphOverlays.setCurrentlyPresentSelection(self._classes2Indices)
        self._mainWindow.enableWidgets()

    def _uploadToSQL(self) -> None:
        """
        Open the window for uploading sample spectra to the SQL database.
        """
        self._mainWindow.disableWidgets()
        self._dbWin = DBUploadWin()
        self._dbWin.setSampleView(self)
        self._dbWin.UploadFinished.connect(self._sqlUploadFinished)
        self._dbWin.recreateLayout()
        self._dbWin.show()

    def _sqlUploadFinished(self) -> None:
        """
        Triggered when the SQL upload has finished.
        """
        self._mainWindow.enableWidgets()
        self._dbWin.UploadFinished.disconnect()
        self._dbWin.close()
        self._dbWin = None

    def _downloadFromSQL(self) -> None:
        """
        Open the window for SQL Query to download spectra.
        """
        self._mainWindow.disableWidgets()
        self._dbQueryWin = DatabaseQueryWindow()
        self._dbQueryWin.QueryFinished.connect(self._finishSQLDownload)
        self._dbQueryWin.AcceptResult.connect(self._acceptSQLDownload)
        self._dbQueryWin.show()

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, dict)
    def _acceptSQLDownload(self, cube: np.ndarray, wavelengths: np.ndarray, classes2ind: Dict[str, Set[int]]) -> None:
        """
        Receives results from SQL download and sets up the sampleview accordingly.
        :param cube: The spectra cube
        :param wavelengths: The wavelengths axis
        :param classes2ind: Dictionary with spectra labels
        """
        self.setCube(cube, wavelengths)
        self._classes2Indices = classes2ind
        self._setupWidgetsFromSampleData()
        self.WavelenghtsChanged.emit()

        self._graphOverlays.setCurrentlyPresentSelection(self._classes2Indices)

    def _finishSQLDownload(self) -> None:
        """
        Triggered when closing the SQL Query Window.
        """
        self._mainWindow.enableWidgets()
        self._dbQueryWin.QueryFinished.disconnect()
        self._dbQueryWin.AcceptResult.disconnect()
        self._dbQueryWin = None
        self._mainWindow.updateClassCreatorClasses()

    @QtCore.pyqtSlot(int, bool)
    def runParticleDetection(self, threshold: Union[int, None], selectBright: bool) -> None:
        """
        Takes a threshold and a brightness bool for creating a thresholded image that is used for creating a new
        list of particles. The particle handler stores this list and the graph view creates the according gui elements.
        :param threshold: The threshold to use (int, 0...255). If None, Otsu's method is used to determine the threshold.
        :param selectBright: If True, the bright areas are selected, otherwise the darker ones.
        """
        cube: np.ndarray = self._sampleData.specObj.getCube()
        try:
            binImg: np.ndarray = getThresholdedImage(cube, self._imgAdjustWidget.getSelectedMaxBrightness(),
                                                     threshold, selectBright)
        except TypeError:
            QtWidgets.QMessageBox.warning(self, "Error", "TypeError in getting thresholded image for particle detection.")

        else:
            particleHandler: 'ParticleHandler' = self._sampleData.particleHandler
            particleHandler.getParticlesFromImage(binImg)
            self._graphOverlays.setParticles(particleHandler.getParticles(), self.scene())
        self._finishParticleDetection()

    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        if self._pixmap is not None:
            painter.drawPixmap(0, 0, self._pixmap)
            if self._isActive:
                painter.setPen(QtCore.Qt.white)
                painter.drawRect(self.boundingRect())


class ImageAdjustWidget(QtWidgets.QWidget):
    """
    Widget for Image Adjustents.
    """
    ValuesChanged: QtCore.pyqtSignal = QtCore.pyqtSignal(float, int, float)  # (maxBrightness, Brightness, Contast)

    def __init__(self):
        super(ImageAdjustWidget, self).__init__()
        self._brightnessSlider: QtWidgets.QSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._contrastSlider: QtWidgets.QSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._maxContrast: float = 5
        self._maxBrightnessSpinbox: QtWidgets.QDoubleSpinBox = QtWidgets.QDoubleSpinBox()

        self._configureWidgets()
        self._createLayout()

    def getSelectedMaxBrightness(self) -> float:
        return self._maxBrightnessSpinbox.value()

    def _configureWidgets(self) -> None:
        self._brightnessSlider.setMinimum(-255)
        self._brightnessSlider.setMaximum(255)
        self._brightnessSlider.setValue(0)
        self._brightnessSlider.setFixedWidth(300)
        self._brightnessSlider.valueChanged.connect(self._emitChangedValues)

        contrastSteps = 100
        self._contrastSlider.setMinimum(0)
        self._contrastSlider.setMaximum(contrastSteps)
        self._contrastSlider.setValue(int(round(contrastSteps / self._maxContrast)))
        self._contrastSlider.setFixedWidth(300)
        self._contrastSlider.valueChanged.connect(self._emitChangedValues)

        self._maxBrightnessSpinbox.setMinimum(0.0)
        self._maxBrightnessSpinbox.setMaximum(100.0)
        self._maxBrightnessSpinbox.setValue(1.5)
        self._maxBrightnessSpinbox.setSingleStep(1.0)
        self._maxBrightnessSpinbox.valueChanged.connect(self._emitChangedValues)

    def _createLayout(self) -> None:
        layout: QtWidgets.QFormLayout = QtWidgets.QFormLayout()
        layout.addRow("Brightness", self._brightnessSlider)
        layout.addRow("Contrast", self._contrastSlider)
        layout.addRow("Max Reflect.", self._maxBrightnessSpinbox)

        self.setLayout(layout)

    def _emitChangedValues(self):
        self.ValuesChanged.emit(self._maxBrightnessSpinbox.value(),
                                self._brightnessSlider.value(),
                                self._contrastSlider.value() / self._maxContrast)


class ActivateToggleButton(QtWidgets.QPushButton):
    """
    Creates a toggle-Button for activating / deactivating the sample views.
    """
    def __init__(self):
        super(ActivateToggleButton, self).__init__()
        self.setCheckable(True)
        self.setChecked(True)
        self._adjustStyleAndMode()
        self.setFixedSize(70, 30)
        self.toggled.connect(self._adjustStyleAndMode)

    def _adjustStyleAndMode(self) -> None:
        if self.isChecked():
            self.setDisabled(True)
            self.setText("ACTIVE")
            self.setStyleSheet("QPushButton{background-color: lightblue; border: 1px solid black; border-radius: 7}")
        else:
            self.setEnabled(True)
            self.setText("Inactive")
            self.setStyleSheet("QPushButton{background-color: lightgrey; border: 1px solid black; border-radius: 7}")
