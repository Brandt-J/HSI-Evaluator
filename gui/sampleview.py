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

from PyQt5 import QtWidgets, QtGui, QtCore
import pickle
import os
from typing import List, Tuple, TYPE_CHECKING, Dict, Set, Union
import numpy as np
from copy import deepcopy

from logger import getLogger
from projectPaths import getAppFolder
from gui.graphOverlays import GraphView
from spectraObject import SpectraObject
from dataObjects import Sample, getSpectraFromIndices
from loadNumpyCube import loadNumpyCube

if TYPE_CHECKING:
    from gui.HSIEvaluator import MainWindow
    from logging import Logger


class MultiSampleView(QtWidgets.QScrollArea):
    """
    Container class for showing multiple sampleviews in a ScrollArea.
    """
    SampleClosed: QtCore.pyqtSignal = QtCore.pyqtSignal()

    def __init__(self, mainWinParent: 'MainWindow'):
        super(MultiSampleView, self).__init__()

        self._mainWinParent: 'MainWindow' = mainWinParent
        self._sampleviews: List['SampleView'] = []
        self._logger: 'Logger' = getLogger('MultiSampleView')

    def addSampleView(self) -> 'SampleView':
        """
        Adds a new sampleview and sets up the graphView properly
        :return: the new sampleview
        """
        newView: 'SampleView' = SampleView()
        newView.setMainWindowReferences(self._mainWinParent)
        newView.SizeChanged.connect(self._recreateLayout)
        newView.Activated.connect(self._viewActivated)
        newView.Closed.connect(self._viewClosed)
        newView.activate()
        self._mainWinParent.setupConnections(newView)

        self._sampleviews.append(newView)
        self._logger.debug("New Sampleview added")
        return newView

    def loadSampleViewFromFile(self, fpath: str) -> None:
        """Loads the sample configuration from the file and creates a sampleview accordingly"""
        newSampleData: 'Sample' = Sample()
        with open(fpath, "rb") as fp:
            loadedSampleData: 'Sample' = pickle.load(fp)
        newSampleData.__dict__.update(loadedSampleData.__dict__)
        self._createNewSampleFromSampleData(newSampleData)
        
    def createListOfSamples(self, sampleList: List['Sample']) -> None:
        """Creates a list of given samples and replaces the currently opened with that."""
        self._logger.info("Closing all samples, opening the following new ones..")
        for sample in self._sampleviews:
            sample.close()
        for sample in sampleList:
            self._createNewSampleFromSampleData(sample)
            self._logger.info(f"Creating sample {sample.name}")
        self._recreateLayout()

    def _createNewSampleFromSampleData(self, sampleData: Sample) -> None:
        """
        Creates a new sample and configures it according to the provided sample data object
        """
        newView: SampleView = self.addSampleView()
        newView.setSampleData(sampleData)
        newView.setupFromSampleData()

        classes: List[str] = list(sampleData.classes2Indices.keys())
        self._mainWinParent.checkForRequiredClasses(classes)

    def saveSamples(self) -> None:
        """
        Saves all the loaded samples individually.
        """
        for sample in self._sampleviews:
            self._saveSampleView(sample)

    def getSampleViews(self) -> List['SampleView']:
        """Returns a list of all samples"""
        return self._sampleviews.copy()

    def getActiveSample(self) -> 'SampleView':
        """
        Returns the currently active sample.
        """
        activeSample: Union[None, 'SampleView'] = None
        for sample in self._sampleviews:
            if sample.isActive():
                activeSample = sample
                break
        assert activeSample is not None
        return activeSample

    def getWavenumbers(self) -> np.ndarray:
        return self._sampleviews[0].getWavenumbers()

    def getLabelledSpectraFromActiveView(self) -> Dict[str, np.ndarray]:
        """
        Gets the labelled Spectra, in form of a dictionary, from the active sampleview
        :return: Dictionary [className, NxM array of N spectra with M wavenumbers]
        """
        spectra: Dict[str, np.ndarray] = {}
        for view in self._sampleviews:
            if view.isActive():
                spectra = view.getVisibleLabelledSpectra()
                break
        return spectra

    def getLabelledSpectraFromAllViews(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Gets the labelled Spectra, in form of a dictionary, from the all sampleviews
        :return: Dictionary [className, NxM array of N spectra with M wavenumbers]
        """
        spectra: Dict[str, Dict[str, np.ndarray]] = {}
        for view in self._sampleviews:
            spectra[view.getName()] = view.getVisibleLabelledSpectra()
        return spectra

    def getBackgroundOfActiveSample(self) -> np.ndarray:
        """
        Returns the averaged background spectrum of the active sample.
        """
        background: Union[None, np.ndarray] = None
        for sample in self._sampleviews:
            if sample.isActive():
                background = sample.getAveragedBackground()
                break
        assert background is not None
        return background

    def getBackgroundsOfAllSamples(self) -> Dict[str, np.ndarray]:
        """
        Returns the averaged backgounds of all samples.
        """
        backgrounds: Dict[str, np.ndarray] = {}
        for view in self._sampleviews:
            backgrounds[view.getName()] = view.getAveragedBackground()
        return backgrounds

    @QtCore.pyqtSlot(str)
    def _viewClosed(self, samplename: str) -> None:
        for view in self._sampleviews:
            if view.getName() == samplename:
                self._sampleviews.remove(view)
                self._logger.info(f"Closed Sample {samplename}")
                self.SampleClosed.emit()
                self._recreateLayout()
                break

    def _saveSampleView(self, sampleview: 'SampleView') -> None:
        """
        Saves the given sampleview.
        """
        directory: str = self.getSampleSaveDirectory()
        savePath: str = os.path.join(directory, sampleview.getSaveFileName())
        sampleData: 'Sample' = sampleview.getSampleDataToSave()
        with open(savePath, "wb") as fp:
            pickle.dump(sampleData, fp)

        self._logger.info(f"Saved sampleview {sampleview.getName()} at {savePath}")

    @QtCore.pyqtSlot(str)
    def _viewActivated(self, samplename: str) -> None:
        """
        Handles activation of a new sampleview, i.e., deactivates the previously active one.
        :param samplename: The name of the sample
        :return:
        """
        for view in self._sampleviews:
            if view.getName() != samplename and view.isActive():
                view.deactivate()

    def _recreateLayout(self) -> None:
        """
        Recreates the groupbox and layout containing the sampleviews, whenever a new sample is loaded.
        This ensures correct sizing of all child items.
        :return:
        """
        group: QtWidgets.QGroupBox = QtWidgets.QGroupBox()
        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        for sample in self._sampleviews:
            layout.addWidget(sample)
        group.setLayout(layout)
        self.setWidget(group)

    @staticmethod
    def getSampleSaveDirectory() -> str:
        """
        Returns the path of a directory used for storing individual sample views.
        """
        path: str = os.path.join(getAppFolder(), "Samples")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def getViewSaveDirectory() -> str:
        """
        Returns the path of a directoy used for storing the entirety of the current selection.
        """
        path: str = os.path.join(getAppFolder(), "Views")
        os.makedirs(path, exist_ok=True)
        return path


class SampleView(QtWidgets.QMainWindow):
    SizeChanged: QtCore.pyqtSignal = QtCore.pyqtSignal()
    Activated: QtCore.pyqtSignal = QtCore.pyqtSignal(str)
    Renamed: QtCore.pyqtSignal = QtCore.pyqtSignal()
    Closed: QtCore.pyqtSignal = QtCore.pyqtSignal(str)
    ClassDeleted: QtCore.pyqtSignal = QtCore.pyqtSignal(str)

    def __init__(self):
        super(SampleView, self).__init__()
        self._sampleData: Sample = Sample()

        self._mainWindow: Union[None, 'MainWindow'] = None
        self._graphView: 'GraphView' = GraphView()
        self._logger: 'Logger' = getLogger('SampleView')

        self._group: QtWidgets.QGroupBox = QtWidgets.QGroupBox()
        self._layout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        self._group.setLayout(self._layout)
        self.setCentralWidget(self._group)

        self._nameLabel: QtWidgets.QLabel = QtWidgets.QLabel()
        self._brightnessSlider: QtWidgets.QSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self._contrastSlider: QtWidgets.QSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self._maxContrast: float = 5
        self._maxBrightnessSpinbox: QtWidgets.QDoubleSpinBox = QtWidgets.QDoubleSpinBox()

        self._activeBtn: ActivateToggleButton = ActivateToggleButton()
        self._editNameBtn: QtWidgets.QPushButton = QtWidgets.QPushButton()
        self._closeBtn: QtWidgets.QPushButton = QtWidgets.QPushButton()
        self._selectAllBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Select All")
        self._selectNoneBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Select None")

        self._toolbar = QtWidgets.QToolBar()
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self._toolbar)
        self._configureWidgets()
        self._establish_connections()

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
        self._graphView.setParentReferences(self, parent)
        self._mainWindow = parent

    def setUp(self, filePath: str, cube: np.ndarray) -> None:
        self._sampleData.filePath = filePath
        self._sampleData.setDefaultName()
        self.setCube(cube)
        self._setupWidgetsFromSampleData()

    def setupFromSampleData(self) -> None:
        cube: np.ndarray = loadNumpyCube(self._sampleData.filePath)
        self.setCube(cube)
        self._graphView.setCurrentlyPresentSelection(self._classes2Indices)
        self._setupWidgetsFromSampleData()

    def _setupWidgetsFromSampleData(self) -> None:
        self._nameLabel.setText(self._sampleData.name)
        self.createLayout()
        self.SizeChanged.emit()

    def setCube(self, cube: np.ndarray) -> None:
        """Sets references to the spec cube."""
        self._sampleData.specObj.setCube(cube)
        self._graphView.setUpToCube(cube)

    def getSpecObj(self) -> 'SpectraObject':
        """
        Returns the spectra object.
        """
        return self._sampleData.specObj

    def getName(self) -> str:
        return self._name

    def getGraphView(self) -> 'GraphView':
        return self._graphView

    def getWavenumbers(self) -> np.ndarray:
        return self._sampleData.specObj.getWavenumbers()

    def getAveragedBackground(self) -> np.ndarray:
        """
        Returns the averaged background spectrum of the sample. If no background was selected, a np.zeros array is returned.
        :return: np.ndarray of background spectrum
        """
        cube: np.ndarray = self._sampleData.specObj.getNotPreprocessedCube()
        background: np.ndarray = np.zeros(cube.shape[0])
        backgroundFound: bool = False
        for cls_name in self._classes2Indices:
            if cls_name.lower() == 'background':
                indices = self._classes2Indices[cls_name]
                background = np.mean(getSpectraFromIndices(np.array(list(indices)), cube), axis=0)
                backgroundFound = True
                break

        if not backgroundFound:
            self._logger.warning(
                f'Sample: {self._name}: No Background found, although it was requested.. '
                f'Present Classes are: {list(self._classes2Indices.keys())}. Returning a np.zeros Background')

        return background

    def getSampleData(self) -> 'Sample':
        return self._sampleData

    def getSampleDataToSave(self) -> 'Sample':
        """
        Returns the sample data that is required for saving the sample to file.
        """
        saveSample: Sample = deepcopy(self._sampleData)
        saveSample.specObj = SpectraObject()
        saveSample.classOverlay = None
        return saveSample

    def getVisibleLabelledSpectra(self) -> Dict[str, np.ndarray]:
        """
        Gets the labelled Spectra, in form of a dictionary.
        :return: Dictionary [className, NxM array of N spectra with M wavenumbers]
        """
        spectra: Dict[str, np.ndarray] = {}
        for name, indices in self._classes2Indices.items():
            if self._mainWindow.classIsVisible(name):
                spectra[name] = getSpectraFromIndices(np.array(list(indices)), self._sampleData.specObj.getNotPreprocessedCube())
        return spectra

    def getSelectedMaxBrightness(self) -> float:
        """
        Get's the user selected max brightness value.
        """
        return self._maxBrightnessSpinbox.value()

    def setSampleData(self, data: 'Sample') -> None:
        self._sampleData = data

    def getSaveFileName(self) -> str:
        return self._sampleData.getFileHash() + '.pkl'

    def isActive(self) -> bool:
        return self._activeBtn.isChecked()

    def activate(self) -> None:
        self._activeBtn.setChecked(True)
        self._activeBtn.setEnabled(False)
        self.Activated.emit(self._name)

    def deactivate(self) -> None:
        self._activeBtn.setChecked(False)
        self._activeBtn.setEnabled(True)

    @QtCore.pyqtSlot(str)
    def removeClass(self, className: str) -> None:
        if className in self._classes2Indices.keys():
            del self._sampleData.classes2Indices[className]
            self._logger.info(f"Sample {self._name}: Deleted Selecion of class {className}")
            self.ClassDeleted.emit(className)
        else:
            self._logger.warning(f"Sample {self._name}: Requested deleting class {className}, but it was not in"
                                 f"dict.. Available keys: {self._classes2Indices.keys()}")

    def _renameSample(self) -> None:
        newName, ok = QtWidgets.QInputDialog.getText(self, "Please enter a new name", "", text=self._name)
        if ok and newName != '':
            self._logger.info(f"Renaming {self._name} into {newName}")
            self._name = newName
            self._nameLabel.setText(newName)
            self.Renamed.emit()

    def _checkActivation(self) -> None:
        if self._activeBtn.isChecked():
            self.Activated.emit(self._name)

    def createLayout(self) -> None:
        adjustLayout: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        adjustLayout.addWidget(VerticalLabel("Brightness"), 0, 0)
        adjustLayout.addWidget(self._brightnessSlider, 0, 1)
        adjustLayout.addWidget(VerticalLabel("Contrast"), 1, 0)
        adjustLayout.addWidget(self._contrastSlider, 1, 1)
        adjustLayout.addWidget(VerticalLabel("Max Refl."), 2, 0)
        adjustLayout.addWidget(self._maxBrightnessSpinbox, 2, 1)
        adjustLayout.addWidget(self._selectAllBtn, 3, 0, 1, 2)
        adjustLayout.addWidget(self._selectNoneBtn, 4, 0, 1, 2)
        self._layout.addLayout(adjustLayout)
        self._layout.addWidget(self._graphView)

    def _configureWidgets(self) -> None:
        self._brightnessSlider.setMinimum(-255)
        self._brightnessSlider.setMaximum(255)
        self._brightnessSlider.setValue(0)
        self._brightnessSlider.setFixedHeight(150)

        contrastSteps = 100
        self._contrastSlider.setMinimum(0)
        self._contrastSlider.setMaximum(contrastSteps)
        self._contrastSlider.setValue(int(round(contrastSteps / self._maxContrast)))
        self._contrastSlider.setFixedHeight(150)

        self._maxBrightnessSpinbox.setMinimum(0.0)
        self._maxBrightnessSpinbox.setMaximum(100.0)
        self._maxBrightnessSpinbox.setValue(1.5)
        self._maxBrightnessSpinbox.setSingleStep(1.0)

        self._maxBrightnessSpinbox.valueChanged.connect(self._initiateImageUpdate)
        self._brightnessSlider.valueChanged.connect(self._initiateImageUpdate)
        self._contrastSlider.valueChanged.connect(self._initiateImageUpdate)

        newFont: QtGui.QFont = QtGui.QFont()
        newFont.setBold(True)
        newFont.setPixelSize(18)
        self._nameLabel.setFont(newFont)

        self._editNameBtn.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogResetButton')))
        self._editNameBtn.released.connect(self._renameSample)

        self._closeBtn.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogDiscardButton')))
        self._closeBtn.released.connect(lambda: self.Closed.emit(self._name))

        self._selectAllBtn.released.connect(self._selectAllFromSample)
        self._selectNoneBtn.released.connect(self._selectNone)

        self._toolbar.addWidget(self._activeBtn)
        self._toolbar.addWidget(QtWidgets.QLabel('      '))
        self._toolbar.addWidget(self._editNameBtn)
        self._toolbar.addWidget(QtWidgets.QLabel('      '))
        self._toolbar.addWidget(self._nameLabel)
        self._toolbar.addWidget(QtWidgets.QLabel('      '))
        self._toolbar.addWidget(self._closeBtn)

    def _establish_connections(self) -> None:
        self._activeBtn.toggled.connect(self._checkActivation)
        self._graphView.NewSelection.connect(self._addNewSelection)

    def _initiateImageUpdate(self) -> None:
        self._graphView.updateImage(self._maxBrightnessSpinbox.value(),
                                    self._brightnessSlider.value(),
                                    self._contrastSlider.value() / self._maxContrast)

    def _getSaveFileName(self) -> str:
        return os.path.join(getAppFolder(), 'saveFiles', self._name + '_savefile.pkl')

    @QtCore.pyqtSlot(str, set)
    def _addNewSelection(self, selectedClass: str,  selectedIndices: Set[int]) -> None:
        if selectedClass in self._classes2Indices:
            self._sampleData.classes2Indices[selectedClass].update(selectedIndices)
        else:
            self._sampleData.classes2Indices[selectedClass] = selectedIndices

    def _selectAllFromSample(self) -> None:
        """
        If confirmed, all "bright" pixles will be assigned to the current class.
        """
        ret = QtWidgets.QMessageBox.question(self, "Continue", "Do you want to select all bright pixels?",
                                             QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                             QtWidgets.QMessageBox.Yes)
        if ret == QtWidgets.QMessageBox.Yes:
            self._graphView.selectAllBrightPixels()

    def _selectNone(self) -> None:
        """
        If confirmed, all pixels are deselected
        """
        ret = QtWidgets.QMessageBox.question(self, "Continue", "Do you want to deselect all pixels?",
                                             QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                             QtWidgets.QMessageBox.Yes)
        if ret == QtWidgets.QMessageBox.Yes:
            self._classes2Indices = {}
            self._graphView.deselectAll()


class VerticalLabel(QtWidgets.QLabel):
    """
    Vertically drawn Text Label, as adapted from:
    https://stackoverflow.com/questions/3757246/pyqt-rotate-a-qlabel-so-that-its-positioned-diagonally-instead-of-horizontally
    """
    def __init__(self, text: str = ''):
        super(VerticalLabel, self).__init__()
        self.text: str = text
        self._width: int = 5
        self._height: int = 5
        self._setWidthHeight()

    def setText(self, text: str) -> None:
        self.text = text
        self._setWidthHeight()

    def paintEvent(self, event):
        if self.text != '':
            painter = QtGui.QPainter(self)
            painter.translate(0, self.height())
            painter.rotate(-90)

            x, y = self._getStartXY()
            painter.drawText(y, x, self.text)
            painter.end()

    def _getStartXY(self) -> Tuple[int, int]:
        """
        Gets start Coordinates for the painter.
        :return:
        """
        xoffset = int(self._width / 2)
        yoffset = int(self._height / 2)
        x = int(self.width() / 2) + yoffset
        y = int(self.height() / 2) - xoffset
        return x, y

    def minimumSizeHint(self):
        return QtCore.QSize(self._height, self._width)

    def sizeHint(self):
        return QtCore.QSize(self._height, self._width)

    def _setWidthHeight(self, padding: int = 5) -> None:
        painter = QtGui.QPainter(self)
        fm = QtGui.QFontMetrics(painter.font())
        self._width = fm.boundingRect(self.text).width() + 2*padding
        self._height = fm.boundingRect(self.text).height() + 2*padding

        self.setFixedSize(self.sizeHint())


class ActivateToggleButton(QtWidgets.QPushButton):
    """
    Creates a toggle-Button for activating / deactivating the sample views.
    """
    def __init__(self):
        super(ActivateToggleButton, self).__init__()
        self.setCheckable(True)
        self.setChecked(True)
        self._adjustStyleAndMode()
        self.setFixedSize(100, 30)
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
