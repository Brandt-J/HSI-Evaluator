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
import random
import json
from PyQt5 import QtWidgets, QtGui, QtCore
import pickle
import os
from typing import List, Tuple, TYPE_CHECKING, Dict, Set, Union
import numpy as np
from copy import deepcopy

from logger import getLogger
from projectPaths import getAppFolder
from spectraObject import SpectraObject, SpectraCollection, getSpectraFromIndices, WavelengthsNotSetError
from dataObjects import Sample
from loadCube import loadCube
from legacyConvert import assertUpToDateSample
from gui.graphOverlays import GraphView, ThresholdSelector
from gui.dbWin import DBUploadWin
from gui.dbQueryWin import DatabaseQueryWindow

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
        self.setMinimumWidth(750)

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
        newView.WavelenghtsChanged.connect(self._assertIdenticalWavelengths)
        newView.activate()
        self._mainWinParent.setupConnections(newView)

        self._sampleviews.append(newView)
        self._logger.debug("New Sampleview added")
        self._recreateLayout()
        return newView

    def _assertIdenticalWavelengths(self) -> None:
        """
        Asserts that all samples have identical wavelenghts. If multiple wavelength axes are present, the shortest
        axis is used.
        """
        shortestWavelenghts, shortestWavelengthsLength = None, np.inf
        for sample in self._sampleviews:
            try:
                curWavelenghts: np.ndarray = sample.getWavelengths()
            except WavelengthsNotSetError:
                pass  # we just skip it here
            else:
                if len(curWavelenghts) < shortestWavelengthsLength:
                    shortestWavelengthsLength = len(curWavelenghts)
                    shortestWavelenghts = curWavelenghts

        if shortestWavelenghts is not None:
            for sample in self._sampleviews:
                try:
                    sample.getSpecObj().remapToWavelenghts(shortestWavelenghts)
                except WavelengthsNotSetError:
                    pass  # We can safely ignore that here.

    def loadSampleViewFromFile(self, fpath: str) -> None:
        """Loads the sample configuration from the file and creates a sampleview accordingly"""
        newSampleData: 'Sample' = Sample()
        with open(fpath, "rb") as fp:
            loadedSampleData: 'Sample' = pickle.load(fp)

        loadedSampleData = assertUpToDateSample(loadedSampleData)
        newSampleData.__dict__.update(loadedSampleData.__dict__)
        self._createNewSampleFromSampleData(newSampleData)
        
    def createListOfSamples(self, sampleList: List['Sample']) -> None:
        """Creates a list of given samples and replaces the currently opened with that."""
        self._logger.info("Closing all samples, opening the following new ones..")
        for sample in self._sampleviews:
            sample.close()
        self._sampleviews = []
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

        self._mainWinParent.updateClassCreatorClasses()

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

    def getWavelengths(self) -> np.ndarray:
        """
        Returns the wavelength axis.
        """
        wavelenghts: Union[None, np.ndarray] = None
        for sample in self._sampleviews:
            try:
                sampleWavelengths: np.ndarray = sample.getWavelengths()
            except WavelengthsNotSetError:
                self._logger.warning(f"No wavelengths set in sample {sample.getName()}")
            else:
                if wavelenghts is None:
                    wavelenghts = sampleWavelengths
                else:
                    assert np.array_equal(wavelenghts, sampleWavelengths)

        assert wavelenghts is not None
        return wavelenghts

    def getClassNamesFromAllSamples(self) -> Set[str]:
        """
        Returns the class names that are used from all the samples that are currently loaded.
        """
        clsNames: List[str] = []
        for sample in self._sampleviews:
            clsNames += list(sample.getClassNames())
        return set(clsNames)

    def getLabelledSpectraFromActiveView(self, preprocessed) -> SpectraCollection:
        """
        Gets the labelled Spectra, in form of a dictionary, from the active sampleview
        :param preprocessed: Whether or not to retrieve preprocessed spectra.
        :return: SpectraCollection with all the daata
        """
        specColl: SpectraCollection = SpectraCollection()
        for view in self._sampleviews:
            if view.isActive():
                spectra: Dict[str, np.ndarray] = view.getVisibleLabelledSpectra(preprocessed)
                specColl.addSpectraDict(spectra, view.getName())
                break
        return specColl

    def getLabelledSpectraFromAllViews(self, preprocessed) -> SpectraCollection:
        """
        Gets the labelled Spectra, in form of a dictionary, from the all sampleviews
        :param preprocessed: Whether or not to retrieve preprocessed spectra.
        :return: SpectraCollectionObject
        """
        specColl: SpectraCollection = SpectraCollection()
        for view in self._sampleviews:
            specColl.addSpectraDict(view.getVisibleLabelledSpectra(preprocessed), view.getName())
        return specColl

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

    def closeAllSamples(self) -> None:
        """
        Closes all opened sample views.
        """
        for sample in self._sampleviews:
            sample.close()
        self._recreateLayout()

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

    def exportSpectra(self) -> None:
        """
        Exports the labelled spectra for use in other software.
        """
        specColl: SpectraCollection = self.getLabelledSpectraFromAllViews()
        specArr, assignments = specColl.getXY()

        folder: str = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory where to save to.")
        numSpecs, ok = QtWidgets.QInputDialog.getInt(self, "Max. Number of Spectra to Export?",
                                                     "Enter the max. number of spectra to export.",
                                                          2000, 100, 10000, 500)
        if folder and ok:
            if len(assignments) > numSpecs:
                randInd: np.ndarray = np.array(random.sample(list(np.arange(len(assignments))), numSpecs))
                specArr = specArr[randInd, :]
                assignments = assignments[randInd]

            uniqueAssignments: List[str] = list(np.unique(assignments))
            assignmentDict: Dict[str, int] = {cls: uniqueAssignments.index(cls)+1 for cls in assignments}  # class 0 get's ignored by PLS Toolbox, hence we have the +1 here.
            numberAssignments: np.ndarray = np.array([assignmentDict[cls] for cls in assignments])

            specPath: str = os.path.join(folder, f"Exported Spectra from {len(self._sampleviews)} samples.txt")
            np.savetxt(specPath, specArr)
            assignPath: str = os.path.join(folder, f"Exported Assignments from {len(self._sampleviews)} samples.txt")
            np.savetxt(assignPath, numberAssignments)

            codePath: str = os.path.join(folder, f"Exported Spectra Encoding from {len(self._sampleviews)} samples.txt")
            with open(codePath, "w") as fp:
                json.dump(assignmentDict, fp)

            QtWidgets.QMessageBox.about(self, "Info", f"Spectra and Assignments saved to\n{folder}\n\n"
                                                      f"Class encoding:\n"
                                                      f"{assignmentDict}")


class SampleView(QtWidgets.QMainWindow):
    SizeChanged: QtCore.pyqtSignal = QtCore.pyqtSignal()
    Activated: QtCore.pyqtSignal = QtCore.pyqtSignal(str)
    Renamed: QtCore.pyqtSignal = QtCore.pyqtSignal()
    Closed: QtCore.pyqtSignal = QtCore.pyqtSignal(str)
    ClassDeleted: QtCore.pyqtSignal = QtCore.pyqtSignal(str)
    BackgroundSelectionChanged: QtCore.pyqtSignal = QtCore.pyqtSignal()
    WavelenghtsChanged: QtCore.pyqtSignal = QtCore.pyqtSignal()

    def __init__(self):
        super(SampleView, self).__init__()
        self._sampleData: Sample = Sample()

        self._mainWindow: Union[None, 'MainWindow'] = None
        self._graphView: 'GraphView' = GraphView()
        self._threshSelector: Union[None, ThresholdSelector] = None
        self._dbQueryWin: Union[None, DatabaseQueryWindow] = None
        self._dbWin: Union[None, DBUploadWin] = None
        self._logger: 'Logger' = getLogger('SampleView')

        self._group: QtWidgets.QGroupBox = QtWidgets.QGroupBox()
        self.setCentralWidget(self._group)

        self._nameLabel: QtWidgets.QLabel = QtWidgets.QLabel()
        self._brightnessSlider: QtWidgets.QSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self._contrastSlider: QtWidgets.QSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self._maxContrast: float = 5
        self._maxBrightnessSpinbox: QtWidgets.QDoubleSpinBox = QtWidgets.QDoubleSpinBox()

        self._activeBtn: ActivateToggleButton = ActivateToggleButton()
        self._editNameBtn: QtWidgets.QPushButton = QtWidgets.QPushButton()
        self._closeBtn: QtWidgets.QPushButton = QtWidgets.QPushButton()
        self._uploadBtn: QtWidgets.QPushButton = QtWidgets.QPushButton()
        self._downloadBtn: QtWidgets.QPushButton = QtWidgets.QPushButton()

        self._trainCheckBox: QtWidgets.QCheckBox = QtWidgets.QCheckBox("Use for training")
        self._inferenceCheckBox: QtWidgets.QCheckBox = QtWidgets.QCheckBox("Use for validation")
        self._selectByBrightnessBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Brightness Select")
        self._selectNoneBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Select None")

        self._toolbar = QtWidgets.QToolBar()
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self._toolbar)

        self._configureWidgets()
        self._establish_connections()
        self._createLayout()
        self._createToolbar()
        self._setupWidgetsFromSampleData()

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

    def setUp(self, filePath: str, cube: np.ndarray, wavelengths: np.ndarray) -> None:
        self._sampleData.filePath = filePath
        self._sampleData.setDefaultName()
        self.setCube(cube, wavelengths)
        self._setupWidgetsFromSampleData()
        self.WavelenghtsChanged.emit()

    def setupFromSampleData(self) -> None:
        cube, wavelengths = loadCube(self._sampleData.filePath)
        self.setCube(cube, wavelengths)
        self._graphView.setCurrentlyPresentSelection(self._classes2Indices)
        self._setupWidgetsFromSampleData()
        self._mainWindow.updateClassCreatorClasses()

    def _setupWidgetsFromSampleData(self) -> None:
        self._nameLabel.setText(self._sampleData.name)
        self.SizeChanged.emit()

    def setCube(self, cube: np.ndarray, wavelengths: np.ndarray) -> None:
        """
        Sets references to the spec cube.
        :param cube: Shape (KxMxN) cube with MxN spectra of K wavelenghts
        :param wavelengths: The corresponding K wavelengths.
        """
        self._sampleData.specObj.setCube(cube, wavelengths)
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

    def getWavelengths(self) -> np.ndarray:
        return self._sampleData.specObj.getWavelengths()

    def getAveragedBackground(self) -> np.ndarray:
        """
        Returns the averaged background spectrum of the sample. If no background was selected, a np.zeros array is returned.
        :return: np.ndarray of background spectrum
        """
        cube: np.ndarray = self.getNotPreprocessedCube()
        background: np.ndarray = np.zeros(cube.shape[0])
        backgrounIndices: Set[int] = self._sampleData.getBackroundIndices()
        if len(backgrounIndices) > 0:
            indices: np.ndarray = np.array(list(backgrounIndices))
            background = np.mean(getSpectraFromIndices(indices, cube), axis=0)

        else:
            self._logger.warning(
                f'Sample: {self._name}: No Background found, although it was requested.. '
                f'Present Classes are: {list(self._classes2Indices.keys())}. Returning a np.zeros Background')

        return background

    def getBackgroundPixelIndices(self) -> Set[int]:
        """
        Returns a list of pixels in the background class
        """
        backgroundIndices: Set[int] = set()
        for cls, pixelIndices in self._classes2Indices.items():
            if cls.lower() == "background":
                backgroundIndices = pixelIndices
                break
        return backgroundIndices

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

    def getVisibleLabelledSpectra(self, preprocessed: bool) -> Dict[str, np.ndarray]:
        """
        Gets the labelled Spectra that are currently set as visible, in form of a dictionary.
        :param preprocessed: Whether or not to retrieve preprocessed spectra.
        :return: Dictionary [className, NxM array of N spectra with M wavelengths]
        """
        spectra: Dict[str, np.ndarray] = {}
        for name, indices in self._classes2Indices.items():
            if self._mainWindow.classIsVisible(name):
                if preprocessed:
                    spectra[name] = getSpectraFromIndices(np.array(list(indices)), self._sampleData.specObj.getPreprocessedCubeIfPossible())
                else:
                    spectra[name] = getSpectraFromIndices(np.array(list(indices)), self._sampleData.specObj.getNotPreprocessedCube())
        return spectra

    def getClassNames(self) -> List[str]:
        """
        Returns the used class names.
        """
        return list(self._classes2Indices.keys())

    def getAllLabelledSpectra(self) -> Dict[str, np.ndarray]:
        """
        Gets all the labelled spectra in form of a dictionary.
        :return: Dictionary [className, NxM array of N spectra with M wavelengths]
        """
        spectra: Dict[str, np.ndarray] = {}
        for name, indices in self._classes2Indices.items():
            spectra[name] = getSpectraFromIndices(np.array(list(indices)),
                                                  self._sampleData.specObj.getPreprocessedCubeIfPossible())
        return spectra

    def getAllLabelledNOTPreprocesssedSpectra(self) -> Dict[str, np.ndarray]:
        """
        Gets all the labelled, NOT preprocessed spectra in form of a dictionary.
        :return: Dictionary [className, NxM array of N spectra with M wavelengths]
        """
        spectra: Dict[str, np.ndarray] = {}
        for name, indices in self._classes2Indices.items():
            spectra[name] = getSpectraFromIndices(np.array(list(indices)),
                                                  self._sampleData.specObj.getNotPreprocessedCube())
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

    def resetClassificationOverlay(self) -> None:
        """
        Resets the current classification overlay.
        """
        self._graphView.resetClassImage()

    def isActive(self) -> bool:
        return self._activeBtn.isChecked()

    def isSelectedForTraining(self) -> bool:
        return self._trainCheckBox.isChecked()

    def isSelectedForInference(self) -> bool:
        return self._inferenceCheckBox.isChecked()

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

    def _createLayout(self) -> None:
        self._layout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        self._group.setLayout(self._layout)

        adjustLayout: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        adjustLayout.addWidget(VerticalLabel("Brightness"), 0, 0)
        adjustLayout.addWidget(self._brightnessSlider, 0, 1)
        adjustLayout.addWidget(VerticalLabel("Contrast"), 1, 0)
        adjustLayout.addWidget(self._contrastSlider, 1, 1)
        adjustLayout.addWidget(VerticalLabel("Max Refl."), 2, 0)
        adjustLayout.addWidget(self._maxBrightnessSpinbox, 2, 1)
        adjustLayout.addWidget(self._selectByBrightnessBtn, 3, 0, 1, 2)
        adjustLayout.addWidget(self._selectNoneBtn, 4, 0, 1, 2)
        self._layout.addLayout(adjustLayout)
        self._layout.addWidget(self._graphView)

    def _createToolbar(self):
        nameGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox("Sample Name")
        nameGroup.setLayout(QtWidgets.QHBoxLayout())
        nameGroup.layout().addWidget(self._editNameBtn)
        nameGroup.layout().addWidget(self._nameLabel)

        clsGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox("Usage in classification:")
        clsGroup.setLayout(QtWidgets.QHBoxLayout())
        clsGroup.layout().addWidget(self._trainCheckBox)
        clsGroup.layout().addWidget(self._inferenceCheckBox)

        actionsGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox("Actions")
        actionsGroup.setLayout(QtWidgets.QHBoxLayout())
        actionsGroup.layout().addWidget(self._uploadBtn)
        actionsGroup.layout().addWidget(self._downloadBtn)
        actionsGroup.layout().addWidget(self._closeBtn)

        toolGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox()
        toolGroup.setFlat(True)
        toolGroup.setLayout(QtWidgets.QHBoxLayout())
        toolGroup.layout().addWidget(self._activeBtn)
        toolGroup.layout().addStretch()
        toolGroup.layout().addWidget(nameGroup)
        toolGroup.layout().addStretch()
        toolGroup.layout().addWidget(clsGroup)
        toolGroup.layout().addStretch()
        toolGroup.layout().addWidget(actionsGroup)

        self._toolbar.addWidget(toolGroup)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        for subwin in [self._dbWin, self._threshSelector]:
            if subwin is not None:
                subwin.close()
        a0.accept()

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
        self._editNameBtn.setToolTip("Rename Sample.")

        self._closeBtn.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogDiscardButton')))
        self._closeBtn.released.connect(lambda: self.Closed.emit(self._name))
        self._closeBtn.setToolTip("Close Sample.")

        self._uploadBtn.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_ArrowUp')))
        self._uploadBtn.released.connect(self._uploadToSQL)
        self._uploadBtn.setToolTip("Upload Spectra to SQL Database.")

        self._downloadBtn.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_ArrowDown')))
        self._downloadBtn.released.connect(self._downloadFromSQL)
        self._downloadBtn.setToolTip("Download Spectra from SQL Database.")

        self._trainCheckBox.setChecked(True)
        self._inferenceCheckBox.setChecked(True)

        self._selectByBrightnessBtn.released.connect(self._selectByBrightness)
        self._selectNoneBtn.released.connect(self._selectNone)

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
        if selectedClass.lower() == 'background':
            self.BackgroundSelectionChanged.emit()

    def _selectByBrightness(self) -> None:
        """
        Opens the Threshold Selector to select bright or dark areas of the image.
        """
        self._mainWindow.disableWidgets()
        self._threshSelector = ThresholdSelector(self._graphView.getAveragedImage())
        self._threshSelector.ThresholdChanged.connect(self._graphView.previewPixelsAccordingThreshold)
        self._threshSelector.ThresholdSelected.connect(self._finishThresholdSelection)
        self._threshSelector.SelectionCancelled.connect(self._cancelThresholdSelection)
        self._threshSelector.show()

    @QtCore.pyqtSlot(int, bool)
    def _finishThresholdSelection(self, thresh: int, bright: bool) -> None:
        """
        Called, when a threshold for image selection is defined and should be applied to the currently selected class.
        :param thresh: Int Threshold (0...255)
        :param bright: If True, the pixels brighter than the threshold are selected, otherwise the darker ones.
        """
        self._graphView.selectPixelsAccordingThreshold(thresh, bright)
        self._closeThresholdSelector()

    @QtCore.pyqtSlot()
    def _cancelThresholdSelection(self) -> None:
        """
        Triggered when the Threshold Selector is closed or the cancel btn is pressed.
        """
        self._graphView.hideSelections()
        self._closeThresholdSelector()

    def _closeThresholdSelector(self):
        """
        Closes and disconnects the threshold selector, re-enables widgets in main window and reset the selection
        widget to the actual selected classes.
        """
        self._threshSelector.ThresholdSelected.disconnect()
        self._threshSelector.ThresholdChanged.disconnect()
        self._threshSelector.SelectionCancelled.disconnect()
        self._threshSelector.close()
        self._threshSelector = None
        self._graphView.setCurrentlyPresentSelection(self._classes2Indices)
        self._mainWindow.enableWidgets()

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

        self._graphView.setCurrentlyPresentSelection(self._classes2Indices)

    def _finishSQLDownload(self) -> None:
        """
        Triggered when closing the SQL Query Window.
        """
        self._mainWindow.enableWidgets()
        self._dbQueryWin.QueryFinished.disconnect()
        self._dbQueryWin.AcceptResult.disconnect()
        self._dbQueryWin = None
        self._mainWindow.updateClassCreatorClasses()


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
