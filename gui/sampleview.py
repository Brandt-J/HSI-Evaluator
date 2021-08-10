import numba
from PyQt5 import QtWidgets, QtGui, QtCore
import pickle
import os
from typing import List, Tuple, TYPE_CHECKING, Dict, Set
import numpy as np

from logger import getLogger
from projectPaths import getAppFolder
from gui.graphOverlays import GraphView
from spectraObject import SpectraObject

if TYPE_CHECKING:
    from gui.ImecEvaluator import MainWindow
    from logging import Logger


class MultiSampleView(QtWidgets.QScrollArea):
    """
    Container class for showing multiple sampleviews in a ScrollArea.
    """
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
        newView.setParentToGraphView(self._mainWinParent)
        newView.SizeChanged.connect(self._recreateLayout)
        newView.Activated.connect(self._viewActivated)
        newView.activate()
        self._mainWinParent.setupConnections(newView)

        self._sampleviews.append(newView)
        self._logger.debug("New Sampleview added")
        return newView

    def getSampleViews(self) -> List['SampleView']:
        return self._sampleviews

    def getWavenumbers(self) -> np.ndarray:
        return self._sampleviews[0].getWavenumbers()

    def getLabelledSpectraFromActiveView(self) -> Dict[str, np.ndarray]:
        """
        Gets the labelled Spectra, in form of a dictionary, from the active ampleview
        :return: Dictionary [className, NxM array of N spectra with M wavenumbers]
        """
        spectra: Dict[str, np.ndarray] = {}
        for view in self._sampleviews:
            if view.isActive():
                spectra = view.getLabelledSpectra()
                break
        return spectra

    def getLabelledSpectraFromAllViews(self) -> Dict[str, Dict[str, np.ndarray]]:
        spectra: Dict[str, Dict[str, np.ndarray]] = {}
        for view in self._sampleviews:
            spectra[view.getName()] = view.getLabelledSpectra()
        return spectra

    @QtCore.pyqtSlot(str)
    def _viewActivated(self, samplename: str) -> None:
        """
        Handles activation of a new sampleview.
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


@numba.njit()
def getSpectraFromIndices(indices: np.ndarray, cube: np.ndarray) -> np.ndarray:
    """
    Retrieves the indices from the cube and returns an NxM array
    :param indices: length N array of flattened indices
    :param cube: XxYxZ spectral cube
    :return: (NxX) array of spectra
    """
    spectra: np.ndarray = np.zeros((len(indices), cube.shape[0]))
    for i, ind in enumerate(indices):
        y = ind // cube.shape[2]
        x = ind % cube.shape[2]
        spectra[i, :] = cube[:, y, x]
    return spectra


class SampleView(QtWidgets.QMainWindow):
    SizeChanged: QtCore.pyqtSignal = QtCore.pyqtSignal()
    Activated: QtCore.pyqtSignal = QtCore.pyqtSignal(str)

    def __init__(self):
        super(SampleView, self).__init__()
        self._name: str = ''
        self._specObj: SpectraObject = SpectraObject()
        self._graphView: 'GraphView' = GraphView()
        self._classes2Indices: Dict[str, Set[int]] = {}
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
        self._toolbar = QtWidgets.QToolBar("vertical")
        self._toolbar.addWidget(self._nameLabel)
        self._toolbar.addWidget(QtWidgets.QLabel('          '))
        self._toolbar.addWidget(self._activeBtn)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self._toolbar)
        self._configureWidgets()
        self._establish_connections()

    def setParentToGraphView(self, parent: 'MainWindow') -> None:
        self._graphView.setMainWindowReference(parent)

    def setUp(self, name: str, cube: np.ndarray) -> None:
        self._name = name
        self._nameLabel.setText(name)
        newFont: QtGui.QFont = QtGui.QFont()
        newFont.setBold(True)
        newFont.setPixelSize(18)
        self._nameLabel.setFont(newFont)

        self._graphView.setCube(cube)
        self._specObj.setCube(cube)
        self._createLayout()
        self.SizeChanged.emit()

    def getName(self) -> str:
        return self._name

    def getGraphView(self) -> 'GraphView':
        return self._graphView

    def getWavenumbers(self) -> np.ndarray:
        return self._specObj.getWavenumbers()

    def getLabelledSpectra(self) -> Dict[str, np.ndarray]:
        """
        Gets the labelled Spectra, in form of a dictionary.
        :return: Dictionary [className, NxM array of N spectra with M wavenumbers]
        """
        spectra: Dict[str, np.ndarray] = {}
        for name, indices in self._classes2Indices.items():
            spectra[name] = getSpectraFromIndices(np.array(list(indices)), self._specObj.getNotPreprocessedCube())
        return spectra

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
            del self._classes2Indices[className]
            self._logger.info(f"Sample {self._name}: Deleted Selecion of class {className}")
        else:
            self._logger.warning(f"Sample {self._name}: Requested deleting class {className}, but it was not in"
                                 f"dict.. Available keys: {self._classes2Indices.keys()}")

    def _checkActivation(self) -> None:
        if self._activeBtn.isChecked():
            self.Activated.emit(self._name)

    def _createLayout(self) -> None:
        adjustLayout: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        adjustLayout.addWidget(VerticalLabel("Brightness"), 0, 0)
        adjustLayout.addWidget(self._brightnessSlider, 0, 1)
        adjustLayout.addWidget(VerticalLabel("Contrast"), 1, 0)
        adjustLayout.addWidget(self._contrastSlider, 1, 1)
        adjustLayout.addWidget(VerticalLabel("Max Refl."), 2, 0)
        adjustLayout.addWidget(self._maxBrightnessSpinbox, 2, 1)
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

    def _establish_connections(self) -> None:
        self._activeBtn.toggled.connect(self._checkActivation)
        self._graphView.NewSelection.connect(self._addNewSelection)

    def _initiateImageUpdate(self) -> None:
        self._graphView.updateImage(self._maxBrightnessSpinbox.value(),
                                    self._brightnessSlider.value(),
                                    self._contrastSlider.value() / self._maxContrast)

    def _getSaveFileName(self) -> str:
        return os.path.join(getAppFolder(), 'saveFiles', self._name + '_savefile.pkl')

    def _saveViewToFile(self, saveFileName: str = 'test.pkl') -> None:
        resDict: dict = {'classesAndPixels': self.getClassesAndPixels(),
                         'descLib': self._specView.getDecsriptorLibrary()}
        with open(saveFileName, "wb") as fp:
            pickle.dump(resDict, fp)
        self._logger.info(f'saving sample view to {saveFileName}')

    def _loadViewFromFile(self, fname: str = 'test.pkl') -> None:
        raise NotImplementedError
        # if os.path.exists(fname):
        #     with open(fname, "rb") as fp:
        #         self._logger.info(f'loading sample view from {fname}')
        #         resDict = pickle.load(fp)
        #         self._specObj.setClasses(resDict['classesAndPixels'])
        #         self._specView.setDescriptorLibrary(resDict["descLib"])
        #         self._specView.updateSpectra()
        #         classes: List[str] = list(resDict['classesAndPixels'].keys())
        #         self._clsCreator.setClasses(classes)
        #         self._clsCreator.activateClass(classes[0])
        #         for cls in classes:
        #             self._graphView.setSelectionPixelsToColor(resDict['classesAndPixels'][cls],
        #                                                       self._clsCreator.getColorOfClassName(cls))

    @QtCore.pyqtSlot(str, set)
    def _addNewSelection(self, selectedClass: str,  selectedIndices: Set[int]) -> None:
        if selectedClass in self._classes2Indices:
            self._classes2Indices[selectedClass].update(selectedIndices)
        else:
            self._classes2Indices[selectedClass] = selectedIndices


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
