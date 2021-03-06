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
import os.path
import pickle
from dataclasses import dataclass

from PyQt5 import QtWidgets, QtCore
from typing import List, Tuple, Dict, Union, TYPE_CHECKING, Callable, cast, Set, Optional
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.pyplot import rcParams
from threading import Thread, Event

from logger import getLogger
from gui.graphOverlays import npy2Pixmap
from dataObjects import Sample
import classification.classifyProcedures as cp
from classification.classifiers import ClassificationError, BaseClassifier, SavedClassifier
from projectPaths import getClassifierSaveFolder

if TYPE_CHECKING:
    from gui.HSIEvaluator import MainWindow
    from gui.sampleview import SampleView
    from logging import Logger


class ClassObject:
    """
    Container to store information about a selectable class.
    """
    def __init__(self, classname: str):
        self.name: str = classname
        self._visible: bool = True
        self.btn: QtWidgets.QRadioButton = QtWidgets.QRadioButton(classname)
        self.visCheckBox: QtWidgets.QCheckBox = QtWidgets.QCheckBox()
        self.visCheckBox.setChecked(self._visible)
        self.visCheckBox.stateChanged.connect(self._saveVisState)
        self.delBtn: QtWidgets.QPushButton = QtWidgets.QPushButton()
        self.delBtn.setIcon(QtWidgets.QGroupBox().style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogCloseButton')))
        self.delBtn.setFixedWidth(20)

    def recreateWidgets(self) -> None:
        self.btn = QtWidgets.QRadioButton(self.name)
        self.visCheckBox = QtWidgets.QCheckBox()
        self.visCheckBox.setChecked(self._visible)
        self.visCheckBox.stateChanged.connect(self._saveVisState)
        self.delBtn: QtWidgets.QPushButton = QtWidgets.QPushButton()
        self.delBtn.setIcon(
            QtWidgets.QGroupBox().style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogCloseButton')))
        self.delBtn.setFixedWidth(20)

    def isSelected(self) -> bool:
        return self.btn.isChecked()

    def isVisible(self) -> bool:
        return self.visCheckBox.isChecked()

    def _saveVisState(self) -> None:
        self._visible = self.visCheckBox.isChecked()


class ClassCreator(QtWidgets.QGroupBox):
    """
    UI Element for creating and deleting classes, and for toggling their visibility.
    """
    ClassCreated: QtCore.pyqtSignal = QtCore.pyqtSignal()
    ClassDeleted: QtCore.pyqtSignal = QtCore.pyqtSignal(str)
    ClassActivated: QtCore.pyqtSignal = QtCore.pyqtSignal(str)
    ClassVisibilityChanged: QtCore.pyqtSignal = QtCore.pyqtSignal()

    def __init__(self):
        super(ClassCreator, self).__init__()
        self._classes: List[ClassObject] = []
        self._colorHandler: ColorHandler = ColorHandler()
        self._logger: 'Logger' = getLogger("ClassCreator")
        self._activeCls: Union[None, str] = None
        self._newClsBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("New Class")
        self._newClsBtn.released.connect(self._createNewClass)
        self._layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)
        self._createAndAcivateBackgroundClass()
        self._recreateLayout()

    def activateClass(self, clsName: str) -> None:
        if clsName in self._classes:
            self._activeCls = clsName
            self.ClassActivated.emit(clsName)
            
    def setupToClasses(self, classes: Set[str]) -> None:
        """
        Creates classes that are needed if they aren't yet present
        """
        if type(classes) == list:
            classes = set(classes)
        presentClassNames: List[str] = [clsObj.name for clsObj in self._classes]
        for cls in classes:
            if cls not in presentClassNames:
                self._addClass(cls)
        self._recreateLayout()

    def deleteAllClasses(self) -> None:
        """
        Deletes all currently present classes.
        """
        self._classes = []
        self._colorHandler.deleteAllClasses()
        self._recreateLayout()

    def getCurrentColor(self) -> Tuple[int, int, int]:
        color: Union[None, tuple] = None
        for cls in self._classes:
            if cls.isSelected():
                color = self.getColorOfClassName(cls.name)
                break
        assert color is not None
        return color

    def getCurrentClass(self) -> str:
        name: str = ''
        for cls in self._classes:
            if cls.isSelected():
                name = cls.name
                break
        return name

    def getClassColorDict(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Returns a dictionary containing rgb colors for all present classes.
        """
        colorDict: Dict[str, Tuple[int, int, int]] = {"unknown": self._getColorOfClassUnknown()}
        for cls in self._classes:
            colorDict[cls.name] = self._colorHandler.getColorOfClassName(cls.name)
        return colorDict

    def getColorOfClassName(self, className: str) -> Tuple[int, int, int]:
        if className == "unknown":
            color: Tuple[int, int, int] = self._getColorOfClassUnknown()
        else:
            color: Tuple[int, int, int] = self._colorHandler.getColorOfClassName(className)
        return color

    def _getColorOfClassUnknown(self) -> Tuple[int, int, int]:
        return 20, 20, 20

    def getClassVisibility(self, className: str) -> bool:
        visible: bool = True
        for cls in self._classes:
            if cls.name == className:
                visible = cls.isVisible()
                break
        return visible

    def _createAndAcivateBackgroundClass(self) -> None:
        self._addClass("Background")
        self._classes[0].btn.setChecked(True)

    def _createNewClass(self) -> None:
        newName, ok = QtWidgets.QInputDialog.getText(self, "Create New Class", "Class Name:", text="New class")
        if ok:
            if newName in [cls.name for cls in self._classes]:
                QtWidgets.QMessageBox.warning(self, "Warning", f"The class {newName} already exists.\n"
                                                               f"Please try again.")
                self._createNewClass()
            else:
                self._addClass(newName)
                self._activateClass(newName)
                self._recreateLayout()

    def _addClass(self, name: str) -> None:
        """
        Creates a new class with the given name and sets up all connections.
        :param name: The desired class name
        """
        newCls: ClassObject = ClassObject(name)
        self._connectClassObj(newCls)
        self._classes.append(newCls)

    def _connectClassObj(self, clsObj: ClassObject) -> None:
        clsObj.btn.released.connect(self._emitClassChangedUpdate)
        clsObj.delBtn.released.connect(self._makeDelLambda(clsObj.name))
        clsObj.visCheckBox.stateChanged.connect(lambda: self.ClassVisibilityChanged.emit())

    def _activateClass(self, clsName: str) -> None:
        self._activeCls = clsName
        self.ClassActivated.emit(clsName)

    def _emitClassChangedUpdate(self) -> None:
        for cls in self._classes:
            if cls.isSelected():
                self._activateClass(cls.name)
                break

    def _makeDelLambda(self, name: str):
        return lambda: self._deleteGroup(name)

    def _deleteGroup(self, groupname: str) -> None:
        for cls in self._classes:
            if cls.name == groupname:
                self._classes.remove(cls)
                break

        self.ClassDeleted.emit(groupname)
        self._colorHandler.removeClass(groupname)
        self._recreateLayout()

    def _recreateLayout(self) -> None:
        for i in reversed(range(self._layout.count())):
            item = self._layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
            else:
                self._layout.removeItem(item)

        self._layout.addWidget(QtWidgets.QLabel("Select Classes:"))

        grid: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        group: QtWidgets.QGroupBox = QtWidgets.QGroupBox()
        group.setLayout(grid)
        grid.addWidget(QtWidgets.QLabel("Visible"), 0, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(QtWidgets.QLabel("Color"), 0, 1, QtCore.Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(QtWidgets.QLabel("Name"), 0, 2, QtCore.Qt.AlignmentFlag.AlignCenter)

        for i, cls in enumerate(self._classes):
            cls.recreateWidgets()
            self._connectClassObj(cls)
            activateWithoutActive = self._activeCls is None and i == 0
            activateWithActive = self._activeCls == cls.name
            if activateWithActive or activateWithoutActive:
                cls.btn.setChecked(True)

            grid.addWidget(cls.visCheckBox, i+2, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(self._getColorLabel(cls.name), i+2, 1, QtCore.Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(cls.btn, i+2, 2, QtCore.Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(cls.delBtn, i+2, 3, QtCore.Qt.AlignmentFlag.AlignCenter)
            
        self._layout.addWidget(group)
        self._layout.addWidget(self._newClsBtn)
        self._layout.addStretch()

    def _getColorLabel(self, name: str) -> QtWidgets.QLabel:
        """
        Creates a color patch label for the given class.
        :param name: the class name
        :return: QLabel
        """
        color: Tuple[int, int, int] = self.getColorOfClassName(name)
        img: np.ndarray = np.zeros((20, 20, 3), dtype=np.uint8)
        for i in range(3):
            img[:, :, i] = color[i]
        label: QtWidgets.QLabel = QtWidgets.QLabel()
        label.setPixmap(npy2Pixmap(img))
        return label


class ColorHandler:
    def __init__(self):
        self._name2color: Dict[str, Tuple[int, int, int]] = {}  # Dictionary storing the colors

    def removeClass(self, className: str) -> None:
        del self._name2color[className]

    def deleteAllClasses(self) -> None:
        self._name2color = {}

    def getColorOfClassName(self, className: str) -> Tuple[int, int, int]:
        if className not in self._name2color:
            self._defineNewColor(className)

        return self._name2color[className]

    def _defineNewColor(self, className: str) -> None:
        """
                Determines a new color and saves it to the dictionary.
                :param className: new name
                :return:
                """
        colorCycle = rcParams['axes.prop_cycle'].by_key()['color']
        newIndex: int = len(self._name2color)
        while newIndex >= len(colorCycle) - 1:
            newIndex -= len(colorCycle)

        color = colorCycle[newIndex]
        color = tuple([int(round(v * 255)) for v in to_rgb(color)])
        self._name2color[className] = color


@dataclass
class ClassInterpretationParams:
    specConfThreshold: float  # Spectra with a confidence lower than that are interpeted as "unknown"
    partConfThreshold: float  # If less than that fraction of all spectra within a particle are of the same class, it's "unknown"
    ignoreUnkowns: bool  # Whether or not to ignore unknown spectra when determining the particle's class


class ClassificationUI(QtWidgets.QGroupBox):
    """
    UI Element for Classification of the current graph view(s).
    """
    ClassTransparencyUpdated: QtCore.pyqtSignal = QtCore.pyqtSignal(float)
    ClassInterpretationParamsChanged: QtCore.pyqtSignal = QtCore.pyqtSignal(ClassInterpretationParams)

    def __init__(self, parent: 'MainWindow'):
        super(ClassificationUI, self).__init__()
        self._mainWin: 'MainWindow' = parent
        self._logger: 'Logger' = getLogger("ClassificationUI")

        self._clfSelector: ClfSelector = ClfSelector(parent)
        self._radioImage: QtWidgets.QRadioButton = QtWidgets.QRadioButton("Whole Image")
        self._radioParticles: QtWidgets.QRadioButton = QtWidgets.QRadioButton("Particles")

        self._applyBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Run Classifier Inference")

        self._transpSlider: QtWidgets.QSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._progressbar: Optional[QtWidgets.QProgressBar] = QtWidgets.QProgressBar()
        self._thread: Thread = Thread()
        self._stopEvent: Event = Event()
        self._timer: QtCore.QTimer = QtCore.QTimer()
        self._samplesFinished: int = 0  # Counter for tracking classifcation process

        self._spinParticleBinning: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
        self._spinSpecConf: QtWidgets.QDoubleSpinBox = QtWidgets.QDoubleSpinBox()
        self._spinPartConf: QtWidgets.QDoubleSpinBox = QtWidgets.QDoubleSpinBox()
        self._checkIgnoreUnknowns: QtWidgets.QCheckBox = QtWidgets.QCheckBox("Ignore 'Unknown'")

        self._layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)
        self._configureWidgets()
        self._createLayout()

    def getClassInterpretationParams(self) -> ClassInterpretationParams:
        """
        Returns a struct containing the current parameters for interpreting the classification results.
        """
        return ClassInterpretationParams(self._spinSpecConf.value(), self._spinPartConf.value(),
                                         self._checkIgnoreUnknowns.isChecked())

    def _runClassification(self) -> None:
        """
        Applies the classifier and runs classification on the selected samples.
        """
        self._mainWin.disableWidgets()

        inferenceSamples: List['SampleView'] = self._getInferenceSamples()
        errMsg: str = ''
        activeClf: Union[None, 'BaseClassifier'] = self._clfSelector.getActiveClassifier()
        if activeClf is None:
            errMsg = "No Classifier selected."
        if len(inferenceSamples) == 0:
            errMsg += "\nNo Samples for Inference selected"

        if len(errMsg) > 0:
            QtWidgets.QMessageBox.about(self, "Error", f"Cannot apply classifier:\n{errMsg}")
        else:
            self._mainWin.disableWidgets()
            clfMode: cp.ClassifyMode = cp.ClassifyMode.WholeImage if self._radioImage.isChecked() else cp.ClassifyMode.Particles
            self._progressbar = QtWidgets.QProgressDialog(f"Inference on {len(inferenceSamples)} samples.", "Abort",
                                                          0, len(inferenceSamples), parent=self)
            self._progressbar.setFixedWidth(300)
            self._progressbar.setMinimumDuration(0)
            self._progressbar.canceled.connect(self._cancelClassification)
            self._stopEvent = Event()
            self._samplesFinished = 0
            self._thread = Thread(target=cp.classifySamples, args=(inferenceSamples, activeClf, clfMode,
                                                                   self._mainWin.getPreprocessorsForClassification(),
                                                                   self._spinParticleBinning.value(),
                                                                   self._stopEvent, self._incrementClassificationCounter))
            self._timer.start(100)
            self._thread.start()

    def _incrementClassificationCounter(self) -> None:
        self._samplesFinished += 1

    def _cancelClassification(self) -> None:
        self._stopEvent.set()
        self._onClassificationFinishedOrAborted()

    def _checkOnClassification(self) -> None:
        if self._progressbar.value() < self._samplesFinished:
            self._progressbar.setValue(self._samplesFinished)

        if not self._thread.is_alive():
            self._onClassificationFinishedOrAborted()

    def _onClassificationFinishedOrAborted(self) -> None:
        self._timer.stop()
        self._mainWin.enableWidgets()

    def _emitTransparencyUpdate(self) -> None:
        self.ClassTransparencyUpdated.emit(self._transpSlider.value() / 100)

    def _getInferenceSamples(self) -> List['SampleView']:
        """
        Returns a list of SampleData objects from all samples selected for inference.
        """
        return [sample for sample in self._mainWin.getAllSamples() if sample.isSelectedForInference()]

    def _configureWidgets(self) -> None:
        self._transpSlider.setMinimum(0)
        self._transpSlider.setValue(80)
        self._transpSlider.setMaximum(100)
        self._transpSlider.valueChanged.connect(self._emitTransparencyUpdate)

        self._applyBtn.released.connect(self._runClassification)

        self._spinParticleBinning.setMinimum(1)
        self._spinParticleBinning.setMaximum(100000)
        self._spinParticleBinning.setValue(1)
        self._spinParticleBinning.setToolTip("Binning during particle inference.\n"
                                             "Within the spectra of a particle, n of the particle spectra are averaged.")

        self._radioParticles.setChecked(True)
        
        for spinbox in [self._spinSpecConf, self._spinPartConf]:
            spinbox.setMinimum(0.0)
            spinbox.setMaximum(1.0)
            spinbox.setSingleStep(0.1)
            spinbox.valueChanged.connect(self._emitClassInterpParamsUpdate)
        
        self._spinPartConf.setValue(0.5)
        self._spinPartConf.setToolTip("Required confidence for infering a particle class.\n"
                                      "Of the n spectra within the particles, the most abundand class needs to represent\n"
                                      "at least the indicated fraction of all spectra.\n"
                                      "If the most abundand class is less abundant, then the particle is labelled 'unknown'.")
        self._spinSpecConf.setValue(0.75)
        self._spinSpecConf.setToolTip("Required confidence for classifying a spectrum.\n"
                                      "The highest class probability needs to be at least the indicated value, otherwise\n"
                                      "the spectrum is labelled as 'unknown'")

        self._checkIgnoreUnknowns.stateChanged.connect(self._emitClassInterpParamsUpdate)
        self._checkIgnoreUnknowns.setToolTip("Whether or not to ignore 'unknown' spectra while deriving a particle's class.")

        self._timer.timeout.connect(self._checkOnClassification)
        self._timer.setSingleShot(False)

    def _createLayout(self) -> None:
        layout = self._layout

        infOptnGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox("Run classification on")
        infOptnLayout: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        infOptnGroup.setLayout(infOptnLayout)
        infOptnLayout.addWidget(self._radioImage, 0, 0)
        infOptnLayout.addWidget(self._radioParticles, 0, 1)
        infOptnLayout.addWidget(QtWidgets.QLabel("Particle Binning"), 1, 0)
        infOptnLayout.addWidget(self._spinParticleBinning, 1, 1)

        confidenceGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox("Classifcation Options")
        confidenceLayout: QtWidgets.QFormLayout = QtWidgets.QFormLayout()
        confidenceGroup.setLayout(confidenceLayout)
        confidenceLayout.addRow("Spectra Confidence", self._spinSpecConf)
        confidenceLayout.addRow("Particle Confidence", self._spinPartConf)
        confidenceLayout.addRow(self._checkIgnoreUnknowns)

        layout.addWidget(self._clfSelector)
        layout.addStretch()
        layout.addWidget(infOptnGroup)
        layout.addWidget(self._applyBtn)
        layout.addWidget(confidenceGroup)
        layout.addStretch()
        layout.addWidget(QtWidgets.QLabel("Set Overlay Transparency"))
        layout.addWidget(self._transpSlider)

        # scrollArea: QtWidgets.QScrollArea = QtWidgets.QScrollArea()
        # scrollArea.setWidget(contentsGroup)
        #
        # selfLayout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        # self.setLayout(selfLayout)
        # selfLayout.addWidget(scrollArea)

    def _emitClassInterpParamsUpdate(self) -> None:
        """
        Sends an update when the spectra or particle confidence settings were changed.
        Will be connected to methods updating the classification results.
        """
        newConf: ClassInterpretationParams = ClassInterpretationParams(self._spinSpecConf.value(),
                                                                       self._spinPartConf.value(),
                                                                       self._checkIgnoreUnknowns.isChecked())
        self.ClassInterpretationParamsChanged.emit(newConf)


class ClfSelector(QtWidgets.QGroupBox):
    """
    Group for selecting a classifier.
    """
    def __init__(self, mainWin: 'MainWindow'):
        super(ClfSelector, self).__init__("Select Classifier")
        self._activeClf: Union[None, 'BaseClassifier'] = None  # the currently selected classifier
        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self._tabView: QtWidgets.QTabWidget = QtWidgets.QTabWidget()
        self._loadClfTab: LoadClfTab = LoadClfTab()
        self._trainClfTab: TrainClfTab = TrainClfTab(mainWin)

        self._loadIndex: int = self._tabView.addTab(self._loadClfTab, "Load Classifier")
        self._trainIndex: int = self._tabView.addTab(self._trainClfTab, "Train Classifier")

        self._validResult: ValidationResult = ValidationResult()
        self._tabView.currentChanged.connect(self._onTabChanged)
        self._trainClfTab.NewValidationResult.connect(self._validResult.showResult)
        self._loadClfTab.NewValidationResult.connect(self._validResult.showResult)

        # scrollArea: QtWidgets.QScrollArea = QtWidgets.QScrollArea()
        # scrollArea.setWidget(self._validResult)  # TODO: Consider reimplementing scrollarea..

        layout.addWidget(self._tabView)
        layout.addWidget(self._validResult)

    def getActiveClassifier(self) -> Union[None, 'BaseClassifier']:
        """
        Returns the active classifier from either the loading or the training tab, depending on which is currently opened.
        """
        clf: Union[None, 'BaseClassifier'] = None
        if self._tabView.currentIndex() == self._loadIndex:
            clf = self._loadClfTab.getActiveClf()
        elif self._tabView.currentIndex() == self._trainIndex:
            clf = self._trainClfTab.getActiveClf()

        return clf

    @QtCore.pyqtSlot(int)
    def _onTabChanged(self) -> None:
        """Called, when the training or loading tab is changed"""
        self._validResult.clearResult()
        self._tabView.currentWidget().onTabActivated()


class LoadClfTab(QtWidgets.QWidget):
    """
    Tab for loading an already trained classifier from disk.
    """
    NewValidationResult: QtCore.pyqtSignal = QtCore.pyqtSignal(dict)

    def __init__(self):
        super(LoadClfTab, self).__init__()
        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        loadBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Load")
        loadBtn.released.connect(self._promptForLoading)
        self._clfLabel: QtWidgets.QLabel = QtWidgets.QLabel("No classifier loaded")

        layout.addWidget(loadBtn)
        layout.addWidget(self._clfLabel)

        self._activeClf: Optional['BaseClassifier'] = None
        self._currentresult: Optional[dict] = None

    def onTabActivated(self) -> None:
        if self._activeClf is not None and self._currentresult is not None:
            self.NewValidationResult.emit(self._currentresult)

    def getActiveClf(self) -> Union[None, 'BaseClassifier']:
        return self._activeClf

    def _promptForLoading(self) -> None:
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select classifier to load",
                                                         directory=getClassifierSaveFolder(), filter="*clf")
        if fname:
            self._loadClassifier(fname)

    def _loadClassifier(self, fname) -> None:
        """
        Loads the classifier from the specified path.
        :param fname: Absolute path to pickled classifier.
        """
        with open(fname, "rb") as fp:
            savedClf: SavedClassifier = pickle.load(fp)
        self.NewValidationResult.emit(savedClf.validReport)

        self._currentresult = savedClf.validReport
        self._activeClf = savedClf.clf
        self._activeClf.afterLoad()
        clfName: str = os.path.basename(fname).split(".")[0]
        self._clfLabel.setText(f"Loaded '{clfName}'")


class TrainClfTab(QtWidgets.QWidget):
    """
    Tab for training a new classifier.
    """
    NewValidationResult: QtCore.pyqtSignal = QtCore.pyqtSignal(dict)

    def __init__(self, mainWin: 'MainWindow'):
        super(TrainClfTab, self).__init__()
        self._layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        self._mainWin: 'MainWindow' = mainWin
        self._logger: 'Logger' = getLogger("ClassifierTraining")
        self._activeClfControls: QtWidgets.QGroupBox = QtWidgets.QGroupBox("No Classifier Selected!")
        self._testFracSpinBox: QtWidgets.QDoubleSpinBox = QtWidgets.QDoubleSpinBox()
        self._currentTrainResult: Optional[dict] = None

        self._thread: Thread = Thread()
        self._stopEvent: Event = Event()
        self._timer: QtCore.QTimer = QtCore.QTimer()
        self._trainResult: Union[None, 'cp.TrainingResult'] = None
        self._progressBar: Union[QtWidgets.QProgressDialog] = None

        self._maxNumSpecsSpinBox: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
        self._balanceMethodComboBox: QtWidgets.QComboBox = QtWidgets.QComboBox()

        self._clfCombo: QtWidgets.QComboBox = QtWidgets.QComboBox()
        self._trainBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Train Classifier")
        self._saveBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Save Classifier")

        self._activeClf: Union[None, 'BaseClassifier'] = None
        self._classifiers: List['BaseClassifier'] = cp.getClassifiers()  # all available classifiers

        self._configureWidgets()
        self._createLayout()
        self._selectClassifier(self._classifiers[0].title)

    def onTabActivated(self) -> None:
        if self._activeClf is not None and self._currentTrainResult is not None:
            self.NewValidationResult.emit(self._currentTrainResult)

    def getActiveClf(self) -> Union[None, 'BaseClassifier']:
        return self._activeClf

    def _configureWidgets(self) -> None:
        self._timer.timeout.connect(self._checKOnTraining)
        self._timer.setSingleShot(False)

        self._clfCombo.addItems([clf.title for clf in self._classifiers])
        self._clfCombo.currentTextChanged.connect(self._selectClassifier)
        self._testFracSpinBox.setMinimum(0.01)
        self._testFracSpinBox.setMaximum(0.99)
        self._testFracSpinBox.setValue(0.1)

        self._maxNumSpecsSpinBox.setMinimum(100)
        self._maxNumSpecsSpinBox.setMaximum(int(1e6))
        self._maxNumSpecsSpinBox.setValue(50000)

        balanceModes: List[str] = list(cp.BalanceMode.__members__.keys())
        self._balanceMethodComboBox.addItems(balanceModes)

        self._trainBtn.released.connect(self._trainClassifier)
        self._saveBtn.released.connect(self._promptToSaveClf)

    def _createLayout(self) -> None:
        trainOptnGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox("Training Options")
        trainOptnLayout: QtWidgets.QFormLayout = QtWidgets.QFormLayout()
        trainOptnGroup.setLayout(trainOptnLayout)
        trainOptnLayout.addRow("Max. Num. of Spectra per class", self._maxNumSpecsSpinBox)
        trainOptnLayout.addRow("Test Fraction", self._testFracSpinBox)
        trainOptnLayout.addRow("Balancing", self._balanceMethodComboBox)

        btnLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        btnLayout.addWidget(self._trainBtn)
        btnLayout.addWidget(self._saveBtn)

        self._layout.addWidget(self._clfCombo)
        self._layout.addWidget(self._activeClfControls)
        self._layout.addWidget(trainOptnGroup)
        self._layout.addLayout(btnLayout)

    @QtCore.pyqtSlot(str)
    def _selectClassifier(self, clfName: str) -> None:
        """
        Executed when a new classifier is selected. Sets up the UI accordingly.
        """
        clfActivated: bool = False
        for clf in self._classifiers:
            if clf.title == clfName:
                self._activeClf = clf
                self._activeClfControls = clf.getControls()
                clfActivated = True
                break
        assert clfActivated, f'Classifier {clfName} was not found in available classifiers...'
        self._placeClfControlsToLayout()

    def _trainClassifier(self) -> None:
        """
        Trains the selected classifier.
        """
        trainSamples: List['Sample'] = self._getTrainingSamples()
        errMsg: str = ''
        if self._activeClf is None:
            errMsg = "No Classifier selected."
        if len(trainSamples) == 0:
            errMsg += "\nNo Samples for Training selected"

        if len(errMsg) > 0:
            QtWidgets.QMessageBox.about(self, "Error", f"Cannot train classifier:\n{errMsg}")
        else:
            self._mainWin.disableWidgets()
            if self._balanceMethodComboBox.currentText() == "NoBalancing":
                balanceMode: cp.BalanceMode = cp.BalanceMode.NoBalancing
            elif self._balanceMethodComboBox.currentText() == "UnderRandom":
                balanceMode: cp.BalanceMode = cp.BalanceMode.UnderRandom
            elif self._balanceMethodComboBox.currentText() == "UnderNearMiss":
                balanceMode: cp.BalanceMode = cp.BalanceMode.UnderNearMiss
            elif self._balanceMethodComboBox.currentText() == "OverRandom":
                balanceMode: cp.BalanceMode = cp.BalanceMode.OverRandom
            elif self._balanceMethodComboBox.currentText() == "OverSMOTE":
                balanceMode: cp.BalanceMode = cp.BalanceMode.OverSMOTE

            self._trainResult = None
            self._stopEvent = Event()
            self._thread = Thread(target=cp.trainClassifier, args=(trainSamples, self._activeClf,
                                                                   self._mainWin.getPreprocessorsForClassification(),
                                                                   self._maxNumSpecsSpinBox.value(),
                                                                   self._testFracSpinBox.value(),
                                                                   balanceMode,
                                                                   self._stopEvent,
                                                                   self._receiveTrainResult))

            self._progressBar = QtWidgets.QProgressDialog(f"Training on {len(trainSamples)} Samples", "Abort", 0, 0, parent=self)
            self._progressBar.setMinimumDuration(0)
            self._progressBar.canceled.connect(self._cancelTraining)
            self._progressBar.setFixedWidth(300)
            self._timer.start(100)
            self._thread.start()

    def _promptToSaveClf(self) -> None:
        """
        Prompts the user for a file location to save the active classifier to.
        """
        ending: str = 'clf'
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select Location to save classfier to",
                                                         directory=getClassifierSaveFolder(), filter=f"*{ending}")
        if not fname.endswith(ending):
            fname += f".{ending}"
        if fname:
            self._saveClassifier(fname)

    def _saveClassifier(self, fname: str) -> None:
        """
        Saves the active classifier to the specified directory.
        :param fname: Full path to classifier save file.
        """
        assert self._activeClf is not None
        assert self._currentTrainResult is not None
        self._activeClf.makePickleable(fname)
        saveClf: SavedClassifier = SavedClassifier(self._activeClf, self._currentTrainResult)
        with open(fname, "wb") as fp:
            pickle.dump(saveClf, fp)

        self._activeClf.restoreNotPickleable()

    def _checKOnTraining(self) -> None:
        """
        Called during training to check if it is finished.
        """
        if not self._thread.is_alive():
            self._onTrainingFinishedOrAborted()

    def _cancelTraining(self) -> None:
        self._stopEvent.set()
        self._thread.join()
        self._onTrainingFinishedOrAborted()

    def _onTrainingFinishedOrAborted(self) -> None:
        if self._trainResult is None:
            self._logger.info("Training finished without getting a result.")
        else:
            self._trainResult = cast(cp.TrainingResult, self._trainResult)
            self._currentTrainResult = self._trainResult.validReportDict
            self.NewValidationResult.emit(self._trainResult.validReportDict)

        self._progressBar.hide()
        self._timer.stop()
        self._mainWin.enableWidgets()

    def _receiveTrainResult(self, trainResult: cp.TrainingResult) -> None:
        """
        Receives the result from the training Thread
        """
        self._trainResult = trainResult

    def _placeClfControlsToLayout(self) -> None:
        """Places the controls of the currently selected classifier into the layout."""
        indexOfControlElement: int = 1  # has to be matched with the layout contruction
        item = self._layout.itemAt(indexOfControlElement)
        self._layout.removeWidget(item.widget())
        self._layout.insertWidget(indexOfControlElement, self._activeClfControls)

    def _getTrainingSamples(self) -> List['Sample']:
        """
        Returns a list of SampleData objects from all samples selected for training.
        """
        return [sample.getSampleData() for sample in self._mainWin.getAllSamples() if sample.isSelectedForTraining()]


class ValidationResult(QtWidgets.QGroupBox):
    """
    Widget for showing classifier validation statistics.
    """
    def __init__(self):
        super(ValidationResult, self).__init__("Classifier Statistics")
        self._currentResults: dict = {}

        self._layout: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        self._defaultLabel: QtWidgets.QLabel = QtWidgets.QLabel("No results yet.\t\t")
        self._lblPrec: QtWidgets.QLabel = QtWidgets.QLabel("Precision")
        self._lblRecall: QtWidgets.QLabel = QtWidgets.QLabel("Recall")
        self._lblF1: QtWidgets.QLabel = QtWidgets.QLabel("F1")
        self._lblSupport: QtWidgets.QLabel = QtWidgets.QLabel("Support")
        self._lblAccuracy: QtWidgets.QLabel = QtWidgets.QLabel("Accuracy")
        self._lblMacroAvg: QtWidgets.QLabel = QtWidgets.QLabel("Macro Avg")
        self._lblWeightedAvg: QtWidgets.QLabel = QtWidgets.QLabel("Weighted Avg")
        self._resultLabels: List[QtWidgets.QLabel] = []
        self._layout.addWidget(self._defaultLabel, 0, 0)
        self.setLayout(self._layout)

    @QtCore.pyqtSlot()
    def clearResult(self) -> None:
        """
        Clears the current result and displays a standard text.
        """
        self._clearLayout()
        self._resultLabels = []
        self._currentResults = {}

        self._defaultLabel.setText("No valid classifier loaded or trained")
        self._layout.addWidget(self._defaultLabel, 0, 0)

    @QtCore.pyqtSlot(dict)
    def showResult(self, reportDict: dict) -> None:
        """
        Takes a validation report dict and adapts to show its content.
        """
        self._clearLayout()
        self._defaultLabel.setText("")
        self._resultLabels = []
        self._currentResults = reportDict

        # Header
        for i, lbl in enumerate(self._getHeaderLabels(), start=1):
            self._layout.addWidget(lbl, 0, i)
        clsList: List[str] = [key for key in reportDict.keys() if key not in ["accuracy", "macro avg", "weighted avg"]]
        for row, cls in enumerate(clsList, start=1):
            clsDict: Dict[str, float] = reportDict[cls]
            newLbl: QtWidgets.QLabel = QtWidgets.QLabel(cls)
            self._resultLabels.append(newLbl)
            self._layout.addWidget(newLbl, row, 0)
            for col, entry in enumerate(["precision", "recall", "f1-score", "support"], start=1):
                newLbl: QtWidgets.QLabel = QtWidgets.QLabel(str(round(clsDict[entry], 2)))
                self._resultLabels.append(newLbl)
                self._layout.addWidget(newLbl, row, col)

        row += 1
        self._layout.addWidget(self._defaultLabel, row, 0)  # the empty default label as blank line

        row += 1  # the accuracy row
        self._layout.addWidget(self._lblAccuracy, row, 0)
        accLbl: QtWidgets.QLabel = QtWidgets.QLabel(str(round(reportDict["accuracy"], 2)))
        suppLbl: QtWidgets.QLabel = QtWidgets.QLabel(str(round(reportDict["macro avg"]["support"], 2)))
        self._layout.addWidget(accLbl, row, 3)
        self._layout.addWidget(suppLbl, row, 4)
        for lbl in [accLbl, suppLbl]:
            self._resultLabels.append(lbl)

        row += 1  # now the macro avg row
        self._layout.addWidget(self._lblMacroAvg, row, 0)
        macroAvgDict: Dict[str, float] = reportDict["macro avg"]
        for col, entry in enumerate(["precision", "recall", "f1-score", "support"], start=1):
            newLbl: QtWidgets.QLabel = QtWidgets.QLabel(str(round(macroAvgDict[entry], 2)))
            self._resultLabels.append(newLbl)
            self._layout.addWidget(newLbl, row, col)

        row += 1  # now the weighted avg row
        self._layout.addWidget(self._lblWeightedAvg, row, 0)
        weightedAvgDict: Dict[str, float] = reportDict["weighted avg"]
        for col, entry in enumerate(["precision", "recall", "f1-score", "support"], start=1):
            newLbl: QtWidgets.QLabel = QtWidgets.QLabel(str(round(weightedAvgDict[entry], 2)))
            self._resultLabels.append(newLbl)
            self._layout.addWidget(newLbl, row, col)

    def _getHeaderLabels(self) -> List[QtWidgets.QLabel]:
        return [self._lblPrec, self._lblRecall, self._lblF1, self._lblSupport]

    def _getAllLabels(self) -> List[QtWidgets.QLabel]:
        return self._resultLabels + self._getHeaderLabels() + [self._defaultLabel] + self._getRowLabels()

    def _getRowLabels(self) -> List[QtWidgets.QLabel]:
        return [self._lblAccuracy, self._lblMacroAvg, self._lblWeightedAvg]

    def _clearLayout(self) -> None:
        for lbl in self._getAllLabels():
            self._layout.removeWidget(lbl)


if __name__ == '__main__':
    import sys
    from sklearn.metrics import classification_report
    app = QtWidgets.QApplication(sys.argv)
    win = ValidationResult()
    win.show()
    a = ['bus'] * 5 + ['auto'] * 3 + ['fahrrad'] * 9
    b = ['bus'] * 4 + ['auto'] * 4 + ['fahrrad'] * 9
    rep = classification_report(a, b, output_dict=True)
    win.showResult(rep)

    a = ['n??'] * 15 + ['doch'] * 13 + ['vielleicht'] * 19
    b = ['n??'] * 14 + ['doch'] * 14 + ['vielleicht'] * 19
    rep = classification_report(a, b, output_dict=True)
    win.showResult(rep)
    app.exec_()
