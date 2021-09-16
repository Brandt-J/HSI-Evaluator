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

from PyQt5 import QtWidgets, QtCore
from typing import List, Tuple, Dict, Union, TYPE_CHECKING, cast
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.pyplot import rcParams
from multiprocessing import Process, Queue

from logger import getLogger
from gui.graphOverlays import npy2Pixmap
from dataObjects import Sample
from classification.classifyProcedures import getClassifiers, trainAndClassify
from classification.classifiers import ClassificationError, BaseClassifier

if TYPE_CHECKING:
    from gui.HSIEvaluator import MainWindow
    from gui.sampleview import SampleView
    from gui.graphOverlays import GraphView
    from preprocessing.preprocessors import Preprocessor
    from spectraObject import SpectraObject
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

    # def setClasses(self, classes: List[str]) -> None:
    #     self._classes = classes
    #     self._update()

    def activateClass(self, clsName: str) -> None:
        if clsName in self._classes:
            self._activeCls = clsName
            self.ClassActivated.emit(clsName)

    def checkForRequiredClasses(self, classes: List[str]) -> None:
        """
        Creates classes that are needed if they aren't yet present.
        """
        presentClassNames: List[str] = [clsObj.name for clsObj in self._classes]
        for cls in classes:
            if cls not in presentClassNames:
                self._addClass(cls)
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
        colorDict: Dict[str, Tuple[int, int, int]] = {}
        for cls in self._classes:
            colorDict[cls.name] = self._colorHandler.getColorOfClassName(cls.name)
        return colorDict

    def getColorOfClassName(self, className: str) -> Tuple[int, int, int]:
        return self._colorHandler.getColorOfClassName(className)
    
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

        self._recreateLayout()
        self.ClassDeleted.emit(groupname)

    def _recreateLayout(self) -> None:
        for i in reversed(range(self._layout.count())):
            widget = self._layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

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

    def getColorOfClassName(self, groupName: str) -> Tuple[int, int, int]:
        if groupName not in self._name2color:
            self._defineNewColor(groupName)

        return self._name2color[groupName]

    def _defineNewColor(self, name: str) -> None:
        """
        Determines a new color and saves it to the dictionary.
        :param name: new name
        :return:
        """
        colorCycle = rcParams['axes.prop_cycle'].by_key()['color']
        newIndex: int = len(self._name2color)
        if newIndex > len(colorCycle)-1:
            newIndex -= len(colorCycle)
            
        color = colorCycle[newIndex]
        color = tuple([int(round(v * 255)) for v in to_rgb(color)])
        self._name2color[name] = color


class ClassificationUI(QtWidgets.QGroupBox):
    """
    UI Element for Classification of the current graph view(s).
    """
    ClassTransparencyUpdated: QtCore.pyqtSignal = QtCore.pyqtSignal(float)

    def __init__(self, parent: 'MainWindow'):
        super(ClassificationUI, self).__init__()
        self._parent: 'MainWindow' = parent
        self._logger: 'Logger' = getLogger("ClassificationUI")
        self._preprocessingRequired: bool = True
        self._classifiers: List['BaseClassifier'] = getClassifiers()  # all available classifiers
        self._activeClf: Union[None, 'BaseClassifier'] = None  # the currently selected classifier
        self._samplesToClassify: List['SampleView'] = []  # List for keeping track of opened samples to classify
        self._activeClfControls: QtWidgets.QGroupBox = QtWidgets.QGroupBox("No Classifier Selected!")

        self._excludeBackgroundCheckbox: QtWidgets.QCheckBox = QtWidgets.QCheckBox()
        self._testFracSpinBox: QtWidgets.QDoubleSpinBox = QtWidgets.QDoubleSpinBox()
        self._updateBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Update Classification")
        self._transpSlider: QtWidgets.QSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._clfCombo: QtWidgets.QComboBox = QtWidgets.QComboBox()
        self._progressbar: QtWidgets.QProgressBar = QtWidgets.QProgressBar()
        self._progressbar.setWindowTitle("Classification in Progress")
        self._validationLabel: QtWidgets.QLabel = QtWidgets.QLabel()

        self._process: Process = Process()
        self._queue: Queue = Queue()
        self._timer: QtCore.QTimer = QtCore.QTimer()
        self._timer.setSingleShot(False)
        self._timer.timeout.connect(self._checkOnComputation)

        self._layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)
        self._configureWidgets()
        self._createLayout()
        self._selectFirstClassifier()

    def _classifyImage(self) -> None:
        """
        Classifies the current image based on the selected pixels and their classes
        :return:
        """
        self._parent.disableWidgets()
        preprocessors: List['Preprocessor'] = self._parent.getPreprocessors()
        trainingSamples: List['SampleView'] = self._getTrainSamples()
        self._samplesToClassify = self._getInferenceSamples()
        allBackgrounds: Dict[str, np.ndarray] = self._parent.getBackgroundsOfAllSamples()

        trainSampleDataList: List['Sample'] = []
        inferenceSampleDataList: List['Sample'] = []
        for i, sample in enumerate(set(self._samplesToClassify + trainingSamples)):
            if sample in trainingSamples:
                trainSampleDataList.append(sample.getSampleData())
            if sample in self._samplesToClassify:
                inferenceSampleDataList.append(sample.getSampleData())
                sample.resetClassificationOverlay()

            specObj: 'SpectraObject' = sample.getSpecObj()
            specObj.preparePreprocessing(preprocessors, allBackgrounds[sample.getName()], sample.getBackgroundPixelIndices())

        assert len(trainSampleDataList) > 0 and len(self._samplesToClassify) > 0, 'Either no training or no inference data..'

        self._progressbar.show()
        self._progressbar.setValue(0)
        self._progressbar.setMaximum(0)  # TODO: Implement a more fine-grained monitoring
        # self._progressbar.setMaximum(len(self._samplesToClassify))
        self._activeClf.makePickleable()
        self._queue = Queue()  # recreate queue object, it could have been closed previously.
        self._process = Process(target=trainAndClassify, args=(trainSampleDataList,
                                                               inferenceSampleDataList,
                                                               self._preprocessingRequired,
                                                               self._excludeBackgroundCheckbox.isChecked(),
                                                               self._activeClf,
                                                               self._testFracSpinBox.value(),
                                                               self._parent.getClassColorDict(),
                                                               self._queue))
        self._process.start()
        self._timer.start()
        self._activeClf.restoreNotPickleable()

    def _emitTransparencyUpdate(self) -> None:
        self.ClassTransparencyUpdated.emit(self._transpSlider.value() / 100)

    @QtCore.pyqtSlot(str)
    def _activateClassifier(self, clfName: str) -> None:
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

    def _selectFirstClassifier(self) -> None:
        """
        To select the first classifier.
        """
        if len(self._classifiers) > 0:
            self._activateClassifier(self._classifiers[0].title)

    @QtCore.pyqtSlot()
    def forcePreprocessing(self) -> None:
        """
        Called when the preprocessing stack was updated or the background selection has changed...
        """
        self._preprocessingRequired = True

    def _configureWidgets(self) -> None:
        self._transpSlider.setMinimum(0)
        self._transpSlider.setValue(80)
        self._transpSlider.setMaximum(100)
        self._transpSlider.valueChanged.connect(self._emitTransparencyUpdate)

        self._testFracSpinBox.setMinimum(0.01)
        self._testFracSpinBox.setMaximum(0.99)
        self._testFracSpinBox.setValue(0.1)

        self._updateBtn.released.connect(self._classifyImage)

        self._excludeBackgroundCheckbox.setChecked(True)
        self._validationLabel.setText("Not yet validated.")

        self._clfCombo.addItems([clf.title for clf in self._classifiers])
        self._clfCombo.currentTextChanged.connect(self._activateClassifier)

    def _createLayout(self) -> None:
        self._layout.addWidget(QtWidgets.QLabel("Select Classifier:"))
        self._layout.addWidget(self._clfCombo)
        self._layout.addWidget(self._activeClfControls)
        self._layout.addStretch()

        optnGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox("Options:")
        optnLayout: QtWidgets.QFormLayout = QtWidgets.QFormLayout()
        optnGroup.setLayout(optnLayout)
        optnLayout.addRow("Test Fraction", self._testFracSpinBox)
        optnLayout.addRow("Exlude Background", self._excludeBackgroundCheckbox)
        optnLayout.addRow(self._updateBtn)
        self._layout.addWidget(optnGroup)
        self._layout.addStretch()
        validationGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox("Validation Result")
        validationGroup.setLayout(QtWidgets.QHBoxLayout())
        validationGroup.layout().addWidget(self._validationLabel)
        self._layout.addWidget(validationGroup)
        self._layout.addWidget(QtWidgets.QLabel("Set Overlay Transparency"))
        self._layout.addWidget(self._transpSlider)

    def _getTrainSamples(self) -> List['SampleView']:
        """Gets a list of the samples that shall be used for classifier training."""
        trainSamples: List['SampleView'] = []
        for sample in self._parent.getAllSamples():
            if sample.isSelectedForTraining():
                trainSamples.append(sample)
        return trainSamples

    def _getInferenceSamples(self) -> List['SampleView']:
        """Gets a list of the samples that shall be used for classifier inference."""
        trainSamples: List['SampleView'] = []
        for sample in self._parent.getAllSamples():
            if sample.isSelectedForInference():
                trainSamples.append(sample)
        return trainSamples

    def _placeClfControlsToLayout(self) -> None:
        """Places the controls of the currently selected classifier into the layout."""
        indexOfControlElement: int = 2  # has to be matched with the layout contruction in the _createLayout method.
        item = self._layout.itemAt(indexOfControlElement)
        self._layout.removeWidget(item.widget())
        self._layout.insertWidget(indexOfControlElement, self._activeClfControls)

    def _checkOnComputation(self) -> None:
        """
        Checks the state of computation and updates the interface accordingly.
        """
        if not self._queue.empty():
            queueContent = self._queue.get()
            errorOccured: bool = False
            if type(queueContent) is ClassificationError:
                error: ClassificationError = cast(ClassificationError, queueContent)
                QtWidgets.QMessageBox.critical(self, "Error in classification", f"The following error occured:\n"
                                                                                f"{error.errorText}")
                errorOccured = True

            elif type(queueContent) == str:
                self._setValidationResult(cast(str, queueContent))

            elif type(queueContent) == Sample:
                finishedData: 'Sample' = cast('Sample', queueContent)
                self._updateClassifiedSample(finishedData)
                # self._progressbar.setValue(self._progressbar.value()+1)  # TODO: Implement a more fine-grained monitoring...

            if len(self._samplesToClassify) == 0 or errorOccured:
                self._finishComputation()
                # if errorOccured:
                #     raise ClassificationError("An error during classification occurred")

    def _updateClassifiedSample(self, finishedData: 'Sample') -> None:
        """
        Takes Sample data from finished classification and updates the according sampleView.
        """
        for sample in self._samplesToClassify:
            if sample.getName() == finishedData.name:
                graphView: 'GraphView' = sample.getGraphView()
                graphView.updateClassImage(finishedData.classOverlay)
                self._samplesToClassify.remove(sample)
                break

    def _setValidationResult(self, report: str) -> None:
        """
        Adjusts the validation lable with the latest classification report.
        """
        self._validationLabel.setText(report)

    def _finishComputation(self) -> None:
        self._preprocessingRequired = False
        self._progressbar.hide()
        self._parent.enableWidgets()
        self._timer.stop()
        self._queue.close()
        self._process.join()
