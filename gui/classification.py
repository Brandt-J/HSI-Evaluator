from PyQt5 import QtWidgets, QtCore
from typing import List, Tuple, Dict, cast, Union, TYPE_CHECKING
import numpy as np
import time
from matplotlib.colors import to_rgb
from matplotlib.pyplot import rcParams

from SpectraProcessing.classification import NeuralNetClassifier, RandomDecisionForest
from logger import getLogger
from gui.graphOverlays import npy2Pixmap

if TYPE_CHECKING:
    from gui.HSIEvaluator import MainWindow
    from SpectraProcessing.classification import BaseClassifier
    from gui.graphOverlays import GraphView
    from spectraObject import SpectraObject
    from logging import Logger


class ClassCreator(QtWidgets.QGroupBox):
    ClassCreated: QtCore.pyqtSignal = QtCore.pyqtSignal()
    ClassDeleted: QtCore.pyqtSignal = QtCore.pyqtSignal(str)
    ClassActivated: QtCore.pyqtSignal = QtCore.pyqtSignal(str)

    def __init__(self):
        super(ClassCreator, self).__init__()
        self._classes: List[str] = ["Background"]
        self._colorHandler: ColorHandler = ColorHandler()
        self._logger: 'Logger' = getLogger("ClassCreator")
        self._activeCls: Union[None, str] = None
        self._btns: List[QtWidgets.QRadioButton] = []
        self._delBtns: List[QtWidgets.QPushButton] = []
        self._newClsBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("New Class")
        self._newClsBtn.released.connect(self._createNewClass)
        self._layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)
        self._update()

    def setClasses(self, classes: List[str]) -> None:
        self._classes = classes
        self._update()

    def activateClass(self, clsName: str) -> None:
        if clsName in self._classes:
            self._activeCls = clsName
            self._update()
            self.ClassActivated.emit(clsName)

    def getCurrentColor(self) -> Tuple[int, int, int]:
        color: Union[None, tuple] = None
        for btn in self._btns:
            if btn.isChecked():
                color = self.getColorOfClassName(btn.text())
                break
        assert color is not None
        return color

    def getCurrentClass(self) -> str:
        name: str = ''
        for btn in self._btns:
            if btn.isChecked():
                name = btn.text()
                break
        return name

    def getClassNamesAndColors(self) -> Tuple[List[str], List[Tuple[int, int, int]]]:
        colors = [self.getColorOfClassName(cls) for cls in self._classes]
        return self._classes.copy(), colors

    def getColorOfClassName(self, className: str) -> Tuple[int, int, int]:
        return self._colorHandler.getColorOfClassName(className)

    def _createNewClass(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Create New Class", "Class Name:")
        if ok:
            self._classes.append(name)
            self._activateClass(name)
            self._update()

    def _activateClass(self, clsName: str) -> None:
        self._activeCls = clsName
        self.ClassActivated.emit(clsName)

    def _updateButtons(self) -> None:
        self._btns, self._delBtns = [], []
        for i, cls in enumerate(self._classes):
            btn: QtWidgets.QRadioButton = QtWidgets.QRadioButton(cls)
            activateWithoutActive = self._activeCls is None and i == 0
            activateWithActive = self._activeCls == cls
            if activateWithActive or activateWithoutActive:
                btn.setChecked(True)
            btn.released.connect(self._emitClassChangedUpdate)
            self._btns.append(btn)
            delBtn: QtWidgets.QPushButton = QtWidgets.QPushButton()
            delBtn.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogCloseButton')))
            delBtn.setFixedWidth(20)
            delBtn.released.connect(self._makeDelLambda(cls))
            self._delBtns.append(delBtn)

    def _emitClassChangedUpdate(self) -> None:
        for btn in self._btns:
            if btn.isChecked():
                self._activateClass(btn.text())
                break

    def _makeDelLambda(self, name: str):
        return lambda: self._deleteGroup(name)

    def _deleteGroup(self, groupname: str) -> None:
        self._classes.remove(groupname)
        self._update()
        self.ClassDeleted.emit(groupname)

    def _update(self) -> None:
        self._updateButtons()
        self._updateLayout()

    def _updateLayout(self) -> None:
        for i in reversed(range(self._layout.count())):
            widget = self._layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        self._layout.addWidget(QtWidgets.QLabel("Select Class:"))
        grid: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        group: QtWidgets.QGroupBox = QtWidgets.QGroupBox()
        group.setLayout(grid)
        for i, (btn, delBtn) in enumerate(zip(self._btns, self._delBtns), start=1):
            grid.addWidget(self._getColorLabel(btn.text()), i, 0)
            grid.addWidget(btn, i, 1)
            grid.addWidget(delBtn, i, 2)
        self._layout.addWidget(group)
        self._layout.addWidget(self._newClsBtn)
        # self._layout.addStretch()

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
        

class ClassifierWidget(QtWidgets.QGroupBox):
    ClassTransparencyUpdated: QtCore.pyqtSignal = QtCore.pyqtSignal(float)

    def __init__(self, parent: 'MainWindow'):
        super(ClassifierWidget, self).__init__()
        self._parent: 'MainWindow' = parent
    #     self._specObj: 'SpectraObject' =
    #     self._clf: Union[None, 'BaseClassifier'] = None
    #     # self._graphView: 'GraphView' = parent.getGraphView()
    #     self._logger: 'Logger' = getLogger("ClassifierWidget")
    #
    #     self._rdfBtn: QtWidgets.QRadioButton = QtWidgets.QRadioButton('Random Decision Forest')
    #     self._rdfBtn.setChecked(True)
    #     self._nnBtn: QtWidgets.QRadioButton = QtWidgets.QRadioButton('Neural Net')
    #
    #     updateBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Update Image")
    #     updateBtn.released.connect(self._classifyImage)
    #
    #     self._transpSlider: QtWidgets.QSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    #     self._transpSlider.setMinimum(0)
    #     self._transpSlider.setValue(80)
    #     self._transpSlider.setMaximum(100)
    #     self._transpSlider.valueChanged.connect(self._emitUpdate)
    #
    #     btnGroup = QtWidgets.QGroupBox('Select Classifier:')
    #     btnLayout = QtWidgets.QVBoxLayout()
    #     btnLayout.addWidget(self._rdfBtn)
    #     btnLayout.addWidget(self._nnBtn)
    #     btnGroup.setLayout(btnLayout)
    #
    #     layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
    #     self.setLayout(layout)
    #     layout.addWidget(btnGroup)
    #     layout.addWidget(updateBtn)
    #     layout.addWidget(QtWidgets.QLabel("Set Overlay Transparency"))
    #     layout.addWidget(self._transpSlider)
    #
    # def getClassifier(self) -> 'BaseClassifier':
    #     return self._clf
    #
    # def _classifyImage(self) -> None:
    #     """
    #     Classifies the current image based on the selected pixels and their classes
    #     :return:
    #     """
    #     self._parent.disableWidgets()
    #     time.sleep(0.1)
    #     imgLimits: QtCore.QRectF = self._graphView.getCurrentViewBounds()
    #     self._specObj.applyPreprocessing(imgLimits)
    #
    #     if self._rdfBtn.isChecked():
    #         self._clf = RandomDecisionForest(self._parent.getDescriptorLibrary())
    #     elif self._nnBtn.isChecked():
    #         self._clf = NeuralNetClassifier(self._specObj.getNumberOfFeatures(), self._specObj.getNumberOfClasses())
    #
    #     try:
    #         self._trainClassifier()
    #         assignments: List[str] = self._getClassesForPixels(imgLimits)
    #
    #         clfImg: np.ndarray = self._createClassImg(assignments, imgLimits)
    #         self._graphView.setClassOverlay(clfImg)
    #     except Exception as e:
    #         QtWidgets.QMessageBox.critical(self, "Error", f"Application of classifier failed:\n{e}")
    #     self._parent.enableWidgets()
    #
    # def _trainClassifier(self) -> None:
    #     """
    #     Trains the classifier
    #     :return:
    #     """
    #     cubeSpectra: Dict[str, np.ndarray] = self._specObj.getClassSpectra(maxSpecPerClas=np.inf)
    #     assigmnents: List[str] = []
    #     spectra: List[np.ndarray] = []  # Wavenumbers in first column
    #     for cls_name, specs in cubeSpectra.items():
    #         for i in range(specs.shape[0]):
    #             spectra.append(specs[i, :])
    #             assigmnents.append(cls_name)
    #     if type(self._clf) == RandomDecisionForest:
    #         self._clf = cast(RandomDecisionForest, self._clf)
    #         self._clf.setWavenumbers(self._specObj.getWavenumbers())
    #
    #     self._clf.trainWithSpectra(np.array(spectra), assigmnents)
    #
    # def _getClassesForPixels(self, imgLimits: QtCore.QRectF) -> List[str]:
    #     """
    #     Estimates the classes for each pixel
    #     :param imgLimits: QtCore.QRectF Image boundaries to consider
    #     :return:
    #     """
    #     t0 = time.time()
    #     specList: List[np.ndarray] = []
    #     cube: np.ndarray = self._specObj.getCube()
    #     for y in range(cube.shape[1]):
    #         if imgLimits.top() <= y < imgLimits.bottom():
    #             for x in range(cube.shape[2]):
    #                 if imgLimits.left() <= x < imgLimits.right():
    #                     specList.append(cube[:, y, x])
    #
    #     result = self._clf.evaluateSpectra(np.array(specList))
    #     self._logger.debug(f'classification took {round(time.time() - t0, 2)} seconds')
    #     return result
    #
    # def _createClassImg(self, assignments: List[str], imgLimits: QtCore.QRectF) -> np.ndarray:
    #     colorCodes: Dict[str, Tuple[int, int, int]] = {}
    #     for cls_name in np.unique(assignments):
    #         colorCodes[cls_name] = self._parent.getColorOfClass(cls_name)
    #
    #     shape = self._specObj.getCube().shape
    #     clfImg: np.ndarray = np.zeros((shape[1], shape[2], 4), dtype=np.uint8)
    #     i: int = 0
    #     for y in range(shape[1]):
    #         for x in range(shape[2]):
    #             if imgLimits.top() <= y < imgLimits.bottom() and imgLimits.left() <= x < imgLimits.right():
    #                 clfImg[y, x, :3] = colorCodes[assignments[i]]
    #                 clfImg[y, x, 3] = 255
    #                 i += 1
    #             else:
    #                 clfImg[y, x, :] = (0, 0, 0, 0)
    #
    #     return clfImg
    #
    # def _emitUpdate(self) -> None:
    #     self.ClassTransparencyUpdated.emit(self._transpSlider.value() / 100)
