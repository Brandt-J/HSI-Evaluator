import random
from PyQt5 import QtWidgets, QtGui, QtCore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.patches as mpatches
from matplotlib.backend_bases import MouseEvent, MouseButton, PickEvent
import numpy as np
from typing import List, Dict, TYPE_CHECKING, cast, Union

from logger import getLogger
from SpectraProcessing.descriptors import DescriptorLibrary, DescriptorSet, TriangleDescriptor
from preprocessors import Background

if TYPE_CHECKING:
    from gui.ImecEvaluator import MainWindow
    from preprocessors import Preprocessor
    from logging import Logger


class SpectraPreviewWidget(QtWidgets.QWidget):
    linestyles: List[Union[str, tuple]] = ["-", "--", "-.",
                                           (0, (1, 10)),
                                           (0, (1, 1)),
                                           (0, (5, 10)),
                                           (0, (5, 1)),
                                           (0, (3, 10, 1, 10)),
                                           (0, (3, 5, 1, 5)),
                                           (0, (3, 1, 1, 1)),
                                           (0, (3, 5, 1, 5, 1, 5))]

    def __init__(self):
        super(SpectraPreviewWidget, self).__init__()
        self._descLib: DescriptorLibrary = DescriptorLibrary()
        self._activeDescSet: Union[None, DescriptorSet] = None
        self._mainWindow: Union[None, 'MainWindow'] = None
        self._descLines: List[List[plt.Line2D]] = []
        self._legendItems: List[mpatches.Patch] = []
        self._selectedDescIndex: int = -1
        self._selectedPoint: int = -1
        self._logger: 'Logger' = getLogger("SpectraPreview")
        self._onClickeEnabled: bool = True
        self._noClickTimer: QtCore.QTimer = QtCore.QTimer()
        self._noClickTimer.timeout.connect(self._enableOnClick)

        self._numSpecSpinner: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
        self._stackSpinner: QtWidgets.QDoubleSpinBox = QtWidgets.QDoubleSpinBox()
        self._avgCheckBox: QtWidgets.QCheckBox = QtWidgets.QCheckBox()
        self._showAllCheckBox: QtWidgets.QCheckBox = QtWidgets.QCheckBox()
        self._showDescCheckBox: QtWidgets.QCheckBox = QtWidgets.QCheckBox()

        self._figure: plt.Figure = plt.Figure()
        self._specAx: plt.Axes = self._figure.add_subplot()
        self._descAx: plt.Axes = self._specAx.twinx()
        self._cursorSpec: Union[plt.Line2D, None] = None
        self._canvas: FigureCanvas = FigureCanvas(self._figure)
        self._canvas.mpl_connect('button_press_event', self._onClick)
        self._canvas.mpl_connect('pick_event', self._onPick)
        self._canvas.mpl_connect('motion_notify_event', self._movePoints)
        self._canvas.mpl_connect('button_release_event', self._releasePoints)

        self._configureWidgets()
        self._createLayout()

    def setMainWinRef(self, mainWin: 'MainWindow') -> None:
        self._mainWindow = mainWin

    @QtCore.pyqtSlot(str)
    def switchToDescriptorSet(self, descSetName: str) -> None:
        presentNames: List[str] = [descSet.name for descSet in self._descLib.get_descriptorSets()]
        if descSetName in presentNames:
            self._activeDescSet = self._descLib.get_descriptorSets()[presentNames.index(descSetName)]
        else:
            newDescSet: DescriptorSet = DescriptorSet(descSetName)
            self._descLib.add_descriptorSet(newDescSet)
            self._activeDescSet = newDescSet

        self._plotDescriptors()

    def getDecsriptorLibrary(self) -> DescriptorLibrary:
        return self._descLib

    def setDescriptorLibrary(self, descLib: 'DescriptorLibrary') -> None:
        self._descLib = descLib
        self._activeDescSet = None
        self._plotDescriptors()
        self._figure.tight_layout()

    def updateSpectra(self) -> None:
        """
        Updating the spec plot
        :return:
        """
        self._specAx.clear()
        self._resetLegendItems()

        if self._mainWindow is not None:
            if self._showAllCheckBox.isChecked():
                self._plotSpectraFromAllSamples()
            else:
                self._plotSpectraDict(self._mainWindow.getLabelledSpectraFromActiveView())

        if self._cursorSpec is not None:
            cursorSpec: np.ndarray = self._cursorSpec.get_ydata()
            self._cursorSpec = self._specAx.plot(self._mainWindow.getWavenumbers(), cursorSpec, color='gray')[0]

        self._specAx.legend(self._legendItems)
        self._plotDescriptors()

    def _plotSpectraFromAllSamples(self) -> None:
        """
        Plots the Spectra from all samples, labelled also with sample names.
        :return:
        """

        i: int = 0
        allSpecs = self._mainWindow.getLabelledSpectraFromAllViews()
        for sampleName, specDict in allSpecs.items():
            self._plotSpectraDict(specDict, sampleName=sampleName, linestyle=self.linestyles[i])
            if i == len(self.linestyles)-2:
                i = 0
            else:
                i += 1

    def _plotSpectraDict(self, specs: Dict[str, np.ndarray], sampleName: Union[None, str] = None,
                         linestyle: [Union[str, tuple]] = "solid") -> None:
        """
        Plots the given spectra dictionary from the currently active sample.
        :param specs: The spectr dictionaty (key: classname, value: spectra (NxM) array of N spectra with M wavenumbers
        :param sampleName: optional samplename to include in legend
        :param linestyle: optional linestyle for the sample spectra
        :return:
        """
        preprocessors: List['Preprocessor'] = self._mainWindow.getPreprocessors()

        for i, (cls_name, cls_specs) in enumerate(specs.items()):
            color = [i / 255 for i in self._mainWindow.getColorOfClass(cls_name)]
            cls_specs = self._limitSpecNumber(cls_specs)
            cls_specs = self._prepareSpecsForPlot(cls_specs, preprocessors)

            self._specAx.plot(self._mainWindow.getWavenumbers(), cls_specs - i*self._stackSpinner.value(),
                              linestyle=linestyle, color=color)

            if sampleName is None:
                legendName = cls_name
            else:
                legendName = f"{cls_name} from {sampleName}"

            self._legendItems.append(legendName)

    def updateCursorSpectrum(self, x: int, y: int) -> None:
        pass  # TODO: REFACTOR to directly take spectrum as input
        # spec: np.ndarray = self._specObj.getSpectrumaAtXY(x, y)
        # if self._cursorSpec is None:
        #     self._cursorSpec = self._specAx.plot(self._specObj.getWavenumbers(), spec, color='gray')[0]
        # else:
        #     self._cursorSpec.set_ydata(spec)
        # self._canvas.draw()

    def _configureWidgets(self) -> None:
        self._numSpecSpinner.setMinimum(0)
        self._numSpecSpinner.setMaximum(100)
        self._numSpecSpinner.setValue(20)

        self._stackSpinner.setMinimum(0)
        self._stackSpinner.setMaximum(2)
        self._stackSpinner.setDecimals(2)
        self._stackSpinner.setSingleStep(0.1)

        self._avgCheckBox.setChecked(True)
        self._showDescCheckBox.setChecked(True)
        self._showAllCheckBox.setChecked(True)

        for spinner in [self._numSpecSpinner, self._stackSpinner]:
            spinner.setMaximumWidth(50)
            spinner.valueChanged.connect(self.updateSpectra)

        for checkbox in [self._avgCheckBox, self._showDescCheckBox, self._showAllCheckBox]:
            checkbox.stateChanged.connect(self.updateSpectra)

    def _createLayout(self) -> None:
        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        optionsLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        leftLayout: QtWidgets.QFormLayout = QtWidgets.QFormLayout()
        rightLayout: QtWidgets.QFormLayout = QtWidgets.QFormLayout()
        leftLayout.addRow("Show All Samples", self._showAllCheckBox)
        leftLayout.addRow("Average per Class", self._avgCheckBox)
        leftLayout.addRow("Show Decriptors", self._showDescCheckBox)
        rightLayout.addRow("Max. Spectra per Class", self._numSpecSpinner)
        rightLayout.addRow("Spectra Group Stacking", self._stackSpinner)

        optionsLayout.addLayout(leftLayout)
        optionsLayout.addLayout(rightLayout)

        layout.addLayout(optionsLayout)
        layout.addWidget(self._canvas)
        self.setLayout(layout)

    def _plotDescriptors(self) -> None:
        self._descAx.clear()
        self._descLines = []
        if self._activeDescSet is not None and self._showDescCheckBox.isChecked():
            for i, desc in enumerate(self._activeDescSet.getDescriptors()):
                x = [desc.start, desc.peak, desc.end]
                y = [0, 1, 0]
                color = 'black' if i != self._selectedDescIndex else 'green'
                line: plt.Line2D = self._descAx.plot(x, y, color=color, marker='o', picker=5)[0]
                self._descLines.append(line)

        self._figure.tight_layout()
        self._canvas.draw()

    def _prepareSpecsForPlot(self, specArr: np.ndarray, preprocessors: List['Preprocessor']) -> np.ndarray:
        """
        Prepares the spec (NxM) array of N spec with M wavenums for plotting.
        Performs averating (if desired) and transpose.
        :param specArr: (NxM) array of N spec with M wavenums
        :param preprocessors: The preprocessors to use
        :return: spec array in (MxN) shape, as required for plt batch plotting
        """
        newArr: np.ndarray = specArr.copy()
        if self._avgCheckBox.isChecked():
            newArr = np.mean(newArr, axis=0)

        onedimensional = len(newArr.shape) == 1
        if onedimensional:
            newArr = newArr[np.newaxis, :]

        for processor in preprocessors:
            if type(processor) == Background:
                self._logger.warning("Background subtraction requested, but currently not implemented. Will be skipped!")
                continue  # TODO: REIMPLEMENT!
                # processor: Background = cast(Background, processor)
                # processor.setBackground(self._specObj.getMeanBackgroundSpec())
            newArr = processor.applyToSpectra(newArr)

        return newArr.transpose()

    def _onClick(self, event: MouseEvent) -> None:
        if self._onClickeEnabled:
            if event.button == MouseButton.LEFT and event.dblclick and self._activeDescSet is not None:
                self._addNewDescriptor(event.xdata)
            elif event.button == MouseButton.LEFT and not event.dblclick:
                self._deselectDescriptor()
            elif event.button == MouseButton.RIGHT and not event.dblclick:
                popMenu = QtWidgets.QMenu()
                popMenu.addAction("Delete Descriptor")
                popMenu.setEnabled(self._selectedDescIndex > -1)
                action = popMenu.exec_(QtGui.QCursor().pos())
                if action:
                    self._deleteSelectedDescriptor()

    def _onPick(self, event: PickEvent):
        self._selectedDescIndex = self._descLines.index(event.artist)
        try:
            self._selectedPoint = int(event.ind)
        except TypeError:
            print(f"Error: selectedDesc: {self._selectedDescIndex}, event.ind: {event.ind}")

        self._onClickeEnabled = False
        self._noClickTimer.start(500)  # To prevent directly also triggering the onClick event
        self._plotDescriptors()

    def _movePoints(self, event: MouseEvent) -> None:
        if self._selectedDescIndex != -1 and self._selectedPoint != -1 and self._activeDescSet is not None:
            desc: TriangleDescriptor = self._activeDescSet.getDescriptors()[self._selectedDescIndex]
            startPeakEnd: List[float] = [desc.start, desc.peak, desc.end]
            startPeakEnd[self._selectedPoint] = event.xdata
            startPeakEnd = sorted(np.abs(startPeakEnd))  # avoid points getting negative and/or out of order

            desc.start, desc.peak, desc.end = startPeakEnd[0], startPeakEnd[1], startPeakEnd[2]

            self._plotDescriptors()

    def _releasePoints(self, _event: MouseEvent) -> None:
        self._selectedPoint = -1

    def _addNewDescriptor(self, middlePos: float):
        if self._activeDescSet is not None:
            middlePos = max([10, middlePos])  # prevent getting a negative start position
            self._activeDescSet.add_descriptor(middlePos-10, middlePos, middlePos+10)
            self.updateSpectra()

    def _enableOnClick(self) -> None:
        self._onClickeEnabled = True

    def _deselectDescriptor(self) -> None:
        self._selectedDescIndex = -1
        self._selectedPoint = -1
        self._plotDescriptors()

    def _deleteSelectedDescriptor(self) -> None:
        if self._selectedDescIndex > -1:
            self._activeDescSet.remove_descriptor_of_index(self._selectedDescIndex)
            self._deselectDescriptor()

    def _resetLegendItems(self) -> None:
        self._legendItems = []

    def _limitSpecNumber(self, specSet: np.ndarray) -> np.ndarray:
        """
        Limits number of given specSet to not exceed the value given with numSpecSpinner.
        :param specSet: (NxM) set of N spectra with M wavenumbers
        :return: (N'xM) set of N' (<= numSpecSpinner.value()) spectra with M wavenumbers
        """
        numSpecs: int = specSet.shape[0]
        maxSpecs: int = self._numSpecSpinner.value()
        if numSpecs > maxSpecs:
            specSet = specSet.copy()
            randInd: np.ndarray = np.array(random.sample(list(np.arange(numSpecs)), maxSpecs))
            specSet = specSet[randInd, :]
        return specSet
