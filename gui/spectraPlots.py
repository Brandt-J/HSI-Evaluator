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
from PyQt5 import QtWidgets, QtGui, QtCore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backend_bases import MouseEvent, MouseButton, PickEvent
import numpy as np
from typing import List, Dict, TYPE_CHECKING, cast, Union, Tuple

from logger import getLogger
from preprocessors import Background
from gui.pcaPlot import PCAPlot
from SpectraProcessing.descriptors import DescriptorLibrary, DescriptorSet, TriangleDescriptor

if TYPE_CHECKING:
    from gui.HSIEvaluator import MainWindow
    from preprocessors import Preprocessor
    from logging import Logger


class ResultPlots(QtWidgets.QWidget):
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
        super(ResultPlots, self).__init__()
        self._mainWindow: Union[None, 'MainWindow'] = None
        self._logger: 'Logger' = getLogger("ResultPlots")

        self._numSpecSpinner: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
        self._showAllCheckBox: QtWidgets.QCheckBox = QtWidgets.QCheckBox()
        self._seedSpinner: QtWidgets.QSpinBox = QtWidgets.QSpinBox()

        self._specPlot: 'SpecPlot' = SpecPlot(self)
        self._pcaPlot: 'PCAPlot' = PCAPlot()

        self._tabView: QtWidgets.QTabWidget = QtWidgets.QTabWidget()
        self._configureWidgets()
        self._createLayout()

    def setMainWinRef(self, mainWin: 'MainWindow') -> None:
        self._mainWindow = mainWin
        self._specPlot.setMainWindow(mainWin)

    @QtCore.pyqtSlot(str)
    def switchToDescriptorSet(self, descSetName: str) -> None:
        self._specPlot.switchToDescriptorSet(descSetName)

    def getDecsriptorLibrary(self) -> DescriptorLibrary:
        return self._specPlot.getDecsriptorLibrary()

    def setDescriptorLibrary(self, descLib: 'DescriptorLibrary') -> None:
        self._specPlot.setDescriptorLibrary(descLib)

    def updatePlots(self) -> None:
        """
        Updating the spec plot
        :return:
        """
        self._specPlot.resetPlots()
        self._pcaPlot.resetPlots()
        if self._mainWindow is not None:
            if self._showAllCheckBox.isChecked():
                self._plotAllSamples()
            else:
                background: np.ndarray = self._mainWindow.getBackgroundOfActiveSample()
                self._plotSpectraDict(self._mainWindow.getLabelledSpectraFromActiveView(), background)

        self._specPlot.finishPlotting()
        self._pcaPlot.finishPlotting()

    def _plotAllSamples(self) -> None:
        """
        Plots the Spectra from all samples, labelled also with sample names.
        :return:
        """
        i: int = 0
        allSpecs: Dict[str, Dict[str, np.ndarray]] = self._mainWindow.getLabelledSpectraFromAllViews()
        allBackgrounds: Dict[str, np.ndarray] = self._mainWindow.getBackgroundsOfAllSamples()

        for sampleName, specDict in allSpecs.items():
            background: np.ndarray = allBackgrounds[sampleName]
            self._plotSpectraDict(specDict, background, sampleName=sampleName, linestyle=self.linestyles[i])

            if i == len(self.linestyles)-2:
                i = 0
            else:
                i += 1

    def _plotSpectraDict(self, specs: Dict[str, np.ndarray], backgroundSpec: np.ndarray,
                         sampleName: Union[None, str] = None,
                         linestyle: [Union[str, tuple]] = "solid") -> None:
        """
        Plots the given spectra dictionary from the currently active sample.
        :param specs: The spectr dictionaty (key: classname, value: spectra (NxM) array of N spectra with M wavenumbers
        :param backgroundSpec: Averaged background spectrum of the corresponding sample.
        :param sampleName: optional samplename to include in legend
        :param linestyle: optional linestyle for the sample spectra
        :return:
        """
        def getLegendName() -> str:
            if sampleName is None:
                nameForLegend = cls_name
            else:
                nameForLegend = f"{cls_name} from {sampleName[:15]}"
            return nameForLegend

        preprocessors: List['Preprocessor'] = self._mainWindow.getPreprocessors()

        for i, (cls_name, cls_specs) in enumerate(specs.items()):
            color = [i / 255 for i in self._mainWindow.getColorOfClass(cls_name)]
            cls_specs = self._limitSpecNumber(cls_specs)
            cls_specs = self._preprocessSpectra(cls_specs, preprocessors, backgroundSpec)

            legendName = getLegendName()
            self._specPlot.plotSpectra(cls_specs, i, linestyle, color, legendName)
            self._pcaPlot.addSpectraToPCA(cls_specs, linestyle, color, legendName)

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
        self._numSpecSpinner.setMaximum(1000)
        self._numSpecSpinner.setValue(100)
        self._numSpecSpinner.setMaximumWidth(50)
        self._numSpecSpinner.valueChanged.connect(self.updatePlots)

        self._seedSpinner.setMinimum(0)
        self._seedSpinner.setMaximum(100)
        self._seedSpinner.setValue(42)
        self._seedSpinner.valueChanged.connect(self.updatePlots)

        self._showAllCheckBox.setChecked(True)
        self._showAllCheckBox.stateChanged.connect(self.updatePlots)

    def _createLayout(self) -> None:
        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        optionsLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        optionsLayout.addWidget(QtWidgets.QLabel("Show all samples:"))
        optionsLayout.addWidget(self._showAllCheckBox)
        optionsLayout.addStretch()
        optionsLayout.addWidget(QtWidgets.QLabel("Max. spectra per class:"))
        optionsLayout.addWidget(self._numSpecSpinner)
        optionsLayout.addStretch()
        optionsLayout.addWidget(QtWidgets.QLabel("Random seed:"))
        optionsLayout.addWidget(self._seedSpinner)
        optionsLayout.addStretch()

        self._tabView.addTab(self._specPlot, "Spectra View")
        self._tabView.addTab(self._pcaPlot, "PCA View")

        layout.addLayout(optionsLayout)
        layout.addWidget(self._tabView)
        self.setLayout(layout)

    def _preprocessSpectra(self, specArr: np.ndarray, preprocessors: List['Preprocessor'],
                             backgroundSpec: np.ndarray) -> np.ndarray:
        """
        Prepares the spec (NxM) array of N spec with M wavenums for plotting.
        Performs averating (if desired) and transpose.
        :param specArr: (NxM) array of N spec with M wavenums
        :param preprocessors: The preprocessors to use
        :param backgroundSpec: Averaged spectrum of background
        :return: spec array in (MxN) shape, as required for plt batch plotting
        """
        newArr: np.ndarray = specArr.copy()
        onedimensional = len(newArr.shape) == 1
        if onedimensional:
            newArr = newArr[np.newaxis, :]

        for processor in preprocessors:
            if type(processor) == Background:
                processor: Background = cast(Background, processor)
                processor.setBackground(backgroundSpec)
            newArr = processor.applyToSpectra(newArr)

        return newArr

    def _limitSpecNumber(self, specSet: np.ndarray) -> np.ndarray:
        """
        Limits number of given specSet to not exceed the value given with numSpecSpinner.
        :param specSet: (NxM) set of N spectra with M wavenumbers
        :return: (N'xM) set of N' (<= numSpecSpinner.value()) spectra with M wavenumbers
        """
        random.seed(self._seedSpinner.value())
        numSpecs: int = specSet.shape[0]
        maxSpecs: int = self._numSpecSpinner.value()
        if numSpecs > maxSpecs:
            specSet = specSet.copy()
            randInd: np.ndarray = np.array(random.sample(list(np.arange(numSpecs)), maxSpecs))
            specSet = specSet[randInd, :]
        return specSet


class SpecPlot(QtWidgets.QWidget):
    def __init__(self, parent: 'ResultPlots'):
        super(SpecPlot, self).__init__()
        self._figure: plt.Figure = plt.Figure()
        self._canvas: FigureCanvas = FigureCanvas(self._figure)

        self._parent: 'ResultPlots' = parent
        self._mainWin: Union[None, 'MainWindow'] = None
        self._showDescCheckBox: QtWidgets.QCheckBox = QtWidgets.QCheckBox()
        self._stackSpinner: QtWidgets.QDoubleSpinBox = QtWidgets.QDoubleSpinBox()
        self._avgCheckBox: QtWidgets.QCheckBox = QtWidgets.QCheckBox()

        self._specAx: plt.Axes = self._figure.add_subplot()
        # self._descAx: plt.Axes = self._specAx.twinx()
        self._cursorSpec: Union[plt.Line2D, None] = None

        self._descLib: DescriptorLibrary = DescriptorLibrary()
        self._activeDescSet: Union[None, DescriptorSet] = None

        self._legendItems: List[str] = []
        self._descLines: List[List[plt.Line2D]] = []
        self._selectedDescIndex: int = -1
        self._selectedPoint: int = -1
        self._onClickeEnabled: bool = True
        self._noClickTimer: QtCore.QTimer = QtCore.QTimer()

        self._configureWidgets()
        self._createLayout()

    def setMainWindow(self, mainWinRef: 'MainWindow') -> None:
        self._mainWin = mainWinRef

    def getDecsriptorLibrary(self) -> DescriptorLibrary:
        return self._descLib

    def setDescriptorLibrary(self, descLib: 'DescriptorLibrary') -> None:
        self._descLib = descLib
        self._activeDescSet = None
        self._plotDescriptors()
        self._figure.tight_layout()

    def switchToDescriptorSet(self, descSetName: str) -> None:
        presentNames: List[str] = [descSet.name for descSet in self._descLib.get_descriptorSets()]
        if descSetName in presentNames:
            self._activeDescSet = self._descLib.get_descriptorSets()[presentNames.index(descSetName)]
        else:
            newDescSet: DescriptorSet = DescriptorSet(descSetName)
            self._descLib.add_descriptorSet(newDescSet)
            self._activeDescSet = newDescSet

        self._plotDescriptors()

    def resetPlots(self) -> None:
        """
        Called before starting to plot a new set of spectra.
        """
        self._specAx.clear()
        self._legendItems = []

    def plotSpectra(self, spectra: np.ndarray, index: int, linestyle: Union[str, tuple],
                     color: List[float], legendName: str) -> None:
        """
        Plots the spectra array.
        :param spectra: (NxM) array of N spectra with M wavenumbers
        :param index: index of specset, used for optional offset.
        :param linestyle: the linestyle code to use
        :param color: The rgb color to use (or rgba)
        :param legendName: The legendname of the given spec set.
        """
        spectra: np.ndarray = self._prepareSpecsForPlot(spectra)
        self._specAx.plot(self._mainWin.getWavenumbers(), spectra - index * self._stackSpinner.value(),
                          linestyle=linestyle, color=color)

        self._legendItems.append(legendName)

    def finishPlotting(self) -> None:
        """
        Called after finishing plotting a set of spectra.
        """
        # if self._cursorSpec is not None:
        #     cursorSpec: np.ndarray = self._cursorSpec.get_ydata()
        #     self._cursorSpec = self._specAx.plot(self._mainWindow.getWavenumbers(), cursorSpec, color='gray')[0]

        self._specAx.set_xlabel("Wavelength (nm)")
        self._specAx.set_ylabel("Intensity (a.u.)")
        self._specAx.legend(self._legendItems)
        self._plotDescriptors()
        self._canvas.draw()

    def _createLayout(self) -> None:
        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        optionsLayout: QtWidgets.QFormLayout = QtWidgets.QFormLayout()
        optionsLayout.addRow("Average spectra per class", self._avgCheckBox)
        optionsLayout.addRow("Stack amount", self._stackSpinner)
        optionsLayout.addRow("Show descriptors", self._showDescCheckBox)

        layout.addLayout(optionsLayout)
        layout.addWidget(self._canvas)
        self.setLayout(layout)

    def _configureWidgets(self) -> None:
        self._stackSpinner.setMinimum(0)
        self._stackSpinner.setMaximum(2)
        self._stackSpinner.setDecimals(2)
        self._stackSpinner.setSingleStep(0.1)

        self._avgCheckBox.setChecked(True)
        self._showDescCheckBox.setChecked(True)

        self._stackSpinner.setMaximumWidth(50)
        self._stackSpinner.valueChanged.connect(self._parent.updatePlots)

        for checkbox in [self._avgCheckBox, self._showDescCheckBox]:
            checkbox.stateChanged.connect(self._parent.updatePlots)

        self._noClickTimer.timeout.connect(self._enableOnClick)

        self._canvas.mpl_connect('button_press_event', self._onClick)
        self._canvas.mpl_connect('pick_event', self._onPick)
        self._canvas.mpl_connect('motion_notify_event', self._movePoints)
        self._canvas.mpl_connect('button_release_event', self._releasePoints)

    def _prepareSpecsForPlot(self, specArr: np.ndarray) -> np.ndarray:
        specArr = specArr.copy()
        if self._avgCheckBox.isChecked():
            specArr = np.mean(specArr, axis=0)
        return specArr.transpose()

    def _plotDescriptors(self) -> None:
        pass
        # self._descAx.clear()
        # self._descLines = []
        # if self._activeDescSet is not None and self._showDescCheckBox.isChecked():
        #     for i, desc in enumerate(self._activeDescSet.getDescriptors()):
        #         x = [desc.start, desc.peak, desc.end]
        #         y = [0, 1, 0]
        #         color = 'black' if i != self._selectedDescIndex else 'green'
        #         line: plt.Line2D = self._descAx.plot(x, y, color=color, marker='o', picker=5)[0]
        #         self._descLines.append(line)
        #
        # self._figure.tight_layout()
        # self._canvas.draw()

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
            self.updatePlots()

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
