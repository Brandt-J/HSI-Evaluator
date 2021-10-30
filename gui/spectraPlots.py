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
from typing import List, Dict, TYPE_CHECKING, cast, Union

from logger import getLogger
from gui.scatterPlot import ScatterPlot
from helperfunctions import getRandomSpectraFromArray
# from SpectraProcessing.descriptors import DescriptorLibrary, DescriptorSet, TriangleDescriptor

if TYPE_CHECKING:
    from gui.HSIEvaluator import MainWindow
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
        self._scatterPlot: 'ScatterPlot' = ScatterPlot()

        self._tabView: QtWidgets.QTabWidget = QtWidgets.QTabWidget()
        self.setMinimumWidth(100)
        self._configureWidgets()
        self._createLayout()

    def setMainWinRef(self, mainWin: 'MainWindow') -> None:
        self._mainWindow = mainWin
        self._specPlot.setMainWindow(mainWin)
        self._scatterPlot.setMainWindow(mainWin)

    def getMaxNumOfSpectraPerCls(self) -> int:
        """
        Gets the max number of spectra per class for previewing in the spectra and the scatter plot.
        """
        return self._numSpecSpinner.value()

    def getRandomSeed(self) -> int:
        """
        Gets the desired random seed.
        """
        return self._seedSpinner.value()

    def setClassAndSampleLabels(self, classLabels, sampleNames) -> None:
        self._specPlot.setClassAndSampleNames(classLabels, sampleNames)
        self._scatterPlot.setClassAndSampleNames(classLabels, sampleNames)

    @QtCore.pyqtSlot(str)
    def switchToDescriptorSet(self, descSetName: str) -> None:
        pass
        # self._specPlot.switchToDescriptorSet(descSetName)

    @QtCore.pyqtSlot(np.ndarray)
    def updateSpecPlot(self, spectra: np.ndarray) -> None:
        self._specPlot.resetPlots()
        self._specPlot.updatePlot(spectra)
        self._specPlot.finishPlotting()

    def updateCursorSpectrum(self, intensities: np.ndarray) -> None:
        self._specPlot.updateCursorSpectrum(intensities)

    @QtCore.pyqtSlot(np.ndarray)
    def updateScatterPlot(self, spectra: np.ndarray) -> None:
        self._scatterPlot.resetPlots()
        self._scatterPlot.updatePlot(spectra)
        self._scatterPlot.finishPlotting()

    # def getDecsriptorLibrary(self) -> DescriptorLibrary:
    #     return self._specPlot.getDecsriptorLibrary()
    #
    # def setDescriptorLibrary(self, descLib: 'DescriptorLibrary') -> None:
    #     self._specPlot.setDescriptorLibrary(descLib)

    def getShowAllSamples(self) -> bool:
        """
        Returns if the spectra of all samples are to be shown or only from the currently active sampleview.
        """
        return self._showAllCheckBox.isChecked()

    def clearPlots(self) -> None:
        """
        Clears the plots. Called before updating the preprocessed spectra.
        """
        self._specPlot.resetPlots()
        self._scatterPlot.resetPlots()

    # def _plotAllSamples(self) -> None:
    #     """
    #     Plots the Spectra from all samples, labelled also with sample names.
    #     :return:
    #     """
    #     i: int = 0
    #     allSpecs: Dict[str, Dict[str, np.ndarray]] = self._mainWindow.getLabelledSpectraFromAllViews()
    #     allBackgrounds: Dict[str, np.ndarray] = self._mainWindow.getBackgroundsOfAllSamples()
    #
    #     for sampleName, specDict in allSpecs.items():
    #         background: np.ndarray = allBackgrounds[sampleName]
    #         self._plotSpectraDict(specDict, background, sampleName=sampleName, linestyle=self.linestyles[i])
    #
    #         if i == len(self.linestyles)-2:
    #             i = 0
    #         else:
    #             i += 1
    #
    # def _plotSpectraDict(self, specs: Dict[str, np.ndarray], backgroundSpec: np.ndarray,
    #                      sampleName: Union[None, str] = None,
    #                      linestyle: [Union[str, tuple]] = "solid") -> None:
    #     """
    #     Plots the given spectra dictionary from the currently active sample.
    #     :param specs: The spectr dictionaty (key: classname, value: spectra (NxM) array of N spectra with M wavelengths
    #     :param backgroundSpec: Averaged background spectrum of the corresponding sample.
    #     :param sampleName: optional samplename to include in legend
    #     :param linestyle: optional linestyle for the sample spectra
    #     :return:
    #     """
    #     def getLegendName() -> str:
    #         if sampleName is None:
    #             nameForLegend = cls_name
    #         else:
    #             nameForLegend = f"{cls_name} from {sampleName[:15]}"
    #         return nameForLegend
    #
    #     preprocessors: List['Preprocessor'] = self._mainWindow.getPreprocessors()
    #
    #     for i, (cls_name, cls_specs) in enumerate(specs.items()):
    #         color = [i / 255 for i in self._mainWindow.getColorOfClass(cls_name)]
    #         cls_specs = self._limitSpecNumber(cls_specs)
    #         cls_specs = self._preprocessSpectra(cls_specs, preprocessors, backgroundSpec)
    #
    #         legendName = getLegendName()
    #         self._specPlot.plotSpectra(cls_specs, i, linestyle, color, legendName)
    #         self._scatterPlot.addSpectraToPCA(cls_specs, linestyle, color, legendName)

    def _configureWidgets(self) -> None:
        self._numSpecSpinner.setMinimum(0)
        self._numSpecSpinner.setMaximum(1000)
        self._numSpecSpinner.setValue(100)
        self._numSpecSpinner.setMaximumWidth(50)

        self._seedSpinner.setMinimum(0)
        self._seedSpinner.setMaximum(100)
        self._seedSpinner.setValue(42)

        self._showAllCheckBox.setChecked(True)

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
        self._tabView.addTab(self._scatterPlot, "Scatter View")

        layout.addLayout(optionsLayout)
        layout.addWidget(self._tabView)
        self.setLayout(layout)

    # def _limitSpecNumber(self, specSet: np.ndarray) -> np.ndarray:
    #     """
    #     Limits number of given specSet to not exceed the value given with numSpecSpinner.
    #     :param specSet: (NxM) set of N spectra with M wavelengths
    #     :return: (N'xM) set of N' (<= numSpecSpinner.value()) spectra with M wavelengths
    #     """
    #     numSpecs: int = specSet.shape[0]
    #     maxSpecs: int = self._numSpecSpinner.value()
    #     if numSpecs > maxSpecs:
    #         specSet = getRandomSpectraFromArray(specSet, maxSpecs, self._seedSpinner.value())
    #     return specSet


class SpecPlot(QtWidgets.QWidget):
    def __init__(self, parent: 'ResultPlots'):
        super(SpecPlot, self).__init__()
        self._figure: plt.Figure = plt.Figure()
        self._canvas: FigureCanvas = FigureCanvas(self._figure)

        self._parent: 'ResultPlots' = parent
        self._mainWin: Union[None, 'MainWindow'] = None
        self._showDescCheckBox: QtWidgets.QCheckBox = QtWidgets.QCheckBox()
        self._legendOutsideCheckBox: QtWidgets.QCheckBox = QtWidgets.QCheckBox()
        self._showLegendCheckBox: QtWidgets.QCheckBox = QtWidgets.QCheckBox()
        self._stackSpinner: QtWidgets.QDoubleSpinBox = QtWidgets.QDoubleSpinBox()
        self._avgCheckBox: QtWidgets.QCheckBox = QtWidgets.QCheckBox()
        self._cursorCheckBox: QtWidgets.QCheckBox = QtWidgets.QCheckBox()

        self._specAx: plt.Axes = self._figure.add_subplot()
        # self._descAx: plt.Axes = self._specAx.twinx()
        self._cursorSpec: Union[plt.Line2D, None] = None
        self._currentSpectra: Union[None, np.ndarray] = None

        # self._descLib: DescriptorLibrary = DescriptorLibrary()
        # self._activeDescSet: Union[None, DescriptorSet] = None

        self._labels: Union[None, np.ndarray] = None
        self._sampleNames: Union[None, np.ndarray] = None

        self._legendItems: List[str] = []
        self._descLines: List[List[plt.Line2D]] = []
        self._selectedDescIndex: int = -1
        self._selectedPoint: int = -1
        self._onClickeEnabled: bool = True
        self._noClickTimer: QtCore.QTimer = QtCore.QTimer()

        self._configureWidgets()
        self._createLayout()

    def updatePlot(self, spectra: np.ndarray) -> None:
        self._currentSpectra = spectra.copy()
        uniqueLabels: np.ndarray = np.unique(self._labels)
        uniqueSamples: np.ndarray = np.unique(self._sampleNames)
        offsets: np.ndarray = self._getoffsetsFromLabelsAndSpectra()
        spectra += offsets
        for uniqueLbl in uniqueLabels:
            self._legendItems.append(uniqueLbl)
            ind: np.ndarray = np.where(self._labels == uniqueLbl)[0]
            specs: np.ndarray = spectra[ind, :]
            if self._avgCheckBox.isChecked():
                specs = np.mean(specs, axis=0)

            color = [i / 255 for i in self._mainWin.getColorOfClass(uniqueLbl)]
            self._specAx.plot(self._mainWin.getWavelengths(), specs.transpose(), color=color)

    def setMainWindow(self, mainWinRef: 'MainWindow') -> None:
        self._mainWin = mainWinRef

    def setClassAndSampleNames(self, classLabels: np.ndarray, sampleNames: np.ndarray) -> None:
        self._labels = classLabels
        self._sampleNames = sampleNames

    # def getDecsriptorLibrary(self) -> DescriptorLibrary:
    #     return self._descLib
    #
    # def setDescriptorLibrary(self, descLib: 'DescriptorLibrary') -> None:
    #     self._descLib = descLib
    #     self._activeDescSet = None
    #     self._plotDescriptors()
    #     self._figure.tight_layout()
    #
    # def switchToDescriptorSet(self, descSetName: str) -> None:
    #     presentNames: List[str] = [descSet.name for descSet in self._descLib.get_descriptorSets()]
    #     if descSetName in presentNames:
    #         self._activeDescSet = self._descLib.get_descriptorSets()[presentNames.index(descSetName)]
    #     else:
    #         newDescSet: DescriptorSet = DescriptorSet(descSetName)
    #         self._descLib.add_descriptorSet(newDescSet)
    #         self._activeDescSet = newDescSet
    #
    #     self._plotDescriptors()

    def resetPlots(self) -> None:
        """
        Called before starting to plot a new set of spectra.
        """
        self._specAx.clear()
        self._legendItems = []
        self._canvas.draw()

    def finishPlotting(self) -> None:
        """
        Called after finishing plotting a set of spectra.
        """
        if self._cursorSpec is not None and self._cursorCheckBox.isChecked():
            cursorSpec: np.ndarray = self._cursorSpec.get_ydata()
            self._cursorSpec = self._specAx.plot(self._mainWin.getWavelengths(), cursorSpec, color='gray')[0]

        self._specAx.set_xlabel("Wavelength (nm)")
        self._specAx.set_ylabel("Intensity (a.u.)")
        if self._showLegendCheckBox.isChecked():
            if self._legendOutsideCheckBox.isChecked():
                self._specAx.legend(self._legendItems, bbox_to_anchor=(1, 1), loc="upper left")
            else:
                self._specAx.legend(self._legendItems)
        self._figure.tight_layout()
        self._plotDescriptors()
        self._canvas.draw()

    def updateCursorSpectrum(self, intensities: np.ndarray) -> None:
        """
        Updates the spectrum under the mouse position.
        """
        if self._cursorCheckBox.isChecked():
            if self._cursorSpec is None:
                wavelengths: np.ndarray = self._mainWin.getWavelengths()
                self._cursorSpec = self._specAx.plot(wavelengths, intensities, color='gray')[0]
            else:
                self._cursorSpec.set_ydata(intensities)

            self._canvas.draw()

    def _createLayout(self) -> None:
        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        optionsLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        col1: QtWidgets.QFormLayout = QtWidgets.QFormLayout()
        col2: QtWidgets.QFormLayout = QtWidgets.QFormLayout()
        col1.addRow("Average spectra per class", self._avgCheckBox)
        col1.addRow("Stack amount", self._stackSpinner)
        col1.addRow("Show legend", self._showLegendCheckBox)
        col2.addRow("Show legend outside plot", self._legendOutsideCheckBox)
        col2.addRow("Show cursor spectrum", self._cursorCheckBox)
        # col2.addRow("Show descriptors", self._showDescCheckBox)
        optionsLayout.addLayout(col1)
        optionsLayout.addLayout(col2)

        layout.addLayout(optionsLayout)
        layout.addWidget(self._canvas)
        self.setLayout(layout)

    def _configureWidgets(self) -> None:
        self._stackSpinner.setMinimum(0)
        self._stackSpinner.setMaximum(2)
        self._stackSpinner.setDecimals(2)
        self._stackSpinner.setSingleStep(0.1)
        self._stackSpinner.valueChanged.connect(self._onPlotSettingsChanged)
        self._stackSpinner.setMaximumWidth(50)

        self._avgCheckBox.setChecked(True)
        self._avgCheckBox.toggled.connect(self._onPlotSettingsChanged)
        self._showDescCheckBox.setChecked(True)

        self._cursorCheckBox.setChecked(True)
        self._cursorCheckBox.toggled.connect(self._onPlotSettingsChanged)

        self._legendOutsideCheckBox.toggled.connect(self._onPlotSettingsChanged)
        self._showLegendCheckBox.setChecked(True)
        self._showLegendCheckBox.toggled.connect(self._onPlotSettingsChanged)
        self._noClickTimer.timeout.connect(self._enableOnClick)

        self._canvas.mpl_connect('button_press_event', self._onClick)
        self._canvas.mpl_connect('pick_event', self._onPick)
        self._canvas.mpl_connect('motion_notify_event', self._movePoints)
        self._canvas.mpl_connect('button_release_event', self._releasePoints)

    # def _prepareSpecsForPlot(self, specArr: np.ndarray) -> np.ndarray:
    #     specArr = specArr.copy()
    #     if self._avgCheckBox.isChecked():
    #         specArr = np.mean(specArr, axis=0)
    #     return specArr.transpose()

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
        pass
        # if self._selectedDescIndex != -1 and self._selectedPoint != -1 and self._activeDescSet is not None:
        #     desc: TriangleDescriptor = self._activeDescSet.getDescriptors()[self._selectedDescIndex]
        #     startPeakEnd: List[float] = [desc.start, desc.peak, desc.end]
        #     startPeakEnd[self._selectedPoint] = event.xdata
        #     startPeakEnd = sorted(np.abs(startPeakEnd))  # avoid points getting negative and/or out of order
        #
        #     desc.start, desc.peak, desc.end = startPeakEnd[0], startPeakEnd[1], startPeakEnd[2]
        #
        #     self._plotDescriptors()

    def _releasePoints(self, _event: MouseEvent) -> None:
        self._selectedPoint = -1

    def _addNewDescriptor(self, middlePos: float):
        if self._activeDescSet is not None:
            middlePos = max([10, middlePos])  # prevent getting a negative start position
            self._activeDescSet.add_descriptor(middlePos-10, middlePos, middlePos+10)

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

    def _onPlotSettingsChanged(self) -> None:
        if self._currentSpectra is not None:
            self.resetPlots()
            self.updatePlot(self._currentSpectra)
            self.finishPlotting()

    def _getoffsetsFromLabelsAndSpectra(self) -> np.ndarray:
        """
        Gets an offset array for the current spectra according to the set labels.
        """
        assert self._currentSpectra is not None
        offsets = np.zeros_like(self._currentSpectra)
        offsetVal: float = self._stackSpinner.value()
        if offsetVal != 0:
            uniquelabels: List[str] = list(np.unique(self._labels))
            for lbl in uniquelabels:
                lblInd: np.ndarray = np.where(self._labels == lbl)[0]
                offsets[lblInd, :] = offsetVal * uniquelabels.index(lbl)

        return offsets
