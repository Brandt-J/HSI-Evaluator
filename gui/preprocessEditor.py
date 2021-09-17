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

from PyQt5 import QtWidgets, QtCore
from typing import List, TYPE_CHECKING, Union
import numpy as np
from collections import Counter

from gui.nodegraph.nodegraph import NodeGraph
if TYPE_CHECKING:
    from gui.HSIEvaluator import MainWindow
    from spectraObject import SpectraCollection
    from gui.spectraPlots import ResultPlots
    from preprocessing.preprocessors import Preprocessor


class PreprocessingSelector(QtWidgets.QGroupBox):
    ProcessorStackUpdated: QtCore.pyqtSignal = QtCore.pyqtSignal()

    def __init__(self, mainWinParent: 'MainWindow', resultPlots: 'ResultPlots'):
        super(PreprocessingSelector, self).__init__()
        self.setWindowTitle("Define Preprocessing")
        
        self._nodeGraph: NodeGraph = NodeGraph()
        self._nodeGraph.ClassificationPathHasChanged.connect(lambda: self.ProcessorStackUpdated.emit())

        self._mainWin: 'MainWindow' = mainWinParent
        self._plots: 'ResultPlots' = resultPlots
        self._nodeGraph.NewSpecsForScatterPlot.connect(self._plots.updateScatterPlot)
        self._nodeGraph.NewSpecsForSpecPlot.connect(self._plots.updateSpecPlot)

        updateBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Update Preview")
        updateBtn.released.connect(self.updatePreviewSpectra)
        updateBtn.setMaximumWidth(130)

        applyToSamplesBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Apply to Samples")
        applyToSamplesBtn.released.connect(self._applyPreprocessingToSpectra)
        applyToSamplesBtn.setMaximumWidth(130)

        btnLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        btnLayout.addWidget(updateBtn)
        btnLayout.addWidget(applyToSamplesBtn)
        btnLayout.addStretch()

        self._layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self._layout.addLayout(btnLayout)
        self._layout.addWidget(self._nodeGraph)
        self.setLayout(self._layout)

    def updatePreviewSpectra(self):
        """
        Updates the data on the input node.
        """
        self._nodeGraph.clearNodeCache()
        self._plots.clearPlots()

        if self._plots.getShowAllSamples():
            specColl: SpectraCollection = self._mainWin.getLabelledSpectraFromAllViews()
        else:
            specColl: SpectraCollection = self._mainWin.getLabelledSpectraFromActiveView()

        if specColl.hasSpectra():
            spectra, labels = specColl.getXY()
            sampleNames = specColl.getSampleNames()
            spectra, labels, sampleNames = self._limitToMaxNumber(spectra, labels, sampleNames)

            self._plots.setClassAndSampleLabels(labels, sampleNames)
            self._nodeGraph.setInputSpecta(spectra)
            self._nodeGraph.updatePlotNodes()
        else:
            self._showNoSpectraWarning()

    def _applyPreprocessingToSpectra(self) -> None:
        pass

    def _showNoSpectraWarning(self) -> None:
        QtWidgets.QMessageBox.about(self, "Info", "Spectra Preprocessing cannot be previewed.\n"
                                                  "There are now spectra currently labelled.")

    def _limitToMaxNumber(self, spectra: np.ndarray, labels: np.ndarray, sampleNames: np.ndarray):
        """
        Limits the number of spectra, labels and samplenames to the maximum number per class given by the result plots.
        """
        maxSpecsPerCls: int = self._plots.getMaxNumOfSpectraPerCls()
        random.seed(self._plots.getRandomSeed())

        newSpecs: Union[None, np.ndarray] = None
        newLabels: List[str] = []
        newSampleNames: List[str] = []

        counter: Counter = Counter(labels)
        for cls, abundancy in counter.items():
            ind: np.ndarray = np.where(labels == cls)[0]
            if abundancy > maxSpecsPerCls:
                ind = np.array(random.sample(list(ind), maxSpecsPerCls))
            if newSpecs is None:
                newSpecs = spectra[ind, :]
            else:
                newSpecs = np.vstack((newSpecs, spectra[ind, :]))
            newLabels += list(labels[ind])
            newSampleNames += list(sampleNames[ind])

        newLabels: np.ndarray = np.array(newLabels)
        newSampleNames: np.ndarray = np.array(newSampleNames)

        return newSpecs, newLabels, newSampleNames

    def getPreprocessors(self) -> List['Preprocessor']:
        """
        Returns a list of the preprocessors connected to the Classification Node.
        """
        return self._nodeGraph.getPreprocessors()

    def getProcessingGraph(self) -> List[dict]:
        """
        Gets a list of nodeDictionaries, representing the currently selected preprocessing setup.
        """
        return self._nodeGraph.getGraphConfig()

    def getPreprocessorNames(self) -> List[str]:
        """
        Returns a list of the currently selected preprocessor names. Used for storing
        """
        return [lbl.text() for lbl in self._selected]

    def applyPreprocessingConfig(self, processConfig: List[dict]) -> None:
        """
        Takes a nodegraph config in form of a list of nodeDictionaries and sets up the graph accordingly.
        :param processConfig: List of Node Configurations (nodeDicts)
        """
        self._nodeGraph.applyGraphConfig(processConfig)
