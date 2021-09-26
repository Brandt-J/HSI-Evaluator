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
import time

from PyQt5 import QtWidgets, QtCore
from typing import List, TYPE_CHECKING, Union, Set, Callable, Tuple
import numpy as np
from collections import Counter
from multiprocessing import Process, Queue, Event

from dataObjects import Sample
from gui.nodegraph.nodegraph import NodeGraph
if TYPE_CHECKING:
    from sampleview import SampleView
    from spectraObject import SpectraCollection, SpectraObject
    from gui.HSIEvaluator import MainWindow
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
        self._nodeGraph.ClassificationPathHasChanged.connect(self._enableApplyToSamples)

        self._processingPerformer: PreprocessingPerformer = PreprocessingPerformer()
        self._processingPerformer.PreprocessingFinished.connect(self._setPreprocessedData)
        self._processingPerformer.PreprocessingAborted.connect(self._preprocessingWasAborted)

        updateBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Preview")
        updateBtn.released.connect(self.updatePreviewSpectra)
        updateBtn.setMaximumWidth(130)

        self._applyToSamplesBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Apply to Samples")
        self._applyToSamplesBtn.released.connect(self._applyPreprocessingToSpectra)
        self._applyToSamplesBtn.setMaximumWidth(130)

        btnLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        btnLayout.addWidget(updateBtn)
        btnLayout.addWidget(self._applyToSamplesBtn)
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
            specColl: SpectraCollection = self._mainWin.getLabelledSpectraFromAllViews(preprocessed=True)
        else:
            specColl: SpectraCollection = self._mainWin.getLabelledSpectraFromActiveView(preprocessed=True)

        if specColl.hasSpectra():
            spectra, labels = specColl.getXY()
            sampleNames = specColl.getSampleNames()
            spectra, labels, sampleNames = self._limitToMaxNumber(spectra, labels, sampleNames)

            self._plots.setClassAndSampleLabels(labels, sampleNames)
            self._nodeGraph.setInputSpecta(spectra)
            self._nodeGraph.updatePlotNodes()
        else:
            self._showNoSpectraWarning()

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

    def _applyPreprocessingToSpectra(self) -> None:
        """
        Applies the preprocessing to all samples
        """
        self._mainWin.disableWidgets()
        samples: List['Sample'] = [sample.getSampleData() for sample in self._mainWin.getAllSamples()]
        processors: List['Preprocessor'] = self._nodeGraph.getPreprocessors()
        self._processingPerformer.startPreprocessing(samples, processors)
        self._applyToSamplesBtn.setDisabled(True)

    @QtCore.pyqtSlot()
    def _enableApplyToSamples(self) -> None:
        """
        Re-enables the apply2samples-Btn, when the processing path for the spectra classification has changed.
        """
        self._applyToSamplesBtn.setEnabled(True)

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

    @QtCore.pyqtSlot(list)
    def _setPreprocessedData(self, preprocessedSamples: List['Sample']) -> None:
        """
        Receives the preprocessed samples from the preprocessing Process and sets the sample data herein accordingly.
        """
        sampleViews: List['SampleView'] = self._mainWin.getAllSamples()
        for preprocSampleData in preprocessedSamples:
            for sampleview in sampleViews:
                if sampleview.getSampleData() == preprocSampleData:
                    sampleview.setSampleData(preprocSampleData)
                    break
        self._mainWin.enableWidgets()

    @QtCore.pyqtSlot()
    def _preprocessingWasAborted(self) -> None:
        self._mainWin.enableWidgets()
        self._enableApplyToSamples()


class PreprocessingPerformer(QtWidgets.QWidget):
    PreprocessingFinished: QtCore.pyqtSignal = QtCore.pyqtSignal(list)
    PreprocessingAborted: QtCore.pyqtSignal = QtCore.pyqtSignal()

    def __init__(self):
        super(PreprocessingPerformer, self).__init__()
        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self._progressbar: QtWidgets.QProgressBar = QtWidgets.QProgressBar()
        self._progressbar.setFixedWidth(600)
        self._progressLabel: QtWidgets.QLabel = QtWidgets.QLabel()
        self._btnCancel: QtWidgets.QPushButton = QtWidgets.QPushButton("Cancel")
        self._btnCancel.setMaximumWidth(150)
        self._btnCancel.released.connect(self._promptForCancel)
        layout.addWidget(self._progressbar)
        layout.addWidget(self._progressLabel)
        layout.addWidget(self._btnCancel)

        self._process: Process = Process()
        self._queue: Queue = Queue()
        self._stopEvent: Event = Event()

        self._timer: QtCore.QTimer = QtCore.QTimer()
        self._timer.setSingleShot(False)
        self._timer.timeout.connect(self._checkOnPreprocessing)

        self._preprocessedSamples: List['Sample'] = []

    def startPreprocessing(self, samples: List['Sample'], preprocessors: List['Preprocessor']) -> None:
        self._queue = Queue()
        self._stopEvent = Event()
        self._process = Process(target=preprocessSamples, args=(samples, preprocessors, self._queue, self._stopEvent))
        self._progressbar.setValue(0)
        self._progressbar.setMaximum(len(samples))
        self._timer.start(100)
        self.setWindowTitle(f"Preprocessing {len(samples)} sample(s) with {len(preprocessors)} preprocessor(s).")
        self._process.start()
        self._btnCancel.setEnabled(True)
        self.show()

    def _checkOnPreprocessing(self) -> None:
        if not self._queue.empty():
            queueContent = self._queue.get()
            if type(queueContent) == Sample:
                self._preprocessedSamples.append(queueContent)
                self._incrementProgressbar()

            if len(self._preprocessedSamples) == self._progressbar.maximum():
                self.PreprocessingFinished.emit(self._preprocessedSamples)
                self._finishProcessing()

    def _incrementProgressbar(self) -> None:
        """
        Increments the progressbar's status by one.
        """
        self._progressbar.setValue(self._progressbar.value() + 1)

    def _promptForCancel(self) -> None:
        if self._process.is_alive():
            reply = QtWidgets.QMessageBox.question(self, "Abort?", "Abort the preprocessing?",
                                                   QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                   QtWidgets.QMessageBox.No)

            if reply == QtWidgets.QMessageBox.Yes:
                self._stopEvent.set()
                self._progressbar.setMaximum(0)
                self._progressbar.setValue(0)
                self.setWindowTitle("Aborting Preprocessing, please wait..")
                self._btnCancel.setEnabled(False)
                self._finishProcessing(aborted=True)

    def _finishProcessing(self, aborted: bool = False) -> None:
        self._timer.stop()
        self._queue.close()
        self._process.join(timeout=2 if aborted else None)
        self._progressbar.setValue(0)
        self._preprocessedSamples = []
        if aborted:
            self.PreprocessingAborted.emit()

        self.hide()


def preprocessSamples(samples: List['Sample'], preprocessors: List['Preprocessor'],
                      queue: Queue, stopEvent: Event) -> None:
    """
    Preprocess a list of samples with the list of preprocessors.
    :param samples: List of SampleData objects to process
    :param preprocessors: List of Preprocessors to use
    :param queue: The dataQueue to put preprocessed samples into
    :param stopEvent: Event to check for process abortion
    """
    for sample in samples:
        if stopEvent.is_set():
            return

        specObj: 'SpectraObject' = sample.specObj
        backgroundInd: Set[int] = sample.getBackroundIndices()
        time.sleep(3)
        specObj.doPreprocessing(preprocessors, backgroundInd)

        queue.put(sample)
