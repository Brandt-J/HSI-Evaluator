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

import os
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
from typing import *
import pickle

from logger import getLogger
from dataObjects import View, getFilePathHash
from loadCube import loadCube
from legacyConvert import assertUpToDateView
from gui.sampleview import MultiSampleView
from gui.graphOverlays import GraphView
from gui.spectraPlots import ResultPlots
from gui.preprocessEditor import PreprocessingSelector
from gui.classification import ClassCreator, ClassificationUI


if TYPE_CHECKING:
    from logging import Logger
    from preprocessing.preprocessors import Preprocessor
    from gui.sampleview import SampleView
    from spectraObject import SpectraCollection


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("HSI Classifier")
        self._logger: 'Logger' = getLogger("MainWindow")

        self._multiSampleView: MultiSampleView = MultiSampleView(self)
        self._resultPlots: ResultPlots = ResultPlots()
        self._preprocSelector: PreprocessingSelector = PreprocessingSelector(self, self._resultPlots)
        self._clsCreator: ClassCreator = ClassCreator()
        self._clfWidget: ClassificationUI = ClassificationUI(self)
        self._clfWidget.setDisabled(True)
        self._saveViewAct: QtWidgets.QAction = QtWidgets.QAction("&Save View")
        self._exportSpecAct: QtWidgets.QAction = QtWidgets.QAction("&Export Spectra to ASCII")

        self._configureWidgets()
        self._createMenuBar()
        self._createLayout()
        self.disableWidgets()

    def setupConnections(self, sampleView: 'SampleView') -> None:
        sampleView.Activated.connect(self._resultPlots.updatePlots)
        sampleView.Renamed.connect(self._resultPlots.updatePlots)
        sampleView.BackgroundSelectionChanged.connect(self._clfWidget.forcePreprocessing)

        graphView: 'GraphView' = sampleView.getGraphView()
        graphView.SelectionChanged.connect(self._resultPlots.updatePlots)
        graphView.SelectionChanged.connect(self._preprocSelector.updatePreviewSpectra)

        self._clfWidget.ClassTransparencyUpdated.connect(graphView.updateClassImgTransp)
        self._clsCreator.ClassDeleted.connect(sampleView.removeClass)
        sampleView.ClassDeleted.connect(graphView.removeColorOfClass)

    def classIsVisible(self, className: str) -> bool:
        """
        Returns, if the given class is set to visible or not-visible in the class selector.
        """
        return self._clsCreator.getClassVisibility(className)

    def checkForRequiredClasses(self, classes: List[str]) -> None:
        """Makes sure that the given classes are present in the class creator. Missing ones are created."""
        self._clsCreator.checkForRequiredClasses(classes)

    def getColorOfClass(self, className: str) -> Tuple[int, int, int]:
        """
        Returns a unique color for the class. Color is returned in 0-255 int values for R, G, B.
        """
        return self._clsCreator.getColorOfClassName(className)

    def getClassColorDict(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Gets a dictionary containing the colors of all present classes.
        """
        return self._clsCreator.getClassColorDict()

    def getresultPlots(self) -> 'ResultPlots':
        return self._resultPlots

    def getLabelledSpectraFromActiveView(self) -> 'SpectraCollection':
        """
        Gets the currently labelled Spectra from the currently active sampleview.
        :return: Spectra Collection with all data
        """
        return self._multiSampleView.getLabelledSpectraFromActiveView()

    def getLabelledSpectraFromAllViews(self) -> 'SpectraCollection':
        """
        Gets the currently labelled Spectra from all active samples, grouped i a dictionary with samplename as key
        :return: Spectra Collection with all data
        """
        return self._multiSampleView.getLabelledSpectraFromAllViews()

    def getBackgroundOfActiveSample(self) -> np.ndarray:
        """
        Returns the averaged background spectrum of the currently active sample.
        """
        return self._multiSampleView.getBackgroundOfActiveSample()

    def getBackgroundsOfAllSamples(self) -> Dict[str, np.ndarray]:
        """
        Returns the averaged backgounds of all samples.
        """
        return self._multiSampleView.getBackgroundsOfAllSamples()

    def getAllSamples(self) -> List['SampleView']:
        """
        Returns a list of the opened samples.
        """
        return self._multiSampleView.getSampleViews()

    def getActiveSample(self) -> 'SampleView':
        """
        Returns the currently active sampleview.
        """
        return self._multiSampleView.getActiveSample()

    def getPreprocessors(self) -> List['Preprocessor']:
        """
        Returns the stack of Preprocessors for the current classification setup (i.e., as connected to the "Classification"
        node in the nodegraph.
        """
        return self._preprocSelector.getPreprocessors()

    def getWavelengths(self) -> np.ndarray:
        return self._multiSampleView.getWavelengths()

    def getCurrentColor(self) -> Tuple[int, int, int]:
        return self._clsCreator.getCurrentColor()

    def getCurrentClass(self) -> str:
        return self._clsCreator.getCurrentClass()

    def _promptLoadNPYSample(self) -> None:
        """Prompts for a npy file to open as sample"""
        fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select Files",
                                                           r"C:\Users\xbrjos\Desktop\Unsynced Files\IMEC HSI",
                                                           filter="*.npy *.hdr")
        for fname in fnames:
            self._loadFile(fname)

    def _loadFile(self, fname: str) -> None:
        name = os.path.basename(fname.split('.npy')[0])
        savedSampleDir: str = self._multiSampleView.getSampleSaveDirectory()
        savedSamplePath: str = os.path.join(savedSampleDir, getFilePathHash(fname) + '.pkl')

        if os.path.exists(savedSamplePath):
            self._multiSampleView.loadSampleViewFromFile(savedSamplePath)
            self._logger.info(f"Loading saved status for sample {name}")
        else:
            cube, wavelengths = loadCube(fname)
            newView: 'SampleView' = self._multiSampleView.addSampleView()
            newView.setUp(fname, cube, wavelengths)
            self._logger.info(f"Creating new sample from: {name}")

        self.enableWidgets()
        self.showMaximized()
        self._clfWidget.forcePreprocessing()

    def getDescriptorLibrary(self) -> 'DescriptorLibrary':
        return self._resultPlots.getDecsriptorLibrary()

    def _promptLoadView(self) -> None:
        """Prompts for a view file to open"""
        directory: str = self._multiSampleView.getViewSaveDirectory()
        loadPath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select File", directory, "*.view")
        if loadPath:
            self._loadView(loadPath)

    def _promptSaveView(self) -> None:
        """Prompts for saving the current view. See method in multisampleview for details."""
        directory: str = self._multiSampleView.getViewSaveDirectory()
        savePath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select File", directory, "*.view")
        if savePath:
            self._saveView(savePath)
            QtWidgets.QMessageBox.about(self, "Done", "Saved the current view.")

    def _saveView(self, savePath: str) -> None:
        """
        Saves the current view (all open samples, the preprocessor and the notepad into a file.
        :param savePath: the full path to where to save the view
        """
        viewObj: View = View()
        viewObj.samples = [sample.getSampleDataToSave() for sample in self._multiSampleView.getSampleViews()]
        # viewObj.processStack = self._preprocSelector.getPreprocessorNames()  # TODO: Reimplement
        viewObj.title = os.path.basename(savePath.split(".")[0])
        with open(savePath, "wb") as fp:
            pickle.dump(viewObj, fp)

    def _loadView(self, fname: str) -> None:
        """
        Loads a view.
        :fname: full path to the file to load.
        """
        with open(fname, "rb") as fp:
            loadedView: View = pickle.load(fp)
            loadedView = assertUpToDateView(loadedView)
        view: View = View()
        view.__dict__.update(loadedView.__dict__)

        if view.title != '':
            self.setWindowTitle(f"HSI Evaluator - {view.title}")
        self._multiSampleView.createListOfSamples(view.samples)
        self._preprocSelector.selectPreprocessors(view.processStack)
        self.enableWidgets()
        self._resultPlots.updatePlots()
        self._clfWidget.forcePreprocessing()
        self.showMaximized()

    def _export(self) -> None:
        raise NotImplementedError

    def enableWidgets(self) -> None:
        for widget in self._getUIWidgetsForSelectiveEnabling():
            widget.setDisabled(False)

    def disableWidgets(self) -> None:
        for widget in self._getUIWidgetsForSelectiveEnabling():
            widget.setDisabled(True)

    def _configureWidgets(self) -> None:
        """
        Sets parameters to the widgets of that window and establishes connections.
        :return:
        """
        self._multiSampleView.SampleClosed.connect(self._resultPlots.updatePlots)

        self._preprocSelector.ProcessorStackUpdated.connect(self._resultPlots.updatePlots)
        self._preprocSelector.ProcessorStackUpdated.connect(self._clfWidget.forcePreprocessing)

        self._clsCreator.ClassDeleted.connect(self._resultPlots.updatePlots)
        self._clsCreator.ClassActivated.connect(self._resultPlots.switchToDescriptorSet)
        self._clsCreator.ClassVisibilityChanged.connect(self._resultPlots.updatePlots)
        self._clsCreator.setMaximumWidth(300)

        self._clfWidget.setMaximumWidth(300)
        self._resultPlots.setMainWinRef(self)

    def _createMenuBar(self) -> None:
        """Creates the Menu bar"""
        filemenu: QtWidgets.QMenu = QtWidgets.QMenu("&File", self)
        openAct: QtWidgets.QAction = QtWidgets.QAction("&Open Sample(s)", self)
        openAct.setShortcut("Ctrl+O")
        openAct.triggered.connect(self._promptLoadNPYSample)

        loadViewAct: QtWidgets.QAction = QtWidgets.QAction("Open &View", self)
        loadViewAct.triggered.connect(self._promptLoadView)
        loadViewAct.setShortcut("Ctrl+Shift+O")

        self._saveViewAct.triggered.connect(self._promptSaveView)
        self._saveViewAct.setShortcut("Ctrl+S")

        self._exportSpecAct.triggered.connect(self._multiSampleView.exportSpectra)
        self._exportSpecAct.setShortcut("Ctrl+E")

        closeAct: QtWidgets.QAction = QtWidgets.QAction("Close &Program", self)
        closeAct.triggered.connect(self.close)

        filemenu.addAction(openAct)
        filemenu.addSeparator()
        filemenu.addAction(loadViewAct)
        filemenu.addAction(self._saveViewAct)
        filemenu.addAction(self._exportSpecAct)
        filemenu.addSeparator()
        filemenu.addAction(closeAct)

        self.menuBar().addMenu(filemenu)

    def _createLayout(self) -> None:
        """
        Creates the actual window layout.
        :return:
        """
        clsLayout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        clsLayout.addWidget(self._clsCreator)
        clsLayout.addStretch()
        clsLayout.addWidget(self._clfWidget)

        splitter1: QtWidgets.QSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter1.addWidget(self._preprocSelector)
        splitter1.addWidget(self._resultPlots)

        splitter2: QtWidgets.QSplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter2.addWidget(self._multiSampleView)
        splitter2.addWidget(splitter1)

        group: QtWidgets.QGroupBox = QtWidgets.QGroupBox()
        layout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        group.setLayout(layout)
        self.setCentralWidget(group)

        layout.addLayout(clsLayout)
        layout.addWidget(splitter2)

    def _getUIWidgetsForSelectiveEnabling(self) -> List[QtWidgets.QWidget]:
        """
        Returns a list of ui widgets that can be disabled and enabled, depedning on current ui context
        """
        widgetList = [self._saveViewAct, self._multiSampleView, self._resultPlots, self._clfWidget, self._clsCreator,
                      self._exportSpecAct, self._preprocSelector]
        return widgetList

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        reply = QtWidgets.QMessageBox.question(self, "Close Program", "Are you sure you want to close?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self._multiSampleView.saveSamples()
            self._multiSampleView.closeAllSamples()
            a0.accept()


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("HSI Evaluator")
    win = MainWindow()
    win.showMaximized()
    app.exec_()


if __name__ == '__main__':
    main()
