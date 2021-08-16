import os
from PyQt5 import QtWidgets, QtCore
import numpy as np
from typing import *

from logger import getLogger
from gui.sampleview import MultiSampleView
from gui.graphOverlays import GraphView
from gui.spectraPlots import ResultPlots
from gui.preprocessEditor import PreprocessingSelector
from gui.classification import ClassCreator, ClassifierWidget


if TYPE_CHECKING:
    from logging import Logger
    from preprocessors import Preprocessor
    from SpectraProcessing.descriptors import DescriptorLibrary
    from gui.sampleview import SampleView


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("HSI Classifier")
        self._logger: 'Logger' = getLogger("MainWindow")

        self._multiSampleView: MultiSampleView = MultiSampleView(self)
        self._preprocSelector: PreprocessingSelector = PreprocessingSelector()
        self._resultPlots: ResultPlots = ResultPlots()
        self._clsCreator: ClassCreator = ClassCreator()
        self._clfWidget: ClassifierWidget = ClassifierWidget(self)
        self._loadBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Load")
        self._exportBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Export")

        self._toolbar: QtWidgets.QToolBar = QtWidgets.QToolBar()
        self.addToolBar(QtCore.Qt.TopToolBarArea, self._toolbar)

        self._configureWidgets()
        self._createLayout()
        self.disableWidgets()
        # self._loadFile(r"C:\Users\xbrjos\Desktop\Unsynced Files\IMEC HSI\Telecentric 2x\PE, PS, PET_corrected.npy")

    def setupConnections(self, sampleView: 'SampleView') -> None:
        sampleView.Activated.connect(self._resultPlots.updatePlots)
        sampleView.Renamed.connect(self._resultPlots.updatePlots)

        graphView: 'GraphView' = sampleView.getGraphView()
        graphView.SelectionChanged.connect(self._resultPlots.updatePlots)
        self._clfWidget.ClassTransparencyUpdated.connect(graphView.updateClassImgTransp)
        self._clsCreator.ClassDeleted.connect(sampleView.removeClass)
        sampleView.ClassDeleted.connect(graphView.removeColorOfClass)

    def getColorOfClass(self, className: str) -> Tuple[int, int, int]:
        """
        Returns a unique color for the class. Color is returned in 0-255 int values for R, G, B.
        """
        return self._clsCreator.getColorOfClassName(className)

    def classIsVisible(self, className: str) -> bool:
        """
        Returns, if the given class is set to visible or not-visible in the class selector.
        """
        return self._clsCreator.getClassVisibility(className)

    def getresultPlots(self) -> 'ResultPlots':
        return self._resultPlots

    def getLabelledSpectraFromActiveView(self) -> Dict[str, np.ndarray]:
        """
        Gets the currently labelled Spectra from the currently active sampleview.
        :return: Dictionary[className, (NxM) specArray of N spectra with M wavenumbers
        """
        return self._multiSampleView.getLabelledSpectraFromActiveView()

    def getLabelledSpectraFromAllViews(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Gets the currently labelled Spectra from all active samples, grouped i a dictionary with samplename as key
        :return:
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

    def getPreprocessors(self) -> List['Preprocessor']:
        return self._preprocSelector.getPreprocessors()

    def getWavenumbers(self) -> np.ndarray:
        return self._multiSampleView.getWavenumbers()

    def getCurrentColor(self) -> Tuple[int, int, int]:
        return self._clsCreator.getCurrentColor()

    def getCurrentClass(self) -> str:
        return self._clsCreator.getCurrentClass()

    def _getFileAndOpen(self) -> None:
        fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select Files",
                                                           r"C:\Users\xbrjos\Desktop\Unsynced Files\IMEC HSI", filter="*.npy")
        for fname in fnames:
            self._loadFile(fname)

    def _loadFile(self, fname: str) -> None:
        name = os.path.basename(fname.split('.npy')[0])
        try:
            cube: np.ndarray = np.load(fname)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "File Error", f"Could not load file {fname}."
                                                               f"\nError is:\n{e}")
            return
        newView: 'SampleView' = self._multiSampleView.addSampleView()
        newView.setUp(name, cube)
        self._logger.info(f"Loaded sample: {name}")
        self.enableWidgets()
        self.showMaximized()

    def getDescriptorLibrary(self) -> 'DescriptorLibrary':
        return self._resultPlots.getDecsriptorLibrary()

    def _export(self) -> None:
        raise NotImplementedError
        # folderPath: str = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Target Directory")
        # if folderPath:
        #     classes, colors = self._clsCreator.getClassNamesAndColors()
        #     uniqueClasses: List[str] = list(np.unique(classes))
        #
        #     cube: np.ndarray = self._specObj.getCube()
        #     assignments: List[int] = []
        #     specs: List[np.ndarray] = []
        #     for cls, rgb in zip(classes, colors):
        #         pixels = self._graphView.getPixelsOfColor(rgb)
        #         for y, x in zip(pixels[0], pixels[1]):
        #             specs.append(cube[:, y, x])
        #             assignments.append(uniqueClasses.index(cls))
        #
        #     specs: np.ndarray = np.array(specs)
        #     self._logger.info(f"Saving spectra to {folderPath}")
        #
        #     np.save(os.path.join(folderPath, f"Assignments of {self._name}.npy"), assignments)
        #     np.save(os.path.join(folderPath, f"Spectra of {self._name}.npy"), specs)
        #     np.savetxt(os.path.join(folderPath, f"Assignments of {self._name}.txt"), assignments)
        #     np.savetxt(os.path.join(folderPath, f"Spectra of {self._name}.txt"), specs)

    def getClassesAndPixels(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        :return: Tuple: ClassName: Tuple[array of y-coordinates, array of x-coordinates]
        """
        raise NotImplementedError
        # classPixels: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        # classes, colors = self._clsCreator.getClassNamesAndColors()
        # for cls, rgb in zip(classes, colors):
        #     classPixels[cls] = self._graphView.getPixelsOfColor(rgb)
        # return classPixels

    def enableWidgets(self) -> None:
        for widget in self._getUIWidgets():
            widget.setDisabled(False)

    def disableWidgets(self) -> None:
        for widget in self._getUIWidgets():
            widget.setDisabled(True)

    def _configureWidgets(self) -> None:
        """
        Sets parameters to the widgets of that window and establishes connections.
        :return:
        """
        self._multiSampleView.SampleClosed.connect(self._resultPlots.updatePlots)

        self._preprocSelector.ProcessorStackUpdated.connect(self._resultPlots.updatePlots)

        self._clsCreator.ClassDeleted.connect(self._resultPlots.updatePlots)
        self._clsCreator.ClassActivated.connect(self._resultPlots.switchToDescriptorSet)
        self._clsCreator.ClassVisibilityChanged.connect(self._resultPlots.updatePlots)
        self._clsCreator.setMaximumWidth(300)

        self._clfWidget.setMaximumWidth(300)

        self._loadBtn.released.connect(self._getFileAndOpen)
        self._exportBtn.released.connect(self._export)

        self._toolbar.addWidget(self._loadBtn)
        self._toolbar.addSeparator()
        self._toolbar.addWidget(self._exportBtn)

        self._resultPlots.setMainWinRef(self)

    def _createLayout(self) -> None:
        """
        Creates the actual window layout.
        :return:
        """
        clsLayout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        clsLayout.addWidget(self._clsCreator)
        clsLayout.addStretch()
        clsLayout.addWidget(self._clfWidget)

        specLayout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        specLayout.addWidget(self._preprocSelector)
        specLayout.addWidget(self._resultPlots)

        group: QtWidgets.QGroupBox = QtWidgets.QGroupBox()
        layout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        group.setLayout(layout)
        self.setCentralWidget(group)

        layout.addLayout(clsLayout)
        layout.addWidget(self._multiSampleView)
        layout.addLayout(specLayout)

    def _getUIWidgets(self) -> List[QtWidgets.QWidget]:
        widgetList = [self._exportBtn, self._resultPlots, self._clfWidget, self._clsCreator]
        return widgetList

    # def closeEvent(self, a0: QtGui.QCloseEvent) -> None:  # TODO: REIMPLEMENT
    #     self._saveViewToFile(self._getSaveFileName())


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("HSI Evaluator")
    win = MainWindow()
    win.show()
    app.exec_()


if __name__ == '__main__':
    main()
