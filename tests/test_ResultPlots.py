from unittest import TestCase
import numpy as np
import sys
from typing import Dict, Union
from PyQt5 import QtWidgets

from gui.spectraPlots import ResultPlots
from gui.pcaPlot import getXYOfColor
from gui.HSIEvaluator import MainWindow


class TestSpectraPreview(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)

    def setUp(self) -> None:
        self._mainWin: MainWindow = MainWindow()
        self._plotWin: ResultPlots = ResultPlots()
        self._plotWin.setMainWinRef(self._mainWin)
        self._specPlot = self._plotWin._specPlot

    def test_limitSpecNumber(self):
        maxSpecs: int = 10  # not more than 10 spectra can be taken
        self._plotWin._numSpecSpinner.setValue(maxSpecs)

        fewSpecs: np.ndarray = np.random.rand(5, 10)
        fewSpecsClipped: np.ndarray = self._plotWin._limitSpecNumber(fewSpecs)
        self.assertTrue(np.array_equal(fewSpecs, fewSpecsClipped))

        moreSpecs: np.ndarray = np.random.rand(30, 10)
        moreSpecsClipped: np.ndarray = self._plotWin._limitSpecNumber(moreSpecs)
        self.assertTrue(moreSpecsClipped.shape[0] == maxSpecs)
        self.assertTrue(moreSpecsClipped.shape[1] == moreSpecs.shape[1])

    def test_plotActiveSpectra(self):
        # create fake data and override main win functions
        numWavenums, numSpecs = 10, 30
        sample1: Dict[str, np.ndarray] = {'class1': np.random.rand(numSpecs, numWavenums),
                                          'class2': np.random.rand(numSpecs, numWavenums)}
        sample2: Dict[str, np.ndarray] = {'class1': np.random.rand(numSpecs, numWavenums),
                                          'class2': np.random.rand(numSpecs, numWavenums),
                                          'class3': np.random.rand(numSpecs, numWavenums)}

        self._mainWin.getWavenumbers = lambda: np.arange(numWavenums)
        self._mainWin.getBackgroundOfActiveSample = lambda: np.zeros(numWavenums)
        self._mainWin.getBackgroundsOfAllSamples = lambda: {'sample1': np.zeros(numWavenums),
                                                            'sample2': np.zeros(numWavenums),
                                                            'sample3': np.zeros(numWavenums)}
        self._mainWin.getLabelledSpectraFromActiveView = lambda: sample1
        self._mainWin.getLabelledSpectraFromAllViews = lambda: {'sample1': sample1,
                                                                'sample2': sample2}

        self._mainWin.getPreprocessors = lambda: []  # we don't take any here...

        self._plotWin._numSpecSpinner.setValue(numSpecs)
        self._plotWin._showAllCheckBox.setChecked(False)
        self._plotWin.updatePlots()  # make sure no errors occur

        self._plotWin._showAllCheckBox.setChecked(True)
        self._plotWin.updatePlots()  # make sure no errors occur

    def test_prepareSpectraForPLotting(self) -> None:
        specs: np.ndarray = np.random.rand(20, 10)
        self._specPlot._avgCheckBox.setChecked(False)
        prepared: np.ndarray = self._specPlot._prepareSpecsForPlot(specs)
        self.assertTrue(np.array_equal(specs, prepared.transpose()))

        self._specPlot._avgCheckBox.setChecked(True)
        prepared: np.ndarray = self._specPlot._prepareSpecsForPlot(specs)
        meanSpec: np.ndarray = np.mean(specs, axis=0)
        self.assertTrue(np.array_equal(meanSpec, prepared))

    def test_getXYOfColor(self) -> None:
        numDifferentColors: int = 4
        colors: np.ndarray = np.random.rand(numDifferentColors, 3)  # four rgb colors
        numColorsPerColor: np.ndarray = np.random.randint(low=5, high=10, size=numDifferentColors)

        allColors: Union[None, np.ndarray] = None
        for i in range(numDifferentColors):
            stackedColor: np.ndarray = np.tile(colors[i, :], (numColorsPerColor[i], 1))
            if i == 0:
                allColors = stackedColor
            else:
                allColors = np.vstack((allColors, stackedColor))

        for i in range(numDifferentColors):
            # we just take the colors also as data, that makes testing easy
            onlyTheColors: np.ndarray = getXYOfColor(colors[i, :], allColors, allColors)
            self.assertEqual(onlyTheColors.shape[0], numColorsPerColor[i])
            for j in range(onlyTheColors.shape[0]):
                self.assertTrue(np.array_equal(onlyTheColors[j, :], colors[i, :]))
