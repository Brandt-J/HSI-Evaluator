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


from unittest import TestCase
import numpy as np
import sys
from typing import Dict, Union
from PyQt5 import QtWidgets

from gui.spectraPlots import ResultPlots
from gui.scatterPlot import getXYOfName
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
            onlyTheColors: np.ndarray = getXYOfName(colors[i, :], allColors, allColors)
            self.assertEqual(onlyTheColors.shape[0], numColorsPerColor[i])
            for j in range(onlyTheColors.shape[0]):
                self.assertTrue(np.array_equal(onlyTheColors[j, :], colors[i, :]))
