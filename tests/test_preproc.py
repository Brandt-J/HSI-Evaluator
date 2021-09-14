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
from typing import List
from unittest import TestCase
from PyQt5 import QtWidgets
import sys
import numpy as np

from gui.preprocessEditor import PreprocessingSelector
from gui.spectraPlots import ResultPlots


class TestPreprocessingEditor(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)

    def test_limitSpecNumber(self):
        resPlots: ResultPlots = ResultPlots()
        selector: PreprocessingSelector = PreprocessingSelector(None, resPlots)

        specs: np.ndarray = np.random.rand(50, 5)
        labels: np.ndarray = np.array([f"class_{i+1}" for i in range(50)])
        sampleNames: np.ndarray = np.array([f"sample_{i + 1}" for i in range(50)])

        resPlots._numSpecSpinner.setValue(20)
        newSpecs, newLabels, newSampleNames = selector._limitToMaxNumber(specs, labels, sampleNames)
        self.assertTrue(newSpecs.shape[0] == len(newLabels) == len(newSampleNames) == 20)

        for i, (label, sampleName) in enumerate(zip(newLabels, newSampleNames)):
            indLabel: int = int(label.split('_')[1])
            indSampleName: int = int(sampleName.split('_')[1])
            self.assertEqual(indLabel, indSampleName)
            self.assertTrue(np.array_equal(newSpecs[i, :], specs[indLabel-1, :]))
