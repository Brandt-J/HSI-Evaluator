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
from PyQt5 import QtWidgets
import sys
from typing import *
import numpy as np
from collections import Counter

from gui.preprocessEditor import PreprocessingSelector
from gui.spectraPlots import ResultPlots


class TestPreprocessingEditor(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)

    def test_limitSpecNumber(self):
        def clsNameFromInt(integer: int) -> str:
            return f"class_{integer+1}"

        resPlots: ResultPlots = ResultPlots()
        selector: PreprocessingSelector = PreprocessingSelector(None, resPlots)

        numClasses, specsPerClass = 3, 50
        specs: np.ndarray = np.random.rand(numClasses*specsPerClass, 5)
        labels: List[str] = []
        for i in range(numClasses):
            labels += [clsNameFromInt(i)] * specsPerClass
        labels: np.ndarray = np.array(labels)

        sampleNames: np.ndarray = np.array([f"sample_{i}" for i in range(numClasses*specsPerClass)])

        maxNumberPerClass: int = 20
        resPlots._numSpecSpinner.setValue(maxNumberPerClass)
        newSpecs, newLabels, newSampleNames = selector._limitToMaxNumber(specs, labels, sampleNames)
        self.assertTrue(newSpecs.shape[0] == len(newLabels) == len(newSampleNames) == maxNumberPerClass*numClasses)

        counter: Counter = Counter(newLabels)
        for i in range(numClasses):
            clsName = clsNameFromInt(i)
            self.assertEqual(counter[clsName], maxNumberPerClass)

        for i, (label, sampleName) in enumerate(zip(newLabels, newSampleNames)):
            indSampleName: int = int(sampleName.split('_')[1])
            self.assertTrue(np.array_equal(newSpecs[i, :], specs[indSampleName, :]))
