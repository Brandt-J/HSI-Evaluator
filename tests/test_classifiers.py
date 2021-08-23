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

import sys
from PyQt5 import QtWidgets
from unittest import TestCase
import numpy as np
from typing import *

from classifiers import SVM
from spectraObject import SpectraObject
from dataObjects import Sample
from gui.classification import ClassificationUI
from gui.sampleview import SampleView


class TestClassifiers(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication(sys.argv)

    def test_SetUniqueLabels(self) -> None:
        allLabels: np.ndarray = np.array(['class1']*10 + ['class2']*20 + ['class3']*30)
        np.random.shuffle(allLabels)
        labelsTrain: np.ndarray = allLabels[:40]
        labelsTest: np.ndarray = allLabels[40:]

        svm: SVM = SVM()
        self.assertDictEqual(svm._uniqueLabels, {})
        svm._setUniqueLabels(labelsTest, labelsTrain)
        self.assertDictEqual(svm._uniqueLabels, {'class1': 0,
                                                 'class2': 1,
                                                 'class3': 2})

    def test_runClassification(self) -> None:
        classUI: ClassificationUI = ClassificationUI(MockMainWin())
        classUI._classifyImage()
        # Make sure it does not fail


class MockMainWin:
    cubeShape = (10, 20, 20)

    def __init__(self):
        data1: Sample = Sample()
        data1.name = 'Sample1'
        data1.classes2Indices = {"class1": np.arange(20),
                                 "class2": np.arange(20)+20}
        sample1: SampleView = SampleView()
        sample1.setSampleData(data1)
        sample1.setCube(np.random.rand(self.cubeShape[0], self.cubeShape[1], self.cubeShape[2]))
        self._samples: List['SampleView'] = [sample1]

    def disableWidgets(self):
        pass

    def enableWidges(self):
        pass

    def getPreprocessors(self):
        return []

    def getAllSamples(self) -> List['SampleView']:
        return self._samples

    # def getSampleOfName(self) -> 'SampleView':

    def getClassColorDict(self) -> Dict[str, Tuple[int, int, int]]:
        return {"class1": (0, 0, 0),
                "class2": (255, 255, 255)}

    def getBackgroundsOfAllSamples(self) -> Dict[str, np.ndarray]:
        backgrounds: Dict[str, np.ndarray] = {}
        for sample in self._samples:
            backgrounds[sample.getName()] = np.zeros(self.cubeShape[0])
        return backgrounds





