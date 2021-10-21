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
import time
from unittest import TestCase
from PyQt5 import QtWidgets
import sys
from typing import *
import numpy as np
from collections import Counter

from logger import getLogger
from preprocessing.routines import NormMode
from gui.nodegraph.nodegraph import NodeGraph
from gui.preprocessEditor import PreprocessingSelector
from gui.spectraPlots import ResultPlots
from gui.nodegraph.nodes import NodeNormalize
from tests.test_classifiers import MockMainWin, getPreprocessors
if TYPE_CHECKING:
    from gui.preprocessEditor import PreprocessingPerformer
    from dataObjects import Sample


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

    def test_PreprocessUI(self) -> None:
        preprocEditor: PreprocessingSelector = PreprocessingSelector(MockMainWin(), MockResultsPlot())
        preprocEditor.getPreprocessors = getPreprocessors
        preprocEditor.applyPreprocessingToSpectra()

        preprocPerformer: 'PreprocessingPerformer' = preprocEditor._processingPerformer
        while preprocPerformer._thread.is_alive():
            time.sleep(0.1)

        for sampleInMainWin, preprocData in zip(preprocEditor._mainWin.getAllSamples(), preprocPerformer._preprocessedSamples):
            sample: 'Sample' = sampleInMainWin.getSampleData()
            self.assertTrue(preprocData == sample)
            self.assertTrue(np.array_equal(preprocData.specObj._preprocessedCube, sample.specObj._preprocessedCube))


class TestPreprocNodes(TestCase):
    def setUp(self) -> None:
        self._nodegraph: NodeGraph = NodeGraph()

    def testNormalize(self) -> None:
        normNode: NodeNormalize = NodeNormalize(self._nodegraph, getLogger("TestNodeNormalize"))
        data: np.ndarray = np.random.rand(10, 20)
        normNode._inputs[0].getValue = lambda: data  # overwrite input function to pipe in the data array

        for modeStr, mode in normNode.lbl2Mode.items():
            normNode._modeCombo.setCurrentText(modeStr)
            preprocData: np.ndarray = normNode.getOutput()
            self.assertTrue(np.array_equal(data.shape, preprocData.shape))
            for i in range(preprocData.shape[0]):
                if mode == NormMode.Max:
                    self.assertEqual(preprocData[i, :].max(), 1.0)
                elif mode == NormMode.Area:
                    self.assertAlmostEqual(getArea(preprocData[i, :]), 1.0)
                elif mode == NormMode.Length:
                    self.assertAlmostEqual(getLength(preprocData[i, :]), 1.0)


def getLength(arr: np.ndarray) -> float:
    return np.linalg.norm(arr)


def getArea(arr: np.ndarray) -> float:
    return np.trapz(arr)


class MockResultsPlot:
    def updateScatterPlot(self) -> None:
        pass

    def updateSpecPlot(self) -> None:
        pass
