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
from typing import *
import numpy as np
from PyQt5 import QtWidgets
import sys

from spectraObject import SpectraObject, SpectraCollection, WavelengthsNotSetError
from preprocessing.preprocessors import splitUpArray
from gui.nodegraph.nodes import nodeTypes
from gui.nodegraph.nodegraph import NodeGraph
if TYPE_CHECKING:
    from preprocessing.preprocessors import Preprocessor
    from gui.nodegraph.nodecore import BaseNode


def getPreprocessors() -> List['Preprocessor']:
    """
    Gets the available preprocessors for spectral data.
    """
    graph: NodeGraph = NodeGraph()
    preprocList: List['Preprocessor'] = []
    for nodetype in nodeTypes.values():
        node: 'BaseNode' = nodetype(graph, None)
        if node.getPreprocessor() is not None:
            preprocList.append(node.getPreprocessor())
    return preprocList


app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)


def fakeSpecProcess(spectra: np.ndarray) -> np.ndarray:
    spectra += 1
    return spectra


class TestSpecObject(TestCase):
    def test_SplitSpecArr(self):
        specArr: np.ndarray = np.random.rand(100, 30)
        for num in range(1, 10):
            arrList: List[np.ndarray] = splitUpArray(specArr, numParts=num)
            self.assertEqual(len(arrList), num)
            self.assertTrue(np.array_equal(specArr[0, :], arrList[0][0, :]))  # first entry is identical
            self.assertTrue(np.array_equal(specArr[-1, :], arrList[-1][-1, :]))  # last entry is identical

    def test_remapToWavelengths(self) -> None:
        specObj: SpectraObject = SpectraObject()
        wavelengths: np.ndarray = np.arange(10)
        origcube: np.ndarray = np.zeros((len(wavelengths), 5, 5))
        for i in range(origcube.shape[0]):
            origcube[i, :, :] = i

        self.assertRaises(WavelengthsNotSetError, specObj.remapToWavelenghts, (wavelengths))  # Wavelengths were not yet set

        specObj.setCube(origcube.copy(), wavelengths)

        # shorter Wavelenghts
        shorterWavelenghts: np.ndarray = np.arange(8)
        specObj.remapToWavelenghts(shorterWavelenghts)
        shorterCube: np.ndarray = specObj._cube
        self.assertEqual(shorterCube.shape[0], len(shorterWavelenghts))
        for i in range(len(shorterWavelenghts)):
            uniqueInLayer: np.ndarray = np.unique(shorterCube[i, :, :])
            self.assertEqual(len(uniqueInLayer), 1)
            self.assertEqual(uniqueInLayer[0], i)

        self.assertTrue(np.array_equal(specObj._wavelengths, shorterWavelenghts))

        # reset cube
        specObj.setCube(origcube, wavelengths)

        # longer Wavelenghts
        longerWavelengths: np.ndarray = np.arange(15)
        specObj.remapToWavelenghts(longerWavelengths)
        longerCube: np.ndarray = specObj._cube
        self.assertEqual(longerCube.shape[0], len(longerWavelengths))
        origLen: int = origcube.shape[0]
        for i in range(len(longerWavelengths)):
            uniqueInLayer: np.ndarray = np.unique(longerCube[i, :, :])
            self.assertEqual(len(uniqueInLayer), 1)

            if i < origLen:
                self.assertEqual(uniqueInLayer[0], i)
            else:
                self.assertEqual(uniqueInLayer[0], origLen-1)

        self.assertTrue(np.array_equal(specObj._wavelengths, longerWavelengths))


class TestSpecCollection(TestCase):
    def test_addSpectra(self) -> None:
        specColl: SpectraCollection = SpectraCollection()
        testDict: Dict[str, np.ndarray] = {"class1": np.random.rand(10, 5),
                                           "class2": np.random.rand(20, 5)}

        specColl.addSpectraDict(testDict, "sample1")
        specColl.addSpectraDict(testDict, "sample2")

        expectedLabels = np.array(["class1"]*10 + ["class2"]*20 + ["class1"]*10 + ["class2"]*20)
        expectedSampleNames = np.array(["sample1"]*30 + ["sample2"]*30)
        expectedSpectra = np.vstack((testDict["class1"], testDict["class2"], testDict["class1"], testDict["class2"]))
        spectra, labels = specColl.getXY()

        self.assertEqual(spectra.shape[0], 60)  # 2 * (10 + 20)
        self.assertEqual(spectra.shape[1], 5)
        self.assertTrue(np.array_equal(expectedSpectra, spectra))

        self.assertEqual(len(labels), 60)
        self.assertTrue(np.array_equal(expectedLabels, labels))

        sampleNames: np.ndarray = specColl.getSampleNames()
        self.assertEqual(len(sampleNames), 60)
        self.assertTrue(np.array_equal(sampleNames, expectedSampleNames))


