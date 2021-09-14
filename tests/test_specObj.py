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

from spectraObject import SpectraObject, splitUpArray, SpectraCollection
from preprocessing.preprocessors import getPreprocessors


class TestSpecObject(TestCase):
    def test_Preprocessing(self):
        specObj: SpectraObject = SpectraObject()
        cubeShape: Tuple[int, int, int] = (100, 3, 3)
        specObj.setCube(np.random.rand(cubeShape[0], cubeShape[1], cubeShape[2]))
        specObj.preparePreprocessing(getPreprocessors(), np.zeros(cubeShape[0]))
        specObj.applyPreprocessing()  # Make sure all preprocessors run nicely. Here we do single process

        specObj: SpectraObject = SpectraObject()
        cubeShape: Tuple[int, int, int] = (100, 50, 50)
        specObj.setCube(np.random.rand(cubeShape[0], cubeShape[1], cubeShape[2]))
        specObj.preparePreprocessing(getPreprocessors(), np.zeros(cubeShape[0]))
        specObj.applyPreprocessing()  # Make sure all preprocessors run nicely. Now we run multiprocessing

    def test_SplitSpecArr(self):
        specArr: np.ndarray = np.random.rand(100, 30)
        for num in range(1, 10):
            arrList: List[np.ndarray] = splitUpArray(specArr, numParts=num)
            self.assertEqual(len(arrList), num)
            self.assertTrue(np.array_equal(specArr[0, :], arrList[0][0, :]))  # first entry is identical
            self.assertTrue(np.array_equal(specArr[-1, :], arrList[-1][-1, :]))  # last entry is identical

    def test_IgnoreBackground(self):
        specObj: SpectraObject = SpectraObject()
        cubeShape: Tuple[int, int, int] = (100, 20, 20)
        testCube: np.ndarray = np.random.rand(cubeShape[0], cubeShape[1], cubeShape[2])

        numPixels: int = cubeShape[1] * cubeShape[2]
        backgroundIndices: Set[int] = set(np.random.randint(0, 100, 15))
        for ind in backgroundIndices:
            y, x = np.unravel_index(ind, cubeShape[1:])
            testCube[:, y, x] = 100.0  # This value is not present in the cube before (np.random procudes values between 0 and 1)

        specObj.setCube(testCube)
        numBackgroundIndices: int = len(backgroundIndices)
        specObj._backgroundIndices = backgroundIndices

        specArr: np.ndarray = specObj._cube2SpecArr(ignoreBackground=False)
        numHundreds = len(np.where(specArr == 100)[0])
        self.assertEqual(len(specArr), numPixels)
        self.assertEqual(numHundreds, numBackgroundIndices*cubeShape[0])

        specArr = specObj._cube2SpecArr(ignoreBackground=True)
        self.assertEqual(len(specArr), numPixels-numBackgroundIndices)
        numHundreds = len(np.where(specArr == 100)[0])
        self.assertEqual(numHundreds, 0)

        reconstructedCube: np.ndarray = specObj._specArr2cube(specArr, ignoreBackground=True)
        numHundreds = len(np.where(reconstructedCube == 100)[0])
        self.assertEqual(numHundreds, numBackgroundIndices*cubeShape[0])
        self.assertTrue(np.array_equal(reconstructedCube, testCube))


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


