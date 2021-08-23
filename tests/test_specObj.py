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

from spectraObject import SpectraObject, splitUpArray
from preprocessing.preprocessors import getPreprocessors


class TestSpecObject(TestCase):
    def test_Preprocessing(self):
        specObj: SpectraObject = SpectraObject()
        cubeShape: Tuple[int, int, int] = (100, 3, 3)
        specObj.setCube(np.random.rand(cubeShape[0], cubeShape[1], cubeShape[2]))
        specObj.preparePreprocessing(getPreprocessors(), np.zeros(cubeShape[0]))
        specObj.applyPreprocessing()  # Make sure all preprocessors run nicely. Here we do single process

        specObj: SpectraObject = SpectraObject()
        cubeShape: Tuple[int, int, int] = (100, 20, 20)
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
