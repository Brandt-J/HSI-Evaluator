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
import dataclasses

from PyQt5 import QtCore
import numpy as np
from typing import List, Dict, Tuple, TYPE_CHECKING, Union, cast
import random
import time
from concurrent.futures import ProcessPoolExecutor

from logger import getLogger
from preprocessing.preprocessors import Background

if TYPE_CHECKING:
    from logging import Logger
    from preprocessing.preprocessors import Preprocessor


class SpectraObject:
    def __init__(self):
        self._wavenumbers: Union[None, np.ndarray] = None
        self._cube: Union[None, np.ndarray] = None
        self._preprocQueue: List['Preprocessor'] = []
        self._background: Union[None, np.ndarray] = None
        self._preprocessedCube: Union[None, np.ndarray] = None
        self._classes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # classname, (y-coordinages, x-coordinates)
        self._logger: 'Logger' = getLogger("SpectraObject")

    def setCube(self, cube: np.ndarray) -> None:
        self._cube = cube
        if self._wavenumbers is None:
            self._setDefaultWavenumbers(cube)

    def preparePreprocessing(self, preprocessingQueue: List['Preprocessor'], background: np.ndarray):
        """
        Sets the preprocessing parameters
        :param preprocessingQueue: List of Preprocessors
        :param background: np.ndarray of background spectrum
        """
        self._preprocQueue = preprocessingQueue
        self._background = background

    def applyPreprocessing(self) -> None:
        """
        Applies the specified preprocessing.

        :return:
        """
        if len(self._preprocQueue) > 0:
            specArr = self._cube2SpecArr()
            t0 = time.time()
            if len(specArr) < 1000:
                specArr = self._preprocessSpectaSingleProcess(specArr)
            else:
                specArr = self._preprocessSpectraMultiProcessing(specArr)

            self._preprocessedCube = self._specArr2cube(specArr)
            print(f'preprocessing spectra took {round(time.time()-t0, 2)} seconds')
        else:
            self._logger.warning("Received empty preprocessingQueue, just returning the original cube.")
            self._preprocessedCube = self._cube.copy()

        self._resetPreprocessing()

    def _preprocessSpectraMultiProcessing(self, specArr: np.ndarray) -> np.ndarray:
        """
        Preprocesses the given spectra array using a Process Pool Executor.
        :param specArr: (NxM) array of N spectra with M wavenumbers
        :return: processed (NxM) array
        """
        self._logger.debug(f"Preprocessing {len(specArr)} spectra with pool process executor.")
        preprocDatas: List[PreprocessData] = []
        splitArrays: List[np.ndarray] = splitUpArray(specArr, numParts=10)
        for partArray in splitArrays:
            preprocDatas.append(PreprocessData(partArray, self._preprocQueue, self._background))
        maxWorkers: int = 6
        with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
            result: List[np.ndarray] = list(executor.map(applyPreprocessing, preprocDatas))

        return recombineSpecArrays(result)

    def _preprocessSpectaSingleProcess(self, specArr: np.ndarray) -> np.ndarray:
        """
        Preprocesses the given spectra array without doing multiprocessing.
        :param specArr: (NxM) array of N spectra with M wavenumbers
        :return: processed (NxM) array
        """
        self._logger.debug(f"Preprocessing {len(specArr)} spectra in single process.")
        for preprocessor in self._preprocQueue:
            if type(preprocessor) == Background:
                preprocessor: Background = cast(Background, preprocessor)
                preprocessor.setBackground(self._background)
            specArr = preprocessor.applyToSpectra(specArr)
        return specArr

    def _resetPreprocessing(self) -> None:
        self._preprocQueue = []
        self._imgLimits = QtCore.QRectF()
        self._background = None

    def _specArr2cube(self, specArr: np.ndarray) -> np.ndarray:
        """
        Takes an (MxN) spec array and reformats into cube layout
        :param specArr: (MxN) array of M spectra with N wavenumbers
        :return: (NxXxY) cube array of X*Y spectra of N wavenumbers (M = X*Y)
        """
        cube = self._cube.copy()
        i = 0
        for y in range(self._cube.shape[1]):
            for x in range(self._cube.shape[2]):
                cube[:, y, x] = specArr[i, :]
                i += 1

        return cube

    def _cube2SpecArr(self) -> np.ndarray:
        """
        Reformats the cube into an MxN spectra matrix of M spectra with N wavenumbers
        :return: (MxN) spec array of M spectra of N wavenumbers
        """
        specArr: List[np.ndarray] = []
        for y in range(self._cube.shape[1]):
            for x in range(self._cube.shape[2]):
                specArr.append(self._cube[:, y, x])

        specArr: np.ndarray = np.array(specArr)  # NxM array of N specs with M wavenumbers
        assert specArr.shape[1] == self._cube.shape[0]
        return specArr

    def getClassSpectra(self, maxSpecPerClas: int, preprocessed: bool = True) -> Dict[str, np.ndarray]:
        """
        Gets the a dictionary for the spectra per class.
        :param maxSpecPerClas: Indicated for
        :param preprocessed: Whether or not to return the preprocessed or, alternatively, the raw spectra.
        :return:
        """
        spectra: Dict[str, np.ndarray] = {}
        random.seed(42)
        for cls_name, cls_indices in self._classes.items():
            curIndices: List[Tuple[int, int]] = list(zip(cls_indices[0], cls_indices[1]))
            if len(curIndices) > 0:
                if len(curIndices) > maxSpecPerClas:
                    curIndices = random.sample(curIndices, maxSpecPerClas)
                spectra[cls_name] = self._getSpecArray(curIndices, preprocessed)

        return spectra

    def getCube(self) -> np.ndarray:
        cube: np.ndarray = self._preprocessedCube
        if cube is None:
            cube = self._cube
        return cube

    def getNotPreprocessedCube(self) -> np.ndarray:
        return self._cube

    def getWavenumbers(self) -> np.ndarray:
        assert self._wavenumbers is not None, 'Wavenumbers have not yet been set! Cannot return them!'
        return self._wavenumbers

    def getAverageSpectra(self) -> Dict[str, np.ndarray]:
        specs: Dict[str, np.ndarray] = self.getClassSpectra(maxSpecPerClas=np.inf)
        avgSpecs: Dict[str, np.ndarray] = {}
        for cls_name, spectra in specs.items():
            avgSpecs[cls_name] = np.mean(spectra, axis=0)
        return avgSpecs

    def getSpectrumaAtXY(self, x: int, y: int) -> np.ndarray:
        if type(x) != int or type(y) != int:
            x, y = int(round(x)), int(round(y))
        x = np.clip(x, 0, self._cube.shape[1]-1)
        y = np.clip(y, 0, self._cube.shape[2]-1)
        return self._cube[:, x, y]

    def getNumberOfClasses(self) -> int:
        return len(self._classes)

    def getNumberOfFeatures(self) -> int:
        return self._cube.shape[0]

    def setWavenumbers(self, wavenumbers: np.ndarray) -> None:
        self._wavenumbers = wavenumbers

    def setClasses(self, classes: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        :param classes: Tuple: ClassName: Tuple[array of y-coordinates, array of x-coordinates]
        :return:
        """
        self._classes = classes

    def _setDefaultWavenumbers(self, cube: np.ndarray) -> None:
        """
        Convenience function to get default wavenumbers if None were set...
        :return:
        """
        self._wavenumbers = np.linspace(1115, 1671, cube.shape[0])

    def _getSpecArray(self, indices: List[Tuple[int, int]], preprocessed: bool) -> np.ndarray:
        """
        Gets the spectra at the indicated pixel coordinates
        :param indices: List of N (y, x) coordinate tuples
        :param preprocessed: whether or not to get the preprocessed or raw spectra
        :return: Shape (N x M) array of N spectra with M wavenumbers
        """
        if preprocessed:
            cube: np.ndarray = self.getCube()
        else:
            cube: np.ndarray = self._cube

        specArr: np.ndarray = np.zeros((len(indices), cube.shape[0]))
        for i in range(specArr.shape[0]):
            specArr[i, :] = cube[:, indices[i][0], indices[i][1]]
        return specArr


def splitUpArray(specArr: np.ndarray, numParts: int = 8) -> List[np.ndarray]:
    """
    Splits up the given array into a list of arrays.
    :param specArr: (NxM) shape array of N spectra with w wavelenghts.
    :param numParts: number of parts
    :param: List with numParts arrays.
    """
    arrList: List[np.ndarray] = []
    stepSize: int = specArr.shape[0] // numParts + 1
    for i in range(numParts):
        start = i*stepSize
        end = min([(i+1)*stepSize, specArr.shape[0]])
        arrList.append(specArr[start:end, :])
    return arrList


@dataclasses.dataclass
class PreprocessData:
    specArr: np.ndarray
    preprocQueue: List['Preprocessor']
    background: np.ndarray


def applyPreprocessing(preprocData: 'PreprocessData') -> np.ndarray:
    for preprocessor in preprocData.preprocQueue:
        if type(preprocessor) == Background:
            preprocessor: Background = cast(Background, preprocessor)
            preprocessor.setBackground(preprocData.background)
        specArr = preprocessor.applyToSpectra(preprocData.specArr)
    return specArr


def recombineSpecArrays(specArrs: List[np.ndarray]) -> np.ndarray:
    """
    Recombines the arrays in the list by vertically stacking them.
    """
    newArr: Union[None, np.ndarray] = None
    for arr in specArrs:
        if newArr is None:
            newArr = arr
        else:
            newArr = np.vstack((newArr, arr))
    return newArr
