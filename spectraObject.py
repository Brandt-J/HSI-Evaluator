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
import numpy as np
from typing import List, Dict, Tuple, TYPE_CHECKING, Union, cast, Set
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
        self._wavelengths: Union[None, np.ndarray] = None
        self._cube: Union[None, np.ndarray] = None
        self._preprocQueue: List['Preprocessor'] = []
        self._background: Union[None, np.ndarray] = None
        self._backgroundIndices: Set[int] = set()
        self._preprocessedCube: Union[None, np.ndarray] = None
        self._classes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # classname, (y-coordinages, x-coordinates)
        self._logger: 'Logger' = getLogger("SpectraObject")

    def setCube(self, cube: np.ndarray, wavelengths: np.ndarray = None) -> None:
        self._cube = cube
        if wavelengths is None:
            self._setDefaultWavelengths(cube)
        else:
            self._wavelengths = wavelengths

    def preparePreprocessing(self, preprocessingQueue: List['Preprocessor'], background: np.ndarray, backgroundIndices: Set[int] = set()):
        """
        Sets the preprocessing parameters
        :param preprocessingQueue: List of Preprocessors
        :param background: np.ndarray of background spectrum
        :param backgroundIndices: Set of indices of pixels of background class. These will be ignored during processing, if required.
        """
        self._preprocQueue = preprocessingQueue
        self._background = background
        self._backgroundIndices = backgroundIndices

    def applyPreprocessing(self, ignoreBackground: bool = False) -> None:
        """
        Applies the specified preprocessing.

        :return:
        """
        if len(self._preprocQueue) > 0:
            specArr = self._cube2SpecArr(ignoreBackground)
            t0 = time.time()
            if len(specArr) < 1000:
                specArr = self._preprocessSpectaSingleProcess(specArr)
            else:
                specArr = self._preprocessSpectraMultiProcessing(specArr)

            self._preprocessedCube = self._specArr2cube(specArr, ignoreBackground)
            print(f'preprocessing spectra took {round(time.time()-t0, 2)} seconds')
        else:
            self._logger.info("Received empty preprocessingQueue, just returning the original cube.")
            self._preprocessedCube = self._cube.copy()

        self._resetPreprocessing()

    def _preprocessSpectraMultiProcessing(self, specArr: np.ndarray) -> np.ndarray:
        """
        Preprocesses the given spectra array using a Process Pool Executor.
        :param specArr: (NxM) array of N spectra with M wavelengths
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
        :param specArr: (NxM) array of N spectra with M wavelengths
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
        self._background = None

    def _specArr2cube(self, specArr: np.ndarray, ignoreBackground: bool) -> np.ndarray:
        """
        Takes an (MxN) spec array and reformats into cube layout
        :param specArr: (MxN) array of M spectra with N wavelengths
        :param ignoreBackground: Whether or not the background pixels where ignored
        :return: (NxXxY) cube array of X*Y spectra of N wavelengths (M = X*Y)
        """
        cube = self._cube.copy()
        i: int = 0  # counter for cube index
        j: int = 0  # counter for spec Array
        for y in range(self._cube.shape[1]):
            for x in range(self._cube.shape[2]):
                if not ignoreBackground or (ignoreBackground and i not in self._backgroundIndices):
                    cube[:, y, x] = specArr[j, :]
                    j += 1

                i += 1

        return cube

    def _cube2SpecArr(self, ignoreBackground: bool) -> np.ndarray:
        """
        Reformats the cube into an MxN spectra matrix of M spectra with N wavelengths
        :param ignoreBackground: If True, background spectra will be skipped
        :return: (MxN) spec array of M spectra of N wavelengths
        """
        i: int = 0
        specArr: List[np.ndarray] = []
        for y in range(self._cube.shape[1]):
            for x in range(self._cube.shape[2]):
                if not ignoreBackground:
                    specArr.append(self._cube[:, y, x])
                elif i not in self._backgroundIndices:
                    specArr.append(self._cube[:, y, x])
                i += 1

        specArr: np.ndarray = np.array(specArr)  # NxM array of N specs with M wavelengths
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

    def getWavelengths(self) -> np.ndarray:
        assert self._wavelengths is not None, 'Wavenumbers have not yet been set! Cannot return them!'
        return self._wavelengths

    def getBackgroundIndices(self) -> Set[int]:
        """Returns pixel indices of background pixels"""
        return self._backgroundIndices

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

    def setWavelengths(self, wavelengths: np.ndarray) -> None:
        self._wavelengths = wavelengths

    def setClasses(self, classes: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        :param classes: Tuple: ClassName: Tuple[array of y-coordinates, array of x-coordinates]
        :return:
        """
        self._classes = classes

    def _setDefaultWavelengths(self, cube: np.ndarray) -> None:
        """
        Convenience function to get default wavelengths if None were set...
        :return:
        """
        self._wavelengths = np.linspace(1115, 1671, cube.shape[0])

    def _getSpecArray(self, indices: List[Tuple[int, int]], preprocessed: bool) -> np.ndarray:
        """
        Gets the spectra at the indicated pixel coordinates
        :param indices: List of N (y, x) coordinate tuples
        :param preprocessed: whether or not to get the preprocessed or raw spectra
        :return: Shape (N x M) array of N spectra with M wavelengths
        """
        if preprocessed:
            cube: np.ndarray = self.getCube()
        else:
            cube: np.ndarray = self._cube

        specArr: np.ndarray = np.zeros((len(indices), cube.shape[0]))
        for i in range(specArr.shape[0]):
            specArr[i, :] = cube[:, indices[i][0], indices[i][1]]
        return specArr


class SpectraCollection:
    """
    Container for keeping spectra from multiple samples.
    """
    def __init__(self):
        self._labels: Union[None, np.ndarray] = None  # Array storing N class names
        self._spectra: Union[None, np.ndarray] = None  # (NxM) Array storing N spectra of M wavelenghts
        self._sampleNames: Union[None, np.ndarray] = None  # Array storing N sample names

    def getXY(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple of Spectra (NxM) Array and class labels (N) array.
        """
        return self._spectra, self._labels

    def getDictionary(self) -> Dict[str, np.ndarray]:
        """
        Converts the collection into a dictionary with class-names as keys and corresponding spectra arrays as values.
        """
        specDict: Dict[str, np.ndarray] = {}
        uniqueClasses: np.ndarray = np.unique(self._labels)
        for cls in uniqueClasses:
            ind: np.ndarray = np.where(self._labels == cls)[0]
            specDict[cls] = self._spectra[ind, :]
        return specDict

    def getSampleDictionary(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Creates a dictionary sample wise. Keys: Sample Names. Values: Sample-Dict with keys: class-names and
        values: spec array
        """
        sampleDict: Dict[str, Dict[str, np.ndarray]] = {}
        uniqueSamples: np.ndarray = np.unique(self._sampleNames)
        uniqueClasses: np.ndarray = np.unique(self._labels)
        for sample in uniqueSamples:
            sampleInd: np.ndarray = np.where(self._sampleNames == sample)[0]
            sampleSpecs: np.ndarray = self._spectra[sampleInd, :]
            sampleLbl: np.ndarray = self._labels[sampleInd]

            specDict: Dict[str, np.ndarray] = {}
            for cls in uniqueClasses:
                clsInd: np.ndarray = np.where(sampleLbl == cls)[0]
                if len(clsInd) > 0:
                    specDict[cls] = sampleSpecs[clsInd, :]

            sampleDict[sample] = specDict

        return sampleDict

    def getSampleNames(self) -> np.ndarray:
        """
        Returns the sample names of each spectrum.
        """
        return self._sampleNames

    def hasSpectra(self) -> bool:
        """
        Returns, whether the collection actually contains any data.
        """
        return self._spectra is not None

    def addSpectraDict(self, specsToAdd: Dict[str, np.ndarray], sampleName: str) -> None:
        """
        Adds the given spectra dictionary of the given sample to the collection.
        :param specsToAdd: (key: className, value: NxM array of N spectra with M wavelenghts)
        :param sampleName: Name of corresponding sample
        """
        for cls, specs in specsToAdd.items():
            numSpecs: int = specs.shape[0]
            if self._spectra is None:
                self._spectra = specs
                self._labels = np.array([cls]*numSpecs)
                self._sampleNames = np.array([sampleName]*numSpecs)
            else:
                self._spectra = np.vstack((self._spectra, specs))
                self._labels = np.append(self._labels, [cls]*numSpecs)
                self._sampleNames = np.append(self._sampleNames, [sampleName]*numSpecs)


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
