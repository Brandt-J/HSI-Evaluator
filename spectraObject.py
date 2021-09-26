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
import numba
import numpy as np
from typing import List, Dict, Tuple, TYPE_CHECKING, Union, Set
import random

from preprocessing.preprocessors import preprocessSpectra
from logger import getLogger

if TYPE_CHECKING:
    from multiprocessing import Queue
    from logging import Logger
    from preprocessing.preprocessors import Preprocessor


class SpectraObject:
    def __init__(self):
        self._wavelengths: Union[None, np.ndarray] = None
        self._cube: Union[None, np.ndarray] = None
        self._preprocessedCube: Union[None, np.ndarray] = None
        self._logger: 'Logger' = getLogger("SpectraObject")
        self._backgroundIndices: Set[int] = set()

    def setCube(self, cube: np.ndarray, wavelengths: np.ndarray = None) -> None:
        self._cube = cube
        if wavelengths is None:
            self._setDefaultWavelengths(cube)
        else:
            self._wavelengths = wavelengths

    def doPreprocessing(self, preprocessors: List['Preprocessor'], backgroundIndices: Set[int]) -> None:
        """
        Takes a list of preprocessors, applies it to the cube and saves the result as preprocessedCube
        :param preprocessors: List of preprocessors
        :param backgroundIndices: Set of indices of background spectra
        """
        if len(backgroundIndices) > 0:
            backgroundSpec: np.ndarray = getSpectraFromIndices(np.array(list(backgroundIndices)), self._cube)
        else:
            backgroundSpec = np.zeros(self._cube.shape[0])
        preprocessedSpectra: np.ndarray = preprocessSpectra(self._cube2SpecArr(), preprocessors, backgroundSpec)
        self._preprocessedCube = self._specArr2cube(preprocessedSpectra)

    def getPreprocessedCubeIfPossible(self) -> np.ndarray:
        """
        Returns the spec cube. If any preprocessing was done, the preprocessed group is returned.
        """
        cube: np.ndarray = self._preprocessedCube
        if cube is None:
            cube = self._cube
        return cube

    def getNotPreprocessedCube(self) -> np.ndarray:
        return self._cube

    def getWavelengths(self) -> np.ndarray:
        assert self._wavelengths is not None, 'Wavenumbers have not yet been set! Cannot return them!'
        return self._wavelengths

    def getSpectrumaAtXY(self, x: int, y: int) -> np.ndarray:
        if type(x) != int or type(y) != int:
            x, y = int(round(x)), int(round(y))
        x = np.clip(x, 0, self._cube.shape[1]-1)
        y = np.clip(y, 0, self._cube.shape[2]-1)
        return self._cube[:, x, y]

    def getBackgroundIndices(self) -> Set[int]:
        return self._backgroundIndices

    def getNumberOfFeatures(self) -> int:
        return self._cube.shape[0]

    def setWavelengths(self, wavelengths: np.ndarray) -> None:
        self._wavelengths = wavelengths

    def _specArr2cube(self, specArr: np.ndarray, ignoreBackground: bool = False) -> np.ndarray:
        """
        Takes an (MxN) spec array and reformats into cube layout
        :param specArr: (MxN) array of M spectra with N wavelengths
        :param ignoreBackground: Whether or not the background pixels where ignored
        :return: (NxXxY) cube array of X*Y spectra of N wavelengths (M = X*Y)
        """
        if specArr.shape[1] == self._cube.shape[0]:
            cube: np.ndarray = self._cube.copy()
        else:
            cube = np.zeros((specArr.shape[1], self._cube.shape[1], self._cube.shape[2]))

        i: int = 0  # counter for cube index
        j: int = 0  # counter for spec Array
        for y in range(cube.shape[1]):
            for x in range(cube.shape[2]):
                if not ignoreBackground or (ignoreBackground and i not in self._backgroundIndices):
                    cube[:, y, x] = specArr[j, :]
                    j += 1
                i += 1

        return cube

    def _cube2SpecArr(self, ignoreBackground: bool = False) -> np.ndarray:
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
            cube: np.ndarray = self.getPreprocessedCubeIfPossible()
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


@numba.njit()
def getSpectraFromIndices(indices: np.ndarray, cube: np.ndarray) -> np.ndarray:
    """
    Retrieves the indices from the cube and returns an NxM array
    :param indices: length N array of flattened indices
    :param cube: XxYxZ spectral cube
    :return: (NxX) array of spectra
    """
    spectra: np.ndarray = np.zeros((len(indices), cube.shape[0]))
    for i, ind in enumerate(indices):
        y = ind // cube.shape[2]
        x = ind % cube.shape[2]
        spectra[i, :] = cube[:, y, x]
    return spectra
