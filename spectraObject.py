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
from typing import List, Dict, Tuple, TYPE_CHECKING, Union

from logger import getLogger

if TYPE_CHECKING:
    from logging import Logger


class WavelengthsNotSetError(BaseException):
    pass


class SpectraObject:
    def __init__(self):
        self._wavelengths: Union[None, np.ndarray] = None
        self._cube: Union[None, np.ndarray] = None
        self._logger: 'Logger' = getLogger("SpectraObject")

    def __eq__(self, other):
        isEqual: bool = False
        if type(other) == type(self):
            dict1, dict2 = self.__dict__, other.__dict__
            if dict1.keys() == dict2.keys():
                allElementsTrue: bool = True
                for key in dict1.keys():
                    if type(dict1[key]) == np.ndarray:
                        if not np.array_equal(dict1[key], dict2[key]):
                            allElementsTrue = False
                            break
                    else:
                        if not dict1[key] == dict2[key]:
                            allElementsTrue = False
                            break
                isEqual = allElementsTrue

        return isEqual

    def setCube(self, cube: np.ndarray, wavelengths: np.ndarray = None) -> None:
        self._cube = cube
        if wavelengths is None:
            self._setDefaultWavelengths(cube)
        else:
            self._wavelengths = wavelengths

    def getCube(self) -> np.ndarray:
        """
        Returns the (LxMxN) spectra cube of MxN spectra with L wavelenghts.
        """
        return self._cube

    def getSpecArray(self) -> np.ndarray:
        """
        Returns an MxN spectra matrix of M spectra with N wavelengths of the cube spectra
        :return: (MxN) spec array of M spectra of N wavelengths
        """
        return self._cube2SpecArr()

    def getWavelengths(self) -> np.ndarray:
        if self._wavelengths is None:
            raise WavelengthsNotSetError()
        return self._wavelengths

    def getSpectrumaAtXY(self, x: int, y: int) -> np.ndarray:
        if type(x) != int or type(y) != int:
            x, y = int(round(x)), int(round(y))
        x = np.clip(x, 0, self._cube.shape[1]-1)
        y = np.clip(y, 0, self._cube.shape[2]-1)
        return self._cube[:, x, y]

    def getNumberOfFeatures(self) -> int:
        return self._cube.shape[0]

    def setWavelengths(self, wavelengths: np.ndarray) -> None:
        self._wavelengths = wavelengths

    def remapToWavelenghts(self, otherWavelenghts: np.ndarray) -> None:
        """
        Takes a wavelength axis and remaps the cube to fit the new wavelength axis.
        It also erases the preprocessed Cube and overwrites the cube wavelengths to the given ones.
        """
        if self._wavelengths is not None:
            if not np.array_equal(self._wavelengths, otherWavelenghts):
                newCube: np.ndarray = np.zeros((len(otherWavelenghts), self._cube.shape[1], self._cube.shape[2]))
                for i, wavelength in enumerate(otherWavelenghts):
                    closestIndex: int = int(np.argmin(np.abs(self._wavelengths - wavelength)))
                    newCube[i, :, :] = self._cube[closestIndex, :, :]

                self._cube = newCube
                self._wavelengths = otherWavelenghts
        else:
            raise WavelengthsNotSetError

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

    def _cube2SpecArr(self) -> np.ndarray:
        """
        Reformats the cube into an MxN spectra matrix of M spectra with N wavelengths
        :return: (MxN) spec array of M spectra of N wavelengths
        """
        i: int = 0
        specArr: List[np.ndarray] = []

        for y in range(self._cube.shape[1]):
            for x in range(self._cube.shape[2]):
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
        self._wavelengths = np.arange(cube.shape[0])

    def _getSpecArray(self, indices: List[Tuple[int, int]]) -> np.ndarray:
        """
        Gets the spectra at the indicated pixel coordinates
        :param indices: List of N (y, x) coordinate tuples
        :return: Shape (N x M) array of N spectra with M wavelengths
        """
        specArr: np.ndarray = np.zeros((len(indices), self._cube.shape[0]))
        for i in range(specArr.shape[0]):
            specArr[i, :] = self._cube[:, indices[i][0], indices[i][1]]
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
                assert self._spectra.shape[1] == specs.shape[1], f'incompatible spectra set shapes'

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
