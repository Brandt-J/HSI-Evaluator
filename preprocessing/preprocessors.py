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

from typing import *
import numpy as np
import dataclasses

from logger import getLogger
if TYPE_CHECKING:
    from logging import Logger

preprocLogger: 'Logger' = getLogger("Preprocessing")


class Preprocessor:
    label: str = ''
    backgroundSpec: Union[None, np.ndarray] = None

    def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
        """
        Takes (N, M) array of N spectra with M wavelengths and applies the processing
        :param spectra:
        :return:
        """
        raise NotImplementedError

    def setBackground(self, background: np.ndarray) -> np.ndarray:
        """
        Sets a background spectrum. Can be overridden, if required.
        """
        raise NotImplementedError


@dataclasses.dataclass
class PreprocessData:
    specArr: np.ndarray
    preprocQueue: List['Preprocessor']
    background: np.ndarray


def preprocessSpectra(specArr: np.ndarray, preprocessors: List['Preprocessor'], background: np.ndarray) -> np.ndarray:
    """
    Applies the specified preprocessing to the spectra array.
    :param specArr: (MxN) shape array of M spectra with N wavelenghts.
    :param preprocessors: List of preprocessors to apply
    :param background: Averaged background spectrum.
    :return: preprocessed spectra array
    """
    specArr = specArr.copy()  # We don't want to override any original data...
    if len(preprocessors) == 0:
        preprocessedSpecs: np.ndarray = specArr
    else:
        preprocData: PreprocessData = PreprocessData(specArr, preprocessors, background)
        preprocessedSpecs: np.ndarray = _applyPreprocessing(preprocData)

    return preprocessedSpecs


def _applyPreprocessing(preprocData: 'PreprocessData') -> np.ndarray:
    """
    Applies preprocessing, as defined by the PreprocessData obect.
    :param preprocData: The PreprocessingData to use
    :return: (NxM) array of N preprocessed spectra with M wavelenghts.
    """
    for preprocessor in preprocData.preprocQueue:
        if preprocessor.label == "Background":
            preprocessor.setBackground(preprocData.background)
        preprocData.specArr = preprocessor.applyToSpectra(preprocData.specArr)
    return preprocData.specArr


def _recombineSpecArrays(specArrs: List[np.ndarray]) -> np.ndarray:
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


# class Background(Preprocessor):
#     label = 'Subtract Background'
#
#     def __init__(self):
#         super(Background, self).__init__()
#         self._backgroundSpec: np.ndarray = None
#
#     def setBackground(self, backgroundSpec: np.ndarray) -> None:
#         if not np.array_equal(backgroundSpec, np.zeros_like(backgroundSpec)):
#             backgroundSpec = (backgroundSpec - backgroundSpec.min()) / (backgroundSpec.max() - backgroundSpec.min())
#         self._backgroundSpec = backgroundSpec
#
#     def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
#         if self._backgroundSpec is not None:
#             spectra = spectra.copy()
#             for i in range(spectra.shape[0]):
#                 curSpec = spectra[i, :].copy()
#                 curSpec = (curSpec - curSpec.min()) / (curSpec.max() - curSpec.min())
#                 curSpec -= self._backgroundSpec
#                 spectra[i, :] = curSpec
#         return spectra
