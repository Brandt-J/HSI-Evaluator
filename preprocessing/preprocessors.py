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

# from abc import ABC, abstractmethod
# from typing import List
# import sys
# import os
# sys.path.append(os.getcwd())
# from preprocessing.processing import *
import numpy as np


class Preprocessor:
    label: str = ''

    def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
        """
        Takes (N, M) array of N spectra with M wavelengths and applies the processing
        :param spectra:
        :return:
        """
        raise NotImplementedError

#
# class MeanCentering(Preprocessor):
#     label = 'MeanCentering'
#
#     def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
#         return mean_center(spectra)
#
#
# class Normalize(Preprocessor):
#     label = 'Normalize'
#
#     def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
#         return normalizeIntensities(spectra)
#
#
# class Detrend(Preprocessor):
#     label = 'Detrend'
#
#     def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
#         return detrend(spectra)
#
#
# class SNV(Preprocessor):
#     label = 'Standard Normal Variate'
#
#     def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
#         return snv(spectra)
#
#
# class Smooth(Preprocessor):
#     label = 'Smooth'
#
#     def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
#         return deriv_smooth(spectra, derivative=0)
#
#
# class Derivative1(Preprocessor):
#     label = '1st Derivative (smooth)'
#
#     def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
#         return deriv_smooth(spectra, derivative=1)
#
#
# class Derivative2(Preprocessor):
#     label = '2nd Derivative (smooth)'
#
#     def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
#         return deriv_smooth(spectra, derivative=2)
#
#
class Background(Preprocessor):
    label = 'Subtract Background'

    def __init__(self):
        super(Background, self).__init__()
        self._backgroundSpec: np.ndarray = None

    def setBackground(self, backgroundSpec: np.ndarray) -> None:
        if not np.array_equal(backgroundSpec, np.zeros_like(backgroundSpec)):
            backgroundSpec = (backgroundSpec - backgroundSpec.min()) / (backgroundSpec.max() - backgroundSpec.min())
        self._backgroundSpec = backgroundSpec

    def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
        if self._backgroundSpec is not None:
            spectra = spectra.copy()
            for i in range(spectra.shape[0]):
                curSpec = spectra[i, :].copy()
                curSpec = (curSpec - curSpec.min()) / (curSpec.max() - curSpec.min())
                curSpec -= self._backgroundSpec
                spectra[i, :] = curSpec
        return spectra
#
#
# def getPreprocessors() -> List[Preprocessor]:
#     return [Background(), Normalize(), MeanCentering(), SNV(), Detrend(), Derivative1(), Derivative2()]
