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
import numpy as np
import random


def getRandomSpectraFromArray(specArr: np.ndarray, number: int, seed: int = 42) -> np.ndarray:
    """
    Returns a random selection of 'number' spectra from the given array.
    :param specArr: (NxM) array of N spectra with M wavenumbers.
    :param number: The number of spectra to return,
    :param seed: optional, the seed to use for the random number generator.
    :return: (number, M) array of spectra selected from the input array.
    """
    numSpecs: int = specArr.shape[0]
    if numSpecs > number:
        random.seed(seed)
        specArr = specArr.copy()
        randInd: np.ndarray = np.array(random.sample(list(np.arange(numSpecs)), number))
        specArr = specArr[randInd, :]
    return specArr
