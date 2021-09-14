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

import os
import sys
from typing import Tuple
import numpy as np

os.environ['PATH'] = r'C:\imec_15\HSI Snapscan\bin' + os.pathsep + os.environ['PATH']
sys.path.append(r'C:\imec_15\HSI Snapscan\python')
try:
    import hsi_snapscan as HSI
    snapscanEnabled: bool = True
except ImportError:
    print("Failed to load HSI Snapscan API")
    snapscanEnabled: bool = False


def loadCube(fname: str, errorPixelThreshold: float = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a numpy cube and sets "defect" pixels to zero. They usually contain very large values (1e34) and thereby hamper
    further evaluation.
    :param fname: Path to .npy file
    :param errorPixelThreshold: The threshold used to detet erroneous pixels.
    :return: Tuple[(NxM) array of N spectra with M wavelenghts, array of M wavelengths]
    """
    if fname.endswith(".npy"):
        cube: np.ndarray = np.load(fname)
        wavelengths: np.ndarray = np.arange(cube.shape[0])
    elif fname.endswith(".hdr"):
        assert snapscanEnabled, f'Cannot load {fname}, snapscan API could not be loaded. Check the "loadCube.npy" for details.'
        cube, wavelengths = loadHDRCube(fname)
    else:
        raise TypeError("The specified file is not supported (only .npy or .hdr)")

    cube[cube > errorPixelThreshold] = 0
    return cube, wavelengths


def loadHDRCube(fname: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a .hdr file, converts to numpy array and returns it together with the wavelengths.
    :param fname: Absolute path to .hdr file
    :return: Tuple[(NxM) array of N spectra with M wavelenghts, array of M wavelengths]
    """
    img = HSI.LoadCube(fname)
    format = img.format.as_dict()
    bands_nm: np.ndarray = format["bands_nm"]
    cube:np.ndarray = HSI.CubeAsArray(img)
    return cube, bands_nm
