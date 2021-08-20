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


def loadNumpyCube(fname: str, errorPixelThreshold: float = 1000) -> np.ndarray:
    """
    Loads a numpy cube and sets "defect" pixels to zero. They usually contain very large values (1e34) and thereby hamper
    further evaluation.
    :param fname: Path to .npy file
    :param errorPixelThreshold: The threshold used to detet erroneous pixels.
    """
    cube: np.ndarray = np.load(fname)
    cube[cube > errorPixelThreshold] = 0
    return cube
