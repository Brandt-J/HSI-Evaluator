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
import hashlib
import os
import numpy as np

from spectraObject import SpectraObject, getSpectraFromIndices
from legacyConvert import currentSampleVersion, currentViewVersion


class Sample:
    """Data Container for a sample view"""
    def __init__(self):
        self.version: int = currentSampleVersion  # tracker for updating versions
        self.name: str = ''  # Sample Name
        self.filePath: str = ''  # Path to the spectra cube (.npy file)
        self.classes2Indices: Dict[str, Set[int]] = {}  # Stores pixel indices of selected classes
        self.specObj: SpectraObject = SpectraObject()  # Spectra Object
        self.classOverlay: Union[None, np.ndarray] = None  # RGBA Overlay used for displaying predicted class labels

    def setDefaultName(self) -> None:
        if len(self.filePath) > 0:
            _name: str = os.path.basename(self.filePath.split('.npy')[0])
        else:
            _name: str = 'NoNameDefined'
        self.name = _name

    def getFileHash(self) -> str:
        """Used for saving the files"""
        return getFilePathHash(self.filePath)

    def getBackroundIndices(self) -> Set[int]:
        """
        Returns the indices of background pixels.
        """
        indices: Set[int] = set()
        for cls_name in self.classes2Indices:
            if cls_name.lower() == 'background':
                indices = self.classes2Indices[cls_name]
                break
        return indices

    def __eq__(self, other) -> bool:
        isEqual: bool = False
        if type(other) == Sample:
            other: Sample = cast(Sample, other)
            if other.name == self.name and other.filePath == self.filePath and other.classes2Indices == self.classes2Indices:
                isEqual = True

        return isEqual

    def getLabelledPreprocessedSpectra(self) -> Dict[str, np.ndarray]:
        """
        Gets the labelled Spectra, in form of a dictionary.
        :return: Dictionary [className, NxM array of N spectra with M wavelengths]
        """
        spectra: Dict[str, np.ndarray] = {}
        for name, indices in self.classes2Indices.items():
            spectra[name] = getSpectraFromIndices(np.array(list(indices)), self.specObj.getPreprocessedCubeIfPossible())
        return spectra

    def setClassOverlay(self, classImage: np.ndarray) -> None:
        self.classOverlay = classImage


class View:
    """Data container for an entire view, including multiple samples and a processing stack"""
    def __init__(self):
        self.version: int = currentViewVersion  # counter for tracking older versions
        self.title: str = ''
        self.samples: List['Sample'] = []
        self.processingGraph: List[dict] = []


def getFilePathHash(fpath: str) -> str:
    """
    Function for hashing a filePath. Needs to be accessible as standalone as well for checking for saved samples
    before actually creating them..
    """
    return hashlib.sha1(fpath.encode()).hexdigest()
