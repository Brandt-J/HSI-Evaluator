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


class Sample:
    """Data Container for a sample view"""
    def __init__(self):
        self.filePath: str = ''  # Path to the spectra cube (.npy file)
        self.classes2Indices: Dict[str, Set[int]] = {}
        self.name: str = ''

    def setDefaultName(self) -> None:
        if len(self.filePath) > 0:
            _name: str = os.path.basename(self.filePath.split('.npy')[0])
        else:
            _name: str = 'NoNameDefined'
        self.name = _name

    def getFileHash(self) -> str:
        """Used for saving the files"""
        return getFilePathHash(self.filePath)

    def __eq__(self, other) -> bool:
        isEqual: bool = False
        if type(other) == Sample:
            other: Sample = cast(Sample, other)
            if other.name == self.name and other.filePath == self.filePath and other.classes2Indices == self.classes2Indices:
                isEqual = True

        return isEqual


class View:
    """Data container for an entire view, including multiple samples and a processing stack"""
    def __init__(self):
        self.title: str = ''
        self.samples: List['Sample'] = []
        self.processStack: List[str] = []


def getFilePathHash(fpath: str) -> str:
    """
    Function for hashing a filePath. Needs to be accessible as standalone as well for checking for saved samples
    before actually creating them..
    """
    return hashlib.sha1(fpath.encode()).hexdigest()
