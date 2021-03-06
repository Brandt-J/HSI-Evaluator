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

import numba
import numpy as np

from spectraObject import SpectraObject, getSpectraFromIndices
from legacyConvert import currentSampleVersion, currentViewVersion
from particles import ParticleHandler

if TYPE_CHECKING:
    from classification.classifiers import BatchClassificationResult
    from particles import Particle


class Sample:
    """Data Container for a sample view"""
    def __init__(self):
        self.version: int = currentSampleVersion  # tracker for updating versions
        self.name: str = 'Empty Sample'  # Sample Name
        self.filePath: str = ''  # Path to the spectra cube (.npy file)
        self.classes2Indices: Dict[str, Set[int]] = {}  # Stores pixel indices of selected classes
        self.specObj: SpectraObject = SpectraObject()  # Spectra Object
        self.batchResult: Union[None, 'BatchClassificationResult'] = None
        self.particleHandler: ParticleHandler = ParticleHandler()
        self.viewCoordinates: Optional[Tuple[float, float]] = None  # x, y coordinates of the sampleview
        self.flippedHorizontally: bool = False
        self.flippedVertically: bool = False

    def setDefaultName(self) -> None:
        if len(self.filePath) > 0:
            _name: str = os.path.basename(self.filePath.split('.npy')[0])
        else:
            _name: str = 'NoNameDefined'
        self.name = _name

    def getFileHash(self) -> str:
        """Used for saving the files"""
        return getFilePathHash(self.filePath)

    def _getBackroundIndices(self) -> Set[int]:
        """
        Returns the indices of background pixels.
        """
        indices: Set[int] = set()
        for cls_name in self.classes2Indices:
            if cls_name.lower() == 'background':
                indices = self.classes2Indices[cls_name]
                break
        return indices

    def flipHorizontally(self) -> None:
        cube: np.ndarray = np.flip(self.specObj.getCube(), axis=2)
        self.specObj.setCube(cube)
        self.particleHandler.flipParticlesHorizontally(cube.shape)
        for cls, indices in self.classes2Indices.items():
            self.classes2Indices[cls] = flipIndicesHorizontally(indices, cube.shape)

        self.flippedHorizontally = not self.flippedHorizontally

    def flipVertically(self) -> None:
        cube: np.ndarray = np.flip(self.specObj.getCube(), axis=1)
        self.specObj.setCube(cube)
        self.particleHandler.flipParticlesVertically(cube.shape)
        for cls, indices in self.classes2Indices.items():
            self.classes2Indices[cls] = flipIndicesVertically(indices, cube.shape)

        self.flippedVertically = not self.flippedVertically

    def getAllParticles(self) -> List['Particle']:
        """
        Returns a list of particles found in the sample.
        """
        return self.particleHandler.getParticles()

    def __eq__(self, other) -> bool:
        isEqual: bool = False
        if type(other) == Sample:
            other: Sample = cast(Sample, other)
            if other.name == self.name and other.filePath == self.filePath and other.classes2Indices == self.classes2Indices:
                isEqual = True

        return isEqual

    def getLabelledSpectra(self) -> Dict[str, np.ndarray]:
        """
        Gets the labelled Spectra, in form of a dictionary.
        :return: Dictionary [className, NxM array of N spectra with M wavelengths]
        """
        spectra: Dict[str, np.ndarray] = {}
        for name, indices in self.classes2Indices.items():
            spectra[name] = getSpectraFromIndices(np.array(list(indices)), self.specObj.getCube())
        return spectra

    def getAveragedBackgroundSpectrum(self) -> np.ndarray:
        cube: np.ndarray = self.specObj.getCube()
        background: np.ndarray = np.zeros(cube.shape[0])
        backgrounIndices: Set[int] = self._getBackroundIndices()
        if len(backgrounIndices) > 0:
            indices: np.ndarray = np.array(list(backgrounIndices))
            background = np.mean(getSpectraFromIndices(indices, cube), axis=0)

        return background

    # def getPreprocessedSpecCube(self) -> np.ndarray:
    #     """
    #     Returns a copy of the preprocessed spectra cube.
    #     """
    #     return self.specObj.getPreprocessedCubeIfPossible().copy()
    #
    def getSpecCube(self) -> np.ndarray:
        return self.specObj.getCube()

    def setBatchResults(self, batchResult: 'BatchClassificationResult') -> None:
        self.batchResult = batchResult

    def getBatchResults(self, cutOff: float) -> np.ndarray:
        assert self.batchResult is not None
        return self.batchResult.getResults(cutOff)

    def resetParticleResults(self) -> None:
        """
        Resets the results of all particles. Called before initiating a new run of classification.
        """
        self.particleHandler.resetParticleResults()

    def getParticleHandler(self) -> 'ParticleHandler':
        """
        Returns a reference to the particle handler.
        """
        return self.particleHandler


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


def flipIndicesVertically(indices: Set[int], cubeShape: np.ndarray) -> Set[int]:
    """
    Flips pixel indices vertically, according to the given spectra cube shape.
    """
    width: int = cubeShape[2]
    height: int = cubeShape[1]
    newIndices = np.zeros(len(indices), dtype=np.int64)
    for i, ind in enumerate(indices):
        y_orig = ind // width
        y_new = height - y_orig - 1
        x = ind % width
        newIndices[i] = y_new * cubeShape[2] + x
    return set(newIndices)


def flipIndicesHorizontally(indices: Set[int], cubeShape: np.ndarray) -> Set[int]:
    """
    Flips pixel indices horizontally, according to the given spectra cube shape.
    """
    width: int = cubeShape[2]
    newIndices = np.zeros(len(indices), dtype=np.int64)
    for i, ind in enumerate(indices):
        y = ind // width
        x_orig = ind % width
        x_new = width - x_orig - 1
        newIndices[i] = y * width + x_new
    return set(newIndices)
