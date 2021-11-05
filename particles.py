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
import cv2
import numpy as np
from typing import *
from collections import Counter
from dataclasses import dataclass

from particledetection.detection import getParticleContours

if TYPE_CHECKING:
    from classification.classifiers import BatchClassificationResult
    from gui.classUI import ClassInterpretationParams
    

class ParticleHandler:
    __particleID: int = -1

    @classmethod
    def getNewParticleID(cls) -> int:
        """
        Returns a unique particle id.
        """
        cls.__particleID += 1
        return cls.__particleID

    def __init__(self):
        self._particles: Dict[int, 'Particle'] = {}  # key: unique id, value: Particle object

    def getParticlesFromImage(self, binaryImage: np.ndarray) -> None:
        """
        Takes a binary image and finds particles.
        """
        contours: List[np.ndarray] = getParticleContours(binaryImage)
        self._particles = {}
        for cnt in contours:
            newID: int = ParticleHandler.getNewParticleID()
            self._particles[newID] = Particle(newID, cnt)

    def getParticles(self) -> List['Particle']:
        """
        Returns the current list of particles.
        """
        return list(self._particles.values())

    def getAssigmentOfParticleOfID(self, id: int, interpretationParams: 'ClassInterpretationParams') -> str:
        """
        Returns the assignment of the partice specified by the id.
        :param id: The particle's id
        :param interpretationParams: The parameters for interpreting the spectra results.
        :return: assignment
        """
        return self._particles[id].getAssignment(interpretationParams)

    def resetParticleResults(self) -> None:
        """
        Resets all particle results
        """
        for particle in self._particles.values():
            particle.resetResult()


@dataclass
class Particle:
    __id: int
    _contour: np.ndarray
    _result: Union[None, 'BatchClassificationResult'] = None

    def getID(self) -> int:
        return self.__id

    def getAssignment(self, params: 'ClassInterpretationParams') -> str:
        """
        Returns the assignment string according the currently set threshold.
        :param params: Parameters for correct result interpretation.
        """
        assignment: str = "unknown"
        if self._result is not None:
            classNames: np.ndarray = self._result.getResults(cutoff=params.specConfThreshold)
            if params.ignoreUnkowns and not np.all(classNames == "unknown"):
                classNames = classNames[classNames != "unknown"]

            counter: Counter = Counter(classNames)
            numTotal: int = sum(counter.values())
            numClasses: int = len(counter)
            mostFreqClass, highestCount = counter.most_common(numClasses)[0]

            if highestCount / numTotal >= params.partConfThreshold:
                assignment = mostFreqClass

        return assignment

    def getContour(self) -> np.ndarray:
        """
        Gets the particle's contour.
        """
        return self._contour

    def getSpectraArray(self, cube: np.ndarray, binning: int = 1) -> np.ndarray:
        """
        Takes the spectrum cube and extracts the spectra according the particle's contour.
        :param cube: 3D spec cube
        :param binning: number of spectra to average. If binning=1, no binning is performed.
        :return: MxN array of M specs with N wavelenghts
        """
        specs: List[np.ndarray] = []
        mask: np.ndarray = np.zeros(cube.shape[1:])
        cv2.drawContours(mask, [self._contour], -1, 1, thickness=-1)
        indY, indX = np.where(mask == 1)
        for y, x in zip(indY, indX):
            specs.append(cube[:, y, x])

        specs: np.ndarray = np.array(specs)

        if binning > 1:
            numSpecs: int = specs.shape[0]
            if binning >= numSpecs:
                binnedSpecs: np.ndarray = np.mean(specs, axis=0)[np.newaxis, :]
            else:
                binnedSpecs: List[np.ndarray] = []
                ind: np.ndarray = np.arange(binning)
                for i in range(numSpecs // binning):
                    if i > 0:
                        ind += binning
                    binnedSpecs.append(np.mean(specs[ind, :], axis=0))
                lastIndex: int = ind[-1]
                if lastIndex < numSpecs - 1:  # one or more spectra are left
                    if lastIndex == numSpecs-2:  # exactly one spec is left
                        binnedSpecs.append(specs[-1, :])
                    else:  # more specs are left
                        binnedSpecs.append(np.mean(specs[lastIndex:, :], axis=0))
                binnedSpecs: np.ndarray = np.array(binnedSpecs)

            specs = binnedSpecs

        return specs

    def setBatchResult(self, batchRes: 'BatchClassificationResult') -> None:
        """
        Sets the results counter with the given assignments.
        """
        self._result = batchRes

    def resetResult(self) -> None:
        """
        Resets the result.
        """
        self._result = None
