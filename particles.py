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
from dataclasses import dataclass
from collections import Counter

from particledetection.detection import getParticleContours


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

    def getAssigmentOfParticleOfID(self, id: int) -> str:
        """
        Returns the assignment of the partice specified by the id.
        :param id: The particle's id
        :return: assignment
        """
        return self._particles[id].getAssignment()

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
    _threshold: float = 0.5
    _result: Union[None, Counter] = None

    def getID(self) -> int:
        return self.__id

    def getAssignment(self) -> str:
        """
        Returns the assignment string according the currently set threshold.
        """
        assignment: str = "unknown"
        if self._result is not None:
            numTotal: int = sum(self._result.values())
            mostFreqClass, highestCount = self._result.most_common(8)[0]
            if highestCount / numTotal >= self._threshold:
                assignment = mostFreqClass

        return assignment

    def setThreshold(self, newThreshold: float) -> None:
        """
        Sets the new threshold for determining the assignment according the result.
        """
        self._threshold = newThreshold

    def getContour(self) -> np.ndarray:
        """
        Gets the particle's contour.
        """
        return self._contour

    def getSpectra(self, cube: np.ndarray) -> np.ndarray:
        """
        Takes the spectrum cube and extracts the spectra according the particle's contour.
        """
        specs: List[np.ndarray] = []
        mask: np.ndarray = np.zeros(cube.shape[1:])
        cv2.drawContours(mask, [self._contour], -1, 1, thickness=-1)
        indY, indX = np.where(mask == 1)
        for y, x in zip(indY, indX):
            specs.append(cube[:, y, x])
        return np.array(specs)

    def setResultFromAssignments(self, assignments: List[str]) -> None:
        """
        Sets the results counter with the given assignments.
        """
        self._result = Counter(assignments)

    def resetResult(self) -> None:
        """
        Resets the result.
        """
        self._result = None