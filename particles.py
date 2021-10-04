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
    def __init__(self):
        self._particles: List['Particle'] = []

    def getParticlesFromImage(self, binaryImage: np.ndarray) -> None:
        """
        Takes a binary image and finds particles.
        """
        contours: List[np.ndarray] = getParticleContours(binaryImage)
        self._particles = [Particle(cnt) for cnt in contours]

    def getParticles(self) -> List['Particle']:
        """
        Returns the current list of particles.
        """
        return self._particles


@dataclass
class Particle:
    _contour: np.ndarray
    _threshold: float = 0.75
    _result: Union[None, Counter] = None

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