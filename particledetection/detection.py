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
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import numpy as np
from typing import *


def getParticleContours(binImg: np.ndarray) -> List[np.ndarray]:
    """
    Takes a grayscale image and returns a list of contours defining particles.
    :param binImg: The binary image, background = black, foreground = white, dtype: np.uint8
    :return: List of contours
    """
    binImg = assertUint8(binImg.copy())
    disttransform: np.ndarray = cv2.distanceTransform(binImg, cv2.DIST_L2, 3)

    sure_fg, sure_bg = getSureForegroundAndBackground(disttransform)
    markers: np.ndarray = getMarkersFromSureFgAndBg(sure_fg, sure_bg)
    markers = watershed(-disttransform, markers, mask=sure_bg, compactness=0.1, watershed_line=True)

    contours: List[np.ndarray] = getContoursFromMarkers(markers)
    return contours


def getContoursFromMarkers(markers: np.ndarray, minSize: int = 5, maxSize: int = np.inf) -> List[np.ndarray]:
    """
    Takes a labelled marker image and returns a list of contours.
    :param markers: Labelled image. 0 = Background, 1... = Particles
    :param minSize: Minimal pixel area for a particle to be considered
    :param maxSize: Maximal pixel area for a particle to be considered
    :return List of particle contours:
    """
    finalContours: List[np.ndarray] = []
    contours, hierarchy = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    tmpcontours = [contours[i] for i in range(len(contours)) if hierarchy[0, i, 3] < 0]
    for cnt in tmpcontours:
        contourArea: float = cv2.contourArea(cnt)
        if minSize <= contourArea <= maxSize:
            tmplabel = markers[cnt[0, 0, 1], cnt[0, 0, 0]]
            if tmplabel == 0:
                continue

            finalContours.append(cnt)
    return finalContours


def getMarkersFromSureFgAndBg(sure_fg: np.ndarray, sure_bg: np.ndarray) -> np.ndarray:
    """
    Takes Sure Foreground and Sure Background and creates a labelled marker image (using connected component search)
    :param sure_fg: Sure Foreground
    :param sure_bg: Sure Background
    :return markers: np.ndarray, 0 = background, 1... = particles
    """
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    return ndi.label(sure_fg)[0]


def getSureForegroundAndBackground(distanceTransform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes a distance transform image and creates a sure-foreground and a sure-background image for watershed segmentation.
    Seedpoints are set where local maxima in the distance transform are found.
    :param distanceTransform: distance transform image
    :return sure_fg: uint8 binary image, black = background, 1 = seedpoints
    """
    sure_fg = np.zeros_like(distanceTransform, dtype=np.uint8)
    localMax = np.uint8(peak_local_max(distanceTransform, 5, exclude_border=False, indices=False))
    localMax[distanceTransform == np.max(distanceTransform)] = 1  # add global maximum

    maxPoints = np.where(localMax == np.max(localMax))
    maxPoints = np.transpose(np.array(maxPoints))
    for point in maxPoints:
        sure_fg[point[0], point[1]] = 1

    sure_bg = cv2.dilate(distanceTransform, np.ones((5, 5)), iterations=1).astype(np.uint8)

    return sure_fg, sure_bg


def assertUint8(img: np.ndarray) -> np.ndarray:
    """
    Makes sure the passed in image is in Uint8 format. If not, it is rescaled to 0 - 255 and converted.
    """
    if img.dtype != np.uint8:
        img -= img.min()
        img /= img.max()
        img = img
        img = img.astype(np.uint8)

    return img
