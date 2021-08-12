import numpy as np
from typing import List, Dict, Tuple, TYPE_CHECKING, cast, Union
import random
import time

from logger import getLogger
from preprocessors import Background

if TYPE_CHECKING:
    from PyQt5 import QtCore
    from logging import Logger
    from preprocessors import Preprocessor


class SpectraObject:
    def __init__(self):
        self._wavenumbers: Union[None, np.ndarray] = None
        self._cube: Union[None, np.ndarray] = None
        self._preprocessedCube: Union[None, np.ndarray] = None
        self._classes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # classname, (y-coordinages, x-coordinates)
        self._logger: 'Logger' = getLogger("SpectraObject")

    def setCube(self, cube: np.ndarray) -> None:
        self._cube = cube
        if self._wavenumbers is None:
            self._setDefaultWavenumbers(cube)

    # TODO: REFACTOR TO DIRECTLY ACCEPT THE PROCESSING QUEUE
    # def applyPreprocessing(self, imgLimits: 'QtCore.QRectF') -> None:
    #     """
    #     Applies the specified preprocessing, if any queue was previously set using "setPreprocessors"
    #     :return:
    #     """
    #     if len(self._preprocessingQueue) > 0:
    #         t0 = time.time()
    #         specArr = self._cube2SpecArr(imgLimits)
    #
    #         for preprocessor in self._preprocessingQueue:
    #             if type(preprocessor) == Background:
    #                 preprocessor: Background = cast(Background, preprocessor)
    #                 preprocessor.setBackground(self.getMeanBackgroundSpec())
    #             specArr = preprocessor.applyToSpectra(specArr)
    #
    #         self._preprocessedCube = self._specArr2cube(specArr, imgLimits)
    #         print(f'preprocessing spectra took {round(time.time()-t0, 2)} seconds')
    #     else:
    #         self._preprocessedCube = self._cube.copy()

    def _specArr2cube(self, specArr: np.ndarray, imgLimits: 'QtCore.QRectF') -> np.ndarray:
        """
        Takes an (MxN) spec array and reformats into cube layout
        :param specArr: (MxN) array of M spectra with N wavenumbers
        :return: (NxXxY) cube array of X*Y spectra of N wavenumbers (M = X*Y)
        """
        cube = self._cube.copy()
        i = 0
        for y in range(self._cube.shape[1]):
            if imgLimits.top() <= y < imgLimits.bottom():
                for x in range(self._cube.shape[2]):
                    if imgLimits.left() <= x < imgLimits.right():
                        cube[:, y, x] = specArr[i, :]
                        i += 1

        return cube

    def _cube2SpecArr(self, imgLimits: 'QtCore.QRectF') -> np.ndarray:
        """
        Reformats the cube into an MxN spectra matrix of M spectra with N wavenumbers
        :return: (MxN) spec array of M spectra of N wavenumbers
        """
        specArr: List[np.ndarray] = []
        for y in range(self._cube.shape[1]):
            if imgLimits.top() <= y < imgLimits.bottom():
                for x in range(self._cube.shape[2]):
                    if imgLimits.left() <= x < imgLimits.right():
                        specArr.append(self._cube[:, y, x])
        specArr: np.ndarray = np.array(specArr)  # NxM array of N specs with M wavenumbers
        assert specArr.shape[1] == self._cube.shape[0]
        return specArr

    def getClassSpectra(self, maxSpecPerClas: int, preprocessed: bool = True) -> Dict[str, np.ndarray]:
        """
        Gets the a dictionary for the spectra per class.
        :param maxSpecPerClas: Indicated for
        :param preprocessed: Whether or not to return the preprocessed or, alternatively, the raw spectra.
        :return:
        """
        spectra: Dict[str, np.ndarray] = {}
        random.seed(42)
        for cls_name, cls_indices in self._classes.items():
            curIndices: List[Tuple[int, int]] = list(zip(cls_indices[0], cls_indices[1]))
            if len(curIndices) > 0:
                if len(curIndices) > maxSpecPerClas:
                    curIndices = random.sample(curIndices, maxSpecPerClas)
                spectra[cls_name] = self._getSpecArray(curIndices, preprocessed)

        return spectra

    def getCube(self) -> np.ndarray:
        cube: np.ndarray = self._preprocessedCube
        if cube is None:
            cube = self._cube
        return cube

    def getNotPreprocessedCube(self) -> np.ndarray:
        return self._cube

    def getWavenumbers(self) -> np.ndarray:
        assert self._wavenumbers is not None, 'Wavenumbers have not yet been set! Cannot return them!'
        return self._wavenumbers

    def getAverageSpectra(self) -> Dict[str, np.ndarray]:
        specs: Dict[str, np.ndarray] = self.getClassSpectra(maxSpecPerClas=np.inf)
        avgSpecs: Dict[str, np.ndarray] = {}
        for cls_name, spectra in specs.items():
            avgSpecs[cls_name] = np.mean(spectra, axis=0)
        return avgSpecs

    def getSpectrumaAtXY(self, x: int, y: int) -> np.ndarray:
        if type(x) != int or type(y) != int:
            x, y = int(round(x)), int(round(y))
        x = np.clip(x, 0, self._cube.shape[1]-1)
        y = np.clip(y, 0, self._cube.shape[2]-1)
        return self._cube[:, x, y]

    def getNumberOfClasses(self) -> int:
        return len(self._classes)

    def getNumberOfFeatures(self) -> int:
        return self._cube.shape[0]

    def setWavenumbers(self, wavenumbers: np.ndarray) -> None:
        self._wavenumbers = wavenumbers

    def setClasses(self, classes: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        :param classes: Tuple: ClassName: Tuple[array of y-coordinates, array of x-coordinates]
        :return:
        """
        self._classes = classes

    def _setDefaultWavenumbers(self, cube: np.ndarray) -> None:
        """
        Convenience function to get default wavenumbers if None were set...
        :return:
        """
        self._wavenumbers = np.linspace(1115, 1671, cube.shape[0])

    def _getSpecArray(self, indices: List[Tuple[int, int]], preprocessed: bool) -> np.ndarray:
        """
        Gets the spectra at the indicated pixel coordinates
        :param indices: List of N (y, x) coordinate tuples
        :param preprocessed: whether or not to get the preprocessed or raw spectra
        :return: Shape (N x M) array of N spectra with M wavenumbers
        """
        if preprocessed:
            cube: np.ndarray = self.getCube()
        else:
            cube: np.ndarray = self._cube

        specArr: np.ndarray = np.zeros((len(indices), cube.shape[0]))
        for i in range(specArr.shape[0]):
            specArr[i, :] = cube[:, indices[i][0], indices[i][1]]
        return specArr
