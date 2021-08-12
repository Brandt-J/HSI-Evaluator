from abc import ABC, abstractmethod
from typing import List
import numpy as np

from SpectraProcessing.Preprocessing.processing import *


class Preprocessor(ABC):
    label: str = ''

    @abstractmethod
    def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
        """
        Takes (N, M) array of N spectra with M wavenumbers and applies the processing
        :param spectra:
        :return:
        """


class MeanCentering(Preprocessor):
    label = 'MeanCentering'

    def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
        return mean_center(spectra)


class Normalize(Preprocessor):
    label = 'Normalize'

    def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
        return normalizeIntensities(spectra)


class Detrend(Preprocessor):
    label = 'Detrend'

    def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
        return detrend(spectra)


class SNV(Preprocessor):
    label = 'Standard Normal Variate'

    def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
        return snv(spectra)


class Background(Preprocessor):
    label = 'Subtract Background'

    def __init__(self):
        super(Background, self).__init__()
        self._backgroundSpec: np.ndarray = None

    def setBackground(self, backgroundSpec: np.ndarray) -> None:
        if not np.array_equal(backgroundSpec, np.zeros_like(backgroundSpec)):
            backgroundSpec = (backgroundSpec - backgroundSpec.min()) / (backgroundSpec.max() - backgroundSpec.min())
        self._backgroundSpec = backgroundSpec

    def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
        if self._backgroundSpec is not None:
            spectra = spectra.copy()
            for i in range(spectra.shape[0]):
                curSpec = spectra[i, :].copy()
                curSpec = (curSpec - curSpec.min()) / (curSpec.max() - curSpec.min())
                curSpec -= self._backgroundSpec
                spectra[i, :] = curSpec
        return spectra


def getPreprocessors() -> List[Preprocessor]:
    return [Background(), Normalize(), MeanCentering(), SNV(), Detrend()]
