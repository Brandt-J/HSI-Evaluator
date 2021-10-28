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
import functools
from typing import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor

from logger import getLogger
from preprocessing.routines import *

if TYPE_CHECKING:
    from logging import Logger

preprocLogger: 'Logger' = getLogger("Preprocessing")


class Preprocessor:
    label: str = ''
    specLabels: Union[None, np.ndarray] = None
    backgroundSpec: Union[None, np.ndarray] = None

    def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
        """
        Takes (N, M) array of N spectra with M wavelengths and applies the processing
        :param spectra:
        :return:
        """
        raise NotImplementedError


class DimReductProc(Preprocessor):
    label = "Dimensionality Reduction"

    def __init__(self):
        self._pca: PCA = PCA(n_components=3)
        self._tsne: TSNE = TSNE(n_components=3)
        self.updateProcAndLabel(pca=True)

    def updateProcAndLabel(self, pca: bool, numComps: int = 3) -> None:
        if pca:
            self._pca = PCA(n_components=numComps)
            self.applyToSpectra = self._pca.fit_transform
            self.label = f"PCA {numComps} Components"
        else:
            self._tsne = TSNE(n_components=numComps)
            self.applyToSpectra = self._tsne.fit_transform
            self.label = f"TSNE {numComps} Components"


class SNVProc(Preprocessor):
    label = "SNV"

    def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
        return snv(spectra)


class DetrendProc(Preprocessor):
    label = "Detrend"

    def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
        return detrend(spectra)


class NormalizeProc(Preprocessor):
    label = "Normalize"

    def __init__(self):
        super(NormalizeProc, self).__init__()
        self._normMode: NormMode = NormMode.Area
        self._updateLabel()

    def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
        return normalizeIntensities(spectra, self._normMode)

    def setNormMode(self, newMode: NormMode) -> None:
        self._normMode = newMode
        self._updateLabel()

    def _updateLabel(self) -> None:
        self.label = f"Normalization: {self._normMode}"


class MSCProc(Preprocessor):
    label = "Normalize"

    def applyToSpectra(self, spectra: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        import matplotlib.pyplot as plt
        from collections import Counter

        if labels is None:
            procSpecs: np.ndarray = msc(spectra)
        else:
            procSpecs: np.ndarray = np.zeros_like(spectra)
            for lbl in np.unique(labels):
                ind: np.ndarray = np.where(labels == lbl)[0]
                correctedSpecs: np.ndarray = msc(spectra[ind, :])


                procSpecs[ind, :] = correctedSpecs

        return procSpecs


class SavGolProc(Preprocessor):
    label = "Savitzky-Golay"

    def __init__(self):
        super(SavGolProc, self).__init__()
        self._savGolFunc: Callable[[np.ndarry], np.ndarray] = functools.partial(deriv_smooth, polydegree=2,
                                                                                derivative=1, windowSize=5)

    def applyToSpectra(self, spectra: np.ndarray) -> np.ndarray:
        return self._savGolFunc(spectra)

    def updatePreprocessor(self, winSize: int, degree: int, deriv: int) -> None:
        self._savGolFunc = functools.partial(deriv_smooth, polydegree=degree,
                                                              derivative=deriv, windowSize=winSize)
        self.label = f"Smooth {winSize} + Derivative {deriv}"
