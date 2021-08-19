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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from typing import List, Union, Dict, TYPE_CHECKING
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from logger import getLogger

if TYPE_CHECKING:
    from logging import Logger


class PCAPlot(FigureCanvas):
    def __init__(self):
        self._figure: plt.Figure = plt.Figure()
        super(PCAPlot, self).__init__(self._figure)
        self._logger: 'Logger' = getLogger("PCA-Plot")
        self._ax: plt.Axes = self._figure.add_subplot()
        self._spectraForPCA: Union[None, np.ndarray] = None
        self._colors: List[List[float]] = []  # lists the color of each data point
        self._allNames: List[str] = []  # lists the label names of each data point
        self._name2colors: Dict[str, List[float]] = {}  # connects legend label to plot color
        self._name2lines: Dict[str, Union[str, tuple]] = {}  # connects legend label to line style

    def resetPlots(self) -> None:
        """
        Called before starting to plot a new set of spectra.
        """
        self._ax.clear()
        self._spectraForPCA = None
        self._colors = []
        self._allNames = []
        self._name2colors = {}
        self._name2lines = {}

    def addSpectraToPCA(self, spectra: np.ndarray, linestyle: Union[str, tuple], color: List[float], legendName: str) -> None:
        """
        Plots the spectra array.
        :param spectra: (NxM) array of N spectra with M wavenumbers
        :param linestyle: the linestyle code to use
        :param color: The rgb color to use (or rgba)
        :param legendName: The legendname of the given spec set.
        """
        if self._spectraForPCA is None:
            self._spectraForPCA = spectra
        else:
            self._spectraForPCA = np.vstack((self._spectraForPCA, spectra))

        self._colors += [color] * spectra.shape[0]
        self._allNames += [legendName] * spectra.shape[0]
        self._name2colors[legendName] = color
        self._name2lines[legendName] = linestyle

    def finishPlotting(self) -> None:
        """
        Called after finishing plotting a set of spectra.
        """
        if self._spectraForPCA is not None:
            if self._spectraForPCA.shape[0] <= 2:
                self._logger.warning(f"Not doing PCA, because only {self._spectraForPCA.shape[0]} spectra were obtained.")
            else:
                pca = PCA(n_components=2)
                princComps: np.ndarray = pca.fit_transform(self._spectraForPCA)
                self._ax.scatter(princComps[:, 0], princComps[:, 1], color=self._colors)
                self._ax.set_xlabel("Scores on PC1")
                self._ax.set_ylabel("Scores on PC2")
                self._drawLegend()
                self._drawConfidenceEllipses(princComps)
                try:
                    self.draw()
                except ValueError as e:
                    self._logger.warning(f"Could not update PCA plot: {e}")

    def _drawLegend(self) -> None:
        """
        Draws the legend into the plot
        """
        lines = []
        # draw a point for each label
        for name, color in self._name2colors.items():
            line,  = self._ax.plot(0, 0, linestyle=self._name2lines[name], color=color, label=name)
        self._ax.legend()  # create the legend based on that

        for line in lines:
            line.remove()

    def _drawConfidenceEllipses(self, princComps: np.ndarray) -> None:
        """
        Draws the confidence ellipses for the data.
        """
        nameArr: np.ndarray = np.array(self._allNames)
        for name, color in self._name2colors.items():
            points: np.ndarray = getXYOfName(np.array(name), nameArr, princComps)
            confidence_ellipse(points, self._ax, edgecolor=color, linestyle=self._name2lines[name])


def getXYOfName(name: np.ndarray, allNames: np.ndarray, datapoints: np.ndarray) -> np.ndarray:
    indices: np.ndarray = np.unique(np.where(allNames == name)[0])
    return datapoints[indices, :]


def confidence_ellipse(princComps: np.ndarray, ax: plt.Axes, edgecolor: List[float],
                       linestyle: Union[str, tuple], n_std: float = 3.0):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    Adapted from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py
    Parameters
    ----------
    :param princComps: (Nx2) princ comps of N data points
    :param ax: matplotlib.axes.Axes The axes object to draw the ellipse into.
    :param edgecolor: color to use
    :param n_std : float The number of standard deviations to determine the ellipse's radiuses.
    :param linestyle: the linestyle to use
    """
    x, y = princComps[:, 0], princComps[:, 1]
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    facecolor = [edgecolor[0], edgecolor[1], edgecolor[2], 0.3]
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, edgecolor=edgecolor,
                      facecolor=facecolor, linestyle=linestyle)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
