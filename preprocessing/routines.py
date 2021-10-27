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
import time
from enum import Enum
import numba
import numpy as np
from scipy.signal import savgol_filter
from scipy.linalg import solveh_banded

from preprocessing.numba_polyfit import fit_poly


def als_baseline(intensities, asymmetry_param=0.05, smoothness_param=1e4, max_iters=5, conv_thresh=1e-5, verbose=False):
    """
    The als_baseline method, including the WhittakerSmoother Class was taken from
    https://github.com/all-umass/superman/blob/master/superman/baseline/als.py
    with permission from github user CJ Carey (perimosocordiae)

    Computes the asymmetric least squares baseline.
    * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
    smoothness_param: Relative importance of smoothness of the predicted response.
    asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                         Setting p=1 is effectively a hinge loss.
    """
    z = intensities
    if max_iters > 0:
        smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
        # Rename p for concision.
        p = asymmetry_param
        # Initialize weights.
        w = np.ones(intensities.shape[0])
        for i in range(max_iters):
            z = smoother.smooth(w)
            mask = intensities > z
            new_w = p * mask + (1 - p) * (~mask)
            conv = np.linalg.norm(new_w - w)
            if verbose:
                print(i + 1, conv)
            if conv < conv_thresh:
                break
            w = new_w

    return z


class WhittakerSmoother(object):
    def __init__(self, signal: np.ndarray, smoothness_param, deriv_order=1):
        self.y = signal.copy()
        assert deriv_order > 0, 'deriv_order must be an int > 0'
        # Compute the fixed derivative of identity (D).
        d = np.zeros(deriv_order * 2 + 1, dtype=int)
        d[deriv_order] = 1
        d = np.diff(d, n=deriv_order)
        n = self.y.shape[0]
        k = len(d)
        s = float(smoothness_param)

        # Here be dragons: essentially we're faking a big banded matrix D,
        # doing s * D.T.dot(D) with it, then taking the upper triangular bands.
        diag_sums = np.vstack([
            np.pad(s * np.cumsum(d[-i:] * d[:i]), ((k - i, 0),), 'constant')
            for i in range(1, k + 1)])
        upper_bands = np.tile(diag_sums[:, -1:], n)
        upper_bands[:, :k] = diag_sums
        for i, ds in enumerate(diag_sums):
            upper_bands[i, -i - 1:] = ds[::-1][:i + 1]
        self.upper_bands = upper_bands

    def smooth(self, w):
        foo = self.upper_bands.copy()
        foo[-1] += w  # last row is the diagonal
        return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)


def mapSpecToWavenumbers(spec: np.ndarray, targetWavenumbers: np.ndarray) -> np.ndarray:
    newSpec = np.zeros((len(targetWavenumbers), 2))
    newSpec[:, 0] = targetWavenumbers
    for i in range(newSpec.shape[0]):
        closestIndex = np.argmin(np.abs(spec[:, 0] - targetWavenumbers[i]))
        newSpec[i, 1] = spec[closestIndex, 1]
    return newSpec


@numba.njit
def normalizeIntensities(input_data: np.ndarray, mode: 'NormMode') -> np.ndarray:
    """
    Normalizes each set of intensities
    :param input_data: (NxM) array of N spectra with M features
    :param mode: Mode for the normalization.
    :return:
    """
    assert mode in [NormMode.Area, NormMode.Length, NormMode.Max]

    normalized: np.ndarray = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):

        if mode == NormMode.Area:
            divisor = np.trapz(input_data[i, :])
        elif mode == NormMode.Length:
            divisor = np.linalg.norm(input_data[i, :])
        else:
            divisor = np.max(input_data[i, :])

        normalized[i, :] = input_data[i, :] / divisor
    return normalized


def autoscale(input_data: np.ndarray) -> np.ndarray:
    """
    Autoscales all the variables, so they have the same importance for subsequent analysis
    and same chance to be picked up.
    Might not be reasonable for spectral data, here the relative intensities do have an importance: Close-to-zero
    wavelengths also indicate less relevant data than wavelengths with high intensities.
    :param input_data: (MxN) array of M samples with N features.
    :return:
    """
    scaled = np.ndarray = np.zeros_like(input_data)
    for i in range(input_data.shape[1]):
        data = input_data[:, i]
        scaled[:, i] = (data - np.mean(data)) / np.std(data)

    return scaled


@numba.njit
def snv(input_data: np.ndarray) -> np.ndarray:
    """
    Standard normal variate Correction. "Autoscale" of rows.
    :param input_data: Shape (NxM) array of N samples with M features
    :return: corrected data in same shape
    """
    output_data: np.ndarray = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        output_data[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])
    return output_data


@numba.njit
def detrend(input_data: np.ndarray) -> np.ndarray:
    """
    Removes a linear baseline.
    :param input_data: Shape (NxM) array of N samples with M features
    :return: corrected data in same shape
    """
    output_data: np.ndarray = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        baseline = np.linspace(input_data[i, 0], input_data[i, -1], len(input_data[i, :]))
        output_data[i, :] = input_data[i, :] - baseline
    return output_data


def deriv_smooth(input_data: np.ndarray, polydegree: int, derivative: int = 0, windowSize: int = 5) -> np.ndarray:
    """
    Applies Savitzky-Golay smoothing to all given data, if desired with derivative.
    :param input_data: Shape (NxM) array of N samples with M features
    :param polydegree: Polynomial degree
    :param derivative: Which derivative to calculate.
    :param windowSize: integer, the window size for smoothing.
    :return: corrected data in same shape
    """
    # Checks for Savitzky-Golay bounds
    if windowSize > input_data.shape[1]:
        windowSize = input_data.shape[1]
    if windowSize % 2 == 0:
        windowSize += 1
    if polydegree >= windowSize:
        windowSize = polydegree + 1

    output_data = savgol_filter(input_data, windowSize, polydegree, deriv=derivative)
    return output_data


@numba.njit
def msc(spectra: np.ndarray):
    """
    Multiplicative Scatter Correction technique performed with mean of the sample data as the reference.
    Adapted from: https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/
    :param spectra: MxN array of M spectra with N wavenumbers
    :returns: corrected spectra: Scatter corrected spectra data
    """
    spectra = spectra.copy()
    # mean centre correction
    for i in range(spectra.shape[0]):
        spectra[i, :] -= spectra[i, :].mean()

    # Estimate ref spectrum as mean of input spectra
    ref = np.zeros(spectra.shape[1])
    for i in range(spectra.shape[1]):
        ref[i] = np.mean(spectra[:, i])

    # Define a new array and populate it with the corrected data
    data_msc = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        # Run regression
        fit = fit_poly(ref, spectra[i, :], deg=1)
        # Apply correction
        if not fit[0] == 0:
            corrected = (spectra[i, :] - fit[1]) / fit[0]
        else:
            corrected = spectra[i, :]

        data_msc[i, :] = corrected

    assert np.all(np.isfinite(data_msc))
    return data_msc


class NormMode(Enum):
    Area = 0
    Length = 1
    Max = 2


if __name__ == '__main__':
    # Some benchmarking with random data...

    arr: np.ndarray = np.random.rand(250000, 100)
    for _ in range(2):
        t0 = time.time()
        normalizeIntensities(arr, NormMode.Length)
        print(f"normalize {time.time()-t0} seconds")

        t0 = time.time()
        deriv_smooth(arr, windowSize=11, derivative=1, polydegree=2)
        print(f"savgol {time.time() - t0} seconds")

        t0 = time.time()
        snv(arr)
        print(f"snv {time.time() - t0} seconds")

        t0 = time.time()
        detrend(arr)
        print(f"detrend {time.time() - t0} seconds")

        t0 = time.time()
        msc(arr)
        print(f"msc {time.time() - t0} seconds")
