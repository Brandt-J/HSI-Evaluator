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

import numpy as np


class SpectralDescriptor(object):
    def __init__(self, start: int, end: int) -> None:
        super(SpectralDescriptor, self).__init__()
        self.start = start
        self.end = end
        self.datapoints: np.ndarray = np.array([])
        self._get_datapoints()

    @property
    def length(self) -> int:
        return self.end - self.start

    def get_correlation_to_signal(self, signal: np.ndarray) -> float:
        corrSignal: np.ndarray = self._correct_signal(signal)
        corr: np.ndarray = np.corrcoef(corrSignal, self.datapoints)
        return corr[1, 0]

    def _correct_signal(self, signal: np.ndarray) -> np.ndarray:
        assert len(signal) >= self.length
        signal = signal[self.start:self.end]
        signal -= min([signal[0], signal[-1]])
        return signal

    def _get_datapoints(self) -> None:
        raise NotImplementedError


class TriangleShape(SpectralDescriptor):
    def __init__(self, start: int, peak: int, end: int) -> None:
        self.peak: int = peak
        super(TriangleShape, self).__init__(start, end)

    def _get_datapoints(self) -> None:
        peak: int = round(self.peak - self.start)
        data: np.ndarray = np.zeros(self.length)
        slope1: float = 1 / peak
        slope2: float = - 1 / (self.end - self.peak - 1)
        for i in range(self.length):
            if i < peak:
                data[i] = i * slope1
            elif i == peak:
                data[i] = 1.0
            else:
                data[i] = (i-peak)*slope2 + 1

        self.datapoints = data
