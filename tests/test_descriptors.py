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

import unittest
from descriptors import *


class TestTriangleDescriptor(unittest.TestCase):
    starts: list = [0, 0, 10, 4]
    peaks: list = [5, 7, 15, 14]
    ends: list = [10, 20, 17, 22]

    def test_triangle_correlation(self):
        for start, peak, end in zip(self.starts, self.peaks, self.ends):
            template: TriangleShape = TriangleShape(start, peak, end)
            data: np.ndarray = template.datapoints
            middle = round(peak-start)
            self.assertEqual(data[0], 0)
            self.assertEqual(data[middle], 1)
            self.assertEqual(data[-1], 0)

    def test_correct_signal(self):
        for start, peak, end in zip(self.starts, self.peaks, self.ends):
            for offset in [2, 3, 5.5]:
                template: TriangleShape = TriangleShape(start, peak, end)
                for signalLength in [22, 51, 100]:
                    signal: np.ndarray = np.zeros(signalLength)
                    ind1: int = start + round((end - start) / 3)
                    ind2: int = start + round((end - start) / 3 * 2)
                    signal[ind1:ind2] += offset

                    corrSignal = template._correct_signal(signal)
                    self.assertEqual(len(corrSignal), end-start)
                    for i in range(end-start):
                        if ind1 <= i+start < ind2:
                            self.assertEqual(corrSignal[i], offset)
                        else:
                            self.assertEqual(corrSignal[i], 0)

    def test_get_correlation(self):
        template: TriangleShape = TriangleShape(0, 20, 40)
        signal: np.ndarray = np.zeros(40)
        signal[10:30] = 5
        corr: float = template.get_correlation_to_signal(signal)
        self.assertTrue(corr > 0.86)

        signal[10:30] = -5
        corr = template.get_correlation_to_signal(signal)
        self.assertTrue(corr < -0.86)

        np.random.seed(10)
        signal = np.random.rand(40)
        corr = template.get_correlation_to_signal(signal)
        self.assertTrue(corr < 0.2)
