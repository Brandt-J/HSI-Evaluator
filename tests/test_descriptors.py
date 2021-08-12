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
