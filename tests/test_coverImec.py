import unittest

import cv2
import numpy as np
from imageOperations.coverIMEC import *


class TestCoverImec(unittest.TestCase):
    def test_getCoveredArea(self):
        binImg: np.ndarray = np.zeros((100, 100), dtype=np.uint8)
        binImg[20:40, 20:40] = 255
        binImg[60:90, 60:90] = 255
        contours: List[np.ndarray] = getContours(binImg)[::-1]  # they would normally come out lat to first in that example

        rect: Rect = Rect(0, 0, 70, 70)

        area1 = cv2.contourArea(contours[0])
        expectedFraction1 = 400 / area1  # example 1: the patch is all 400 pixels should be covered
        self.assertEqual(rect.getContourOverlap(contours[0]), expectedFraction1)

        area2 = cv2.contourArea(contours[1])
        expectedFraction2 = 100 / area2  # in example 2, only 100 pixels are covered
        self.assertEqual(rect.getContourOverlap(contours[1]), expectedFraction2)

    def test_getCenterDistances(self) -> None:
        cnt1: np.ndarray = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]])  # Center = 5, 5
        cnt2: np.ndarray = np.array([[[0, -10]], [[0, 10]], [[10, 10]], [[10, -10]]])  # Center = 5, 0
        cnt3: np.ndarray = np.array([[[0, -10]], [[0, 10]], [[20, 10]], [[20, -10]]])  # Center = 10, 0
        cnt4: np.ndarray = np.array([[[0, 0]], [[0, 20]], [[20, 20]], [[20, 0]]])  # Center = 10, 10

        distances: np.ndarray = getCenterDistances([cnt1, cnt2, cnt3, cnt4])
        distDiagonal5 = np.sqrt(25 + 25)  # Length of diagonal with dx = 5 and dy = 5
        self.assertEqual(distances[0, 1], 5)
        self.assertEqual(distances[0, 2], distDiagonal5)
        self.assertEqual(distances[0, 3], distDiagonal5)

        self.assertEqual(distances[1, 2], 5)
        self.assertEqual(distances[1, 3], np.sqrt(25 + 100))
