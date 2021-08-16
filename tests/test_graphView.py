from unittest import TestCase
from typing import Set, List
from PyQt5.QtCore import QPoint
import numpy as np

from gui.graphOverlays import SelectionOverlay, getBrightIndices


class TestSelectionView(TestCase):
    def test_newSelection(self):
        selectionOverlay: SelectionOverlay = SelectionOverlay()
        x0, y0 = 2, 3
        endPoint: QPoint = QPoint(4, 4)

        selectionOverlay._startDrag = x0, y0
        selectionOverlay._overlayArr = np.zeros((5, 5))
        selected: Set[int] = selectionOverlay.finishSelection(endPoint)
        """
        This is the array index layout
        00, 01, 02, 03, 04,
        05, 06, 07, 08, 09,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24
        """
        self.assertEqual(selected, set([17, 18, 19, 22, 23, 24]))


class TestGraphView(TestCase):
    def test_selectAll(self) -> None:
        width, height, channel = 10, 10, 5
        cube: np.ndarray = np.zeros((channel, width, height))
        np.random.seed(42)
        brightPixelIndices: Set[int] = set()
        i: int = 0
        for y in range(height):
            for x in range(width):
                if np.random.rand() > 0.5:
                    cube[:, y, x] = 1
                    brightPixelIndices.add(i)
                i += 1

        foundIndices: Set[int] = getBrightIndices(cube)
        self.assertEqual(brightPixelIndices, foundIndices)

