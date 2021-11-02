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
import sys
from unittest import TestCase
from typing import Set
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QPoint
import numpy as np

from gui.graphOverlays import SelectionOverlay, getBrightOrDarkIndices, GraphOverlays


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
    def test_selectBrightOrDark(self) -> None:
        width, height, channel = 10, 10, 5
        cube: np.ndarray = np.zeros((channel, width, height))
        np.random.seed(42)
        brightPixelIndices: Set[int] = set()
        darkPixelsIndices: Set[int] = set()
        i: int = 0
        for y in range(height):
            for x in range(width):
                if np.random.rand() > 0.5:
                    cube[:, y, x] = 1
                    brightPixelIndices.add(i)
                else:
                    darkPixelsIndices.add(i)
                i += 1

        foundBrightIndices: Set[int] = getBrightOrDarkIndices(cube, 1.0, 128, bright=True)
        self.assertEqual(brightPixelIndices, foundBrightIndices)

        foundDarkIndices: Set[int] = getBrightOrDarkIndices(cube, 1.0, 128, bright=False)
        self.assertEqual(darkPixelsIndices, foundDarkIndices)
