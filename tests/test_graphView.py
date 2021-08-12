from unittest import TestCase
from typing import Set
from PyQt5.QtCore import QPoint
import numpy as np

from gui.graphOverlays import SelectionOverlay


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
