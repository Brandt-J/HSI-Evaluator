import unittest
from typing import *

from dataObjects import flipIndicesHorizontally, flipIndicesVertically


class TestFlipIndices(unittest.TestCase):
    def test_flipIndices(self):
        indices: Set[int] = {0, 1, 2, 3, 4}
        cubeShape: Tuple[int, int, int] = 5, 10, 10

        flippedVertically: Set[int] = flipIndicesVertically(indices, cubeShape)
        self.assertEqual(flippedVertically, {90, 91, 92, 93, 94})

        flippedHorizontally: Set[int] = flipIndicesHorizontally(indices, cubeShape)
        self.assertEqual(flippedHorizontally, {5, 6, 7, 8, 9})
