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


from unittest import TestCase
import numpy as np
import os
import tempfile

from loadNumpyCube import loadNumpyCube


class TestImportNumpy(TestCase):
    def test_importNumpy(self) -> None:
        cube: np.ndarray = np.random.rand(20, 100, 100)
        cube[5, 3, 6] = 1e34
        cube[2, 5, 7] = 2e34

        with tempfile.TemporaryDirectory() as tmpdirname:
            savePath: str = os.path.join(tmpdirname, "testcube.npy")
            np.save(savePath, cube)

            loadedCube: np.ndarray = loadNumpyCube(savePath)
            self.assertEqual(loadedCube[5, 3, 6], 0)
            self.assertEqual(loadedCube[2, 5, 7], 0)
