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
