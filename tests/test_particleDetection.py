from unittest import TestCase

from particles import ParticleHandler
from particledetection.detection import *


class TestParticleHandler(TestCase):
    def test_assertUint8(self):
        floatImg: np.ndarray = np.random.rand(10, 10)
        convImg: np.ndarray = assertUint8(floatImg)
        self.assertEqual(convImg.dtype, np.uint8)
        self.assertEqual(convImg.min(), 0)
        self.assertEqual(convImg.max(), 1)

    def test_get_particles_from_image(self):
        binImg: np.ndarray = np.zeros((50, 50), dtype=np.uint8)
        # introduce three "particles"
        binImg[5:15, 5:15] = 1
        binImg[20:30, 20:30] = 1
        binImg[5:15, 30:45] = 1

        partHandler: ParticleHandler = ParticleHandler()
        self.assertEqual(len(partHandler._particles), 0)
        partHandler.getParticlesFromImage(binImg)
        self.assertEqual(len(partHandler._particles), 3)
