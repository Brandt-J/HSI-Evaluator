from unittest import TestCase

import numpy as np

from particles import ParticleHandler, Particle
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

    def test_getSpectraFromParticle(self):
        binImg: np.ndarray = np.zeros((50, 50), dtype=np.uint8)
        cube: np.ndarray = np.zeros((10, 50, 50))

        valPart1, valPart2 = np.random.rand(), np.random.rand()
        # a 10x10 px Particle
        binImg[5:15, 5:15] = 1
        cube[:, 5:15, 5:15] = valPart1

        # a 5x5 px Particle
        binImg[35:40, 35:40] = 1
        cube[:, 35:40, 35:40] = valPart2

        contours, hierarchy = cv2.findContours(binImg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        particleList: List[Particle] = [Particle(ParticleHandler.getNewParticleID(), cnt) for cnt in contours]
        self.assertEqual(len(particleList), 2)

        specArr1, specArr2 = particleList[0].getSpectra(cube), particleList[1].getSpectra(cube)
        specVals1: np.ndarray = np.unique(specArr1)
        specVals2: np.ndarray = np.unique(specArr2)

        self.assertEqual(specArr1.shape[0], 25)  # the 5x5 px Particle. If that ever fails and shows 100 px, maybe the algorithms where changed and now first the 10x10 px particle is returned??
        self.assertEqual(specArr1.shape[1], cube.shape[0])
        self.assertEqual(len(specVals1), 1)
        self.assertEqual(specVals1[0], valPart2)

        self.assertEqual(specArr2.shape[0], 100)  # the 10x10 px Particle
        self.assertEqual(specArr2.shape[1], cube.shape[0])
        self.assertEqual(len(specVals2), 1)
        self.assertEqual(specVals2[0], valPart1)

    def test_getAssignment(self) -> None:
        result: List[str] = ['class1']*9 + ['class2']*1
        particle: Particle = Particle(0, None)
        self.assertEqual(particle.getAssignment(), "unknown")

        particle.setResultFromAssignments(result)
        particle.setThreshold(0.9)
        self.assertEqual(particle.getAssignment(), "class1")

        particle.setThreshold(0.95)
        self.assertEqual(particle.getAssignment(), "unknown")
