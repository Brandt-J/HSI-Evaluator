from unittest import TestCase

from sklearn.preprocessing import LabelEncoder

from particles import ParticleHandler, Particle
from particledetection.detection import *
from classification.classifiers import BatchClassificationResult
from gui.classUI import ClassInterpretationParams


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

        specArr1, specArr2 = particleList[0].getSpectraArray(cube), particleList[1].getSpectraArray(cube)
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
        encoder: LabelEncoder = LabelEncoder().fit(np.array(["class1", "class2", "class3"]))
        highProb, lowProb = 0.8, 0.6
        highOtherProb, lowOtherProb = (1-highProb) / 2, (1-lowProb) / 2

        numClass1HighProb, numClass1LowProb = 5, 3
        numClass2HighProb, numClass2LowProb = 2, 2
        numClass3HighProb, numClass3LowProb = 2, 1

        probList = []
        for _ in range(numClass1HighProb):
            probList.append([highProb, highOtherProb, highOtherProb])
        for _ in range(numClass1LowProb):
            probList.append([lowProb, lowOtherProb, lowOtherProb])
        for _ in range(numClass2HighProb):
            probList.append([highOtherProb, highProb, highOtherProb])
        for _ in range(numClass2LowProb):
            probList.append([lowOtherProb, lowProb, lowOtherProb])
        for _ in range(numClass3HighProb):
            probList.append([highOtherProb, highOtherProb, highProb])
        for _ in range(numClass3LowProb):
            probList.append([lowOtherProb, lowOtherProb, lowProb])

        probMat = np.array(probList)
        batchRes: BatchClassificationResult = BatchClassificationResult(probMat, encoder)
        particle: Particle = Particle(0, None)
        for ignoreUnknown in [True, False]:
            for specConfCutoff in [0.7, 0.9]:
                params: ClassInterpretationParams = ClassInterpretationParams(specConfCutoff, 0.5, ignoreUnknown)
                self.assertEqual(particle.getAssignment(params), "unknown")

        particle.setBatchResult(batchRes)

        for ignoreUnknown in [True, False]:
            for specConfCutoff in [0.4, 0.7, 0.9]:  # all specs count, only high prob specs count, all specs unknown
                for partthresh in [0.5, 0.75]:  # class1 abundancy is enough, class1 abundancy is NOT enough
                    params = ClassInterpretationParams(specConfCutoff, partthresh, ignoreUnknown)
                    particle.getAssignment(params)  # make sure all go through nicely.
