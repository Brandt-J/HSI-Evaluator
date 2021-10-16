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
import time
import cv2
from PyQt5 import QtWidgets
from unittest import TestCase
import numpy as np
from typing import *

from classification.classifiers import SVM, NeuralNet
from classification.classifyProcedures import TrainingResult, ClassifyMode
from dataObjects import Sample
from particles import Particle
from gui.classUI import ClassificationUI
from gui.sampleview import SampleView
from tests.test_specObj import getPreprocessors

if TYPE_CHECKING:
    from collections import Counter
    from gui.graphOverlays import GraphView


class TestClassifiers(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication(sys.argv)

    def test_SetUniqueLabels(self) -> None:
        allLabels: np.ndarray = np.array(['class1']*10 + ['class2']*20 + ['class3']*30)
        np.random.shuffle(allLabels)
        labelsTrain: np.ndarray = allLabels[:40]
        labelsTest: np.ndarray = allLabels[40:]

        svm: SVM = SVM()
        self.assertDictEqual(svm._uniqueLabels, {})
        svm._setUniqueLabels(labelsTest, labelsTrain)
        self.assertDictEqual(svm._uniqueLabels, {'class1': 0,
                                                 'class2': 1,
                                                 'class3': 2})

    def testTrainAndClassify(self) -> None:
        mainWin: MockMainWin = MockMainWin()
        classUI: ClassificationUI = ClassificationUI(mainWin)
        self.assertFalse(classUI._applyBtn.isEnabled())
        self.assertTrue(classUI._activeClf is not None)

        for clf in classUI._classifiers:
            classUI._activateClassifier(clf.title)
            if type(clf) == NeuralNet:
                clf: NeuralNet = cast(NeuralNet, clf)
                clf._numEpochs = 2

            # Train the classifier
            classUI._trainClassifier()
            while classUI._trainProcessWindow._process.is_alive():
                classUI._trainProcessWindow._checkOnProcess()  # we have to call it manually here, the Qt main loop isn't running and timers don't work
                time.sleep(0.1)

            result: 'TrainingResult' = classUI._trainProcessWindow.getResult()
            self.assertTrue(type(result) == TrainingResult)
            clfReport: dict = result.validReportDict
            self.assertTrue("class1" in clfReport.keys())
            self.assertTrue("class2" in clfReport.keys())
            for subDict in [clfReport["class1"], clfReport["class2"]]:
                if type(clf) != NeuralNet:  # the neural net probably performs pretty badly in this setup
                    self.assertTrue(subDict["precision"] == subDict["recall"] == 1.0)  # The other classifiers should separate them perfectly.

            if type(clf) == NeuralNet:
                self.assertTrue(classUI._activeClf._currentTraininghash == result.classifier._currentTraininghash)
            else:
                self.assertTrue(classUI._activeClf._clf is result.classifier._clf)  # The actual classifier has to be the very same object!

            # Now we test inference
            self.assertTrue(classUI._applyBtn.isEnabled())
            for mode in [ClassifyMode.WholeImage, ClassifyMode.Particles]:
                if mode == ClassifyMode.WholeImage:
                    classUI._radioImage.setChecked(True)
                    classUI._radioParticles.setChecked(False)
                else:
                    classUI._radioParticles.setChecked(True)
                    classUI._radioImage.setChecked(False)

                classUI._runClassification()
                while classUI._inferenceProcessWindow._process.is_alive():
                    classUI._inferenceProcessWindow._checkOnProcess()
                    time.sleep(0.1)

                result: List['Sample'] = classUI._inferenceProcessWindow.getResult()
                self.assertTrue(type(result) == list)
                self.assertEqual(len(result), 2)

                # Check that everything is correct
                if mode == ClassifyMode.WholeImage:
                    allCorrect = self.graphViewsUpdatedProperly(mainWin, result)
                    self.assertTrue(allCorrect)

                elif mode == ClassifyMode.Particles:
                    self.assertParticlesCorrect(mainWin, checkForCorrectAssignment=type(clf) != NeuralNet)

    def assertParticlesCorrect(self, mainWin, checkForCorrectAssignment: bool):
        for sample in mainWin.getAllSamples():
            sampleParticles: List[Particle] = sample.getSampleData().getAllParticles()
            self.assertEqual(len(sampleParticles), 2)
            foundParticleClasses: Set[str] = set()
            for particle in sampleParticles:
                res: 'Counter' = particle._result
                self.assertEqual(len(res), 1)  # only one class found
                for clsName in res.keys():  # Dict keys cannot be indexed, hence the loop...
                    foundParticleClasses.add(clsName)

            if checkForCorrectAssignment:
                self.assertEqual(len(foundParticleClasses), 2)
                self.assertTrue("class1" in foundParticleClasses)
                self.assertTrue("class2" in foundParticleClasses)

    def graphViewsUpdatedProperly(self, mainWin: 'MockMainWin', finishedSamples: List['Sample']):
        allCorrect: bool = True
        for finishedSample in finishedSamples:
            sampleCorrect: bool = False
            for sample in mainWin.getAllSamples():
                if sample.getName() == finishedSample.name:
                    graphView: 'GraphView' = sample.getGraphView()
                    self.assertTrue(graphView._classOverlay._overlayArr is not None)
                    sampleCorrect = True
                    break
            if not sampleCorrect:
                allCorrect = False
                break
        return allCorrect


class MockMainWin:
    cubeShape: np.ndarray = np.array([10, 20, 20])

    def __init__(self):
        data1: Sample = Sample()
        data1.name = 'Sample1'
        data1.classes2Indices = {"class1": set(np.arange(20) + 20),
                                 "class2": set(np.arange(20) + 60)}
        sample1: SampleView = SampleView()
        sample1._trainCheckBox.setChecked(True)
        sample1._inferenceCheckBox.setChecked(True)
        sample1.setSampleData(data1)

        cube1: np.ndarray = createRandomCubeToClassLabels(self.cubeShape, data1.classes2Indices)
        sample1.setCube(cube1, np.arange(self.cubeShape[0]))
        sample1.getSampleData().particleHandler._particles = getParticlesForCube(cube1)

        data2: Sample = Sample()
        data2.name = 'Sample2'
        data2.classes2Indices = {"class1": set(np.arange(20) + 20),
                                 "class2": set(np.arange(20) + 60)}
        sample2: SampleView = SampleView()
        sample2._trainCheckBox.setChecked(True)
        sample2._inferenceCheckBox.setChecked(True)
        sample2.setSampleData(data2)
        cube2: np.ndarray = createRandomCubeToClassLabels(self.cubeShape, data2.classes2Indices)
        sample2.setCube(cube2, np.arange(self.cubeShape[0]))
        sample2.getSampleData().particleHandler._particles = getParticlesForCube(cube2)

        self._samples: List['SampleView'] = [sample1, sample2]

    def disableWidgets(self):
        pass

    def enableWidgets(self):
        pass

    def getPreprocessors(self):
        return getPreprocessors()

    def getAllSamples(self) -> List['SampleView']:
        return self._samples

    def getClassColorDict(self) -> Dict[str, Tuple[int, int, int]]:
        return {"class1": (0, 0, 0),
                "class2": (255, 255, 255)}

    def getBackgroundsOfAllSamples(self) -> Dict[str, np.ndarray]:
        backgrounds: Dict[str, np.ndarray] = {}
        for sample in self._samples:
            backgrounds[sample.getName()] = np.zeros(self.cubeShape[0])
        return backgrounds


def createRandomCubeToClassLabels(cubeShape: np.ndarray, cls2Ind: Dict[str, Set[int]]) -> np.ndarray:
    """
    Creates a spec Cube with random values between 0 and 1. Then, for each class, at the given indices the values
    are increased by 1 (subsequently).
    """
    cube: np.ndarray = np.random.rand(cubeShape[0], cubeShape[1], cubeShape[2]) * 0.1
    for i, indices in enumerate(cls2Ind.values(), start=1):
        j: int = 0
        for y in range(cubeShape[1]):
            for x in range(cubeShape[2]):
                if j in indices:
                    cube[:, y, x] += i
                j += 1
    return cube


def getParticlesForCube(cube: np.ndarray) -> Dict[int, Particle]:
    """
    Takes the spectra cube and creates a particle list for it.
    """
    avgImg: np.ndarray = np.mean(cube, axis=0)
    binImg: np.ndarray = np.zeros_like(avgImg)
    binImg[avgImg > 1] = 1
    contours, _ = cv2.findContours(binImg.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    particles: Dict[int, Particle] = {}
    for i, cnt in enumerate(contours):
        particles[i] = Particle(i, cnt)
    return particles
