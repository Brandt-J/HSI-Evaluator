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
import os
import sys
import time
import cv2
from PyQt5 import QtWidgets
from unittest import TestCase
import numpy as np
from typing import *
import tempfile
from sklearn.preprocessing import LabelEncoder

from classification.classifiers import SVM, NeuralNet, BatchClassificationResult
from classification.classifyProcedures import TrainingResult, ClassifyMode
from tests.test_specObj import getPreprocessors
from dataObjects import Sample
from particles import Particle
from gui.classUI import ClassificationUI
from gui.HSIEvaluator import MainWindow
from gui.sampleview import SampleView

if TYPE_CHECKING:
    from gui.classUI import TrainClfTab, LoadClfTab
    from collections import Counter
    from gui.graphOverlays import GraphView
    from preprocessing.preprocessors import Preprocessor


class TestBatchClassificationResult(TestCase):
    def test_get_results(self):
        encoder: LabelEncoder = LabelEncoder().fit(np.array(["class1", "class2", "class3"]))
        probMat: np.ndarray = np.array([[0.1, 0.8, 0.1],
                                        [0.4, 0.4, 0.2],
                                        [0.7, 0.3, 0.0]])

        batchRes: BatchClassificationResult = BatchClassificationResult(probMat, encoder)
        results: np.ndarray = batchRes.getResults(cutoff=0.0)
        self.assertTrue(np.array_equal(results, np.array(["class2", "class1", "class1"])))

        results = batchRes.getResults(cutoff=0.5)
        self.assertTrue(np.array_equal(results, np.array(["class2", "unknown", "class1"])))

        results = batchRes.getResults(cutoff=0.81)
        self.assertTrue(np.array_equal(results, np.array(["unknown", "unknown", "unknown"])))


class TestClassifiers(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication(sys.argv)

    def test_fitLabelEncoder(self) -> None:
        allLabels: np.ndarray = np.array(['class1'] * 10 + ['class2'] * 20 + ['class3'] * 30)
        np.random.shuffle(allLabels)
        labelsTrain: np.ndarray = allLabels[:40]
        labelsTest: np.ndarray = allLabels[40:]

        svm: SVM = SVM()
        self.assertTrue(svm._labelEncoder is None)
        svm._fitLabelEncoder(labelsTest, labelsTrain)
        self.assertEqual(list(svm._labelEncoder.classes_), ['class1', 'class2', 'class3'])

    def testTrain_Classify_and_Save(self) -> None:
        mainWin: MockMainWin = MockMainWin()
        classUI: ClassificationUI = mainWin._clfWidget
        trainTab: 'TrainClfTab' = classUI._clfSelector._trainClfTab
        loadTab: 'LoadClfTab' = classUI._clfSelector._loadClfTab

        self.assertTrue(trainTab._activeClf is not None)

        for clf in trainTab._classifiers:
            classUI._clfSelector._tabView.setCurrentIndex(classUI._clfSelector._trainIndex)  # select Train-Tab
            classUI._clfSelector._onTabChanged()
            trainTab._selectClassifier(clf.title)
            if type(clf) == NeuralNet:
                clf: NeuralNet = cast(NeuralNet, clf)
                clf._numEpochs = 2

            self._test_classifierTraining(clf, trainTab)
            self._test_classifierInference(classUI, clf, mainWin)

            with tempfile.TemporaryDirectory() as tmpdirname:
                savePath: str = os.path.join(tmpdirname, "testclfsave.clf")
                trainTab._saveClassifier(savePath)
                self.assertTrue(os.path.exists(savePath))
                curResultDict: dict = trainTab._currentTrainResult

                classUI._clfSelector._tabView.setCurrentIndex(classUI._clfSelector._trainIndex)  # select Load Tab
                classUI._clfSelector._onTabChanged()

                loadTab._loadClassifier(savePath)
                classUI._clfSelector._validResult.showResult(classUI._clfSelector._validResult._currentResults)  # have to call it manually again, signals not working here..
                self.assertDictEqual(classUI._clfSelector._validResult._currentResults, curResultDict)

                self.assertTrue(loadTab._activeClf._clf is not None)
                self.assertTrue(classUI._clfSelector.getActiveClassifier()._clf is not None)

    def _test_classifierInference(self, classUI, clf, mainWin):
        for mode in [ClassifyMode.WholeImage, ClassifyMode.Particles]:
            if mode == ClassifyMode.WholeImage:
                classUI._radioImage.setChecked(True)
                classUI._radioParticles.setChecked(False)
            else:
                classUI._radioParticles.setChecked(True)
                classUI._radioImage.setChecked(False)

            classUI._runClassification()
            while classUI._thread.is_alive():
                classUI._checkOnClassification()
                time.sleep(0.1)

            # Check that everything is correct
            if mode == ClassifyMode.WholeImage:
                allCorrect = self.graphViewsUpdatedProperly(mainWin)
                self.assertTrue(allCorrect)

            elif mode == ClassifyMode.Particles:
                self.assertParticlesCorrect(mainWin, checkForCorrectAssignment=type(clf) != NeuralNet)

    def _test_classifierTraining(self, clf, trainTab):
        trainTab._trainClassifier()
        while trainTab._thread.is_alive():
            trainTab._checKOnTraining()  # we have to call it manually here, the Qt main loop isn't running and timers don't work
            time.sleep(0.1)

        trainTab._onTrainingFinishedOrAborted()  # Again, call it manually

        result: 'TrainingResult' = trainTab._trainResult
        self.assertTrue(type(result) == TrainingResult)
        clfReport: dict = result.validReportDict
        self.assertTrue("class1" in clfReport.keys())
        self.assertTrue("class2" in clfReport.keys())
        if type(clf) != NeuralNet:  # the neural net probably performs pretty badly in this setup
            for subDict in [clfReport["class1"], clfReport["class2"]]:
                self.assertTrue(subDict["precision"] == subDict["recall"] == 1.0)  # The other classifiers should separate them perfectly.

    def assertParticlesCorrect(self, mainWin, checkForCorrectAssignment: bool):
        for sample in mainWin.getAllSamples():
            sampleParticles: List[Particle] = sample.getSampleData().getAllParticles()
            self.assertEqual(len(sampleParticles), 2)
            foundParticleClasses: Set[str] = set()
            for particle in sampleParticles:
                res: 'BatchClassificationResult' = particle._result
                resCounter: Counter = Counter(res.getResults(0.0))
                self.assertEqual(len(resCounter), 1)  # only one class found
                for clsName in resCounter.keys():  # Dict keys cannot be indexed, hence the loop...
                    foundParticleClasses.add(clsName)

            if checkForCorrectAssignment:
                self.assertEqual(len(foundParticleClasses), 2)
                self.assertTrue("class1" in foundParticleClasses)
                self.assertTrue("class2" in foundParticleClasses)

    def graphViewsUpdatedProperly(self, mainWin: 'MockMainWin'):
        allCorrect: bool = True
        for sample in mainWin.getAllSamples():
            sampleCorrect: bool = False
            graphView: 'GraphView' = sample.getGraphView()
            self.assertTrue(graphView._classOverlay._overlayArr is not None)
            sampleCorrect = True

            if not sampleCorrect:
                allCorrect = False
                break

        return allCorrect


class MockMainWin(MainWindow):
    cubeShape: np.ndarray = np.array([10, 20, 20])

    def __init__(self):
        super(MockMainWin, self).__init__()
        data1: Sample = Sample()
        data1.name = 'Sample1'
        data1.classes2Indices = {"class1": set(np.arange(20) + 20),
                                 "class2": set(np.arange(20) + 60)}
        sample1: SampleView = SampleView()
        sample1._trainCheckBox.setChecked(True)
        sample1._inferenceCheckBox.setChecked(True)
        sample1.setSampleData(data1)
        sample1.setMainWindowReferences(self)

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
        sample2.setMainWindowReferences(self)
        cube2: np.ndarray = createRandomCubeToClassLabels(self.cubeShape, data2.classes2Indices)
        sample2.setCube(cube2, np.arange(self.cubeShape[0]))
        sample2.getSampleData().particleHandler._particles = getParticlesForCube(cube2)

        self._multiSampleView._sampleviews = [sample1, sample2]

    def disableWidgets(self):
        pass

    def enableWidgets(self):
        pass

    def getPreprocessorsForClassification(self):
        returnProc: List['Preprocessor'] = []
        for proc in getPreprocessors():
            if proc.label.find("PCA") != -1:  # other preprocessors might disturb the constructed classification here (we basically separate the fake samples by their absolute offset.
                returnProc.append(proc)
        return []

    def getClassColorDict(self) -> Dict[str, Tuple[int, int, int]]:
        return {"class1": (0, 0, 0),
                "class2": (255, 255, 255),
                "unknown": (20, 20, 20)}


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
