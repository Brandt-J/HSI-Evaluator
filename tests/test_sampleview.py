import os.path
import tempfile
from unittest import TestCase
import sys
from PyQt5 import QtWidgets
from typing import List, Dict, TYPE_CHECKING, Set
import numpy as np
import pickle

from gui.HSIEvaluator import MainWindow
from gui.sampleview import MultiSampleView, SampleView, getSpectraFromIndices, Sample
from gui.graphOverlays import GraphView
from spectraObject import SpectraObject

if TYPE_CHECKING:
    from gui.classification import ClassCreator


def specDictEqual(dict1: Dict[str, np.ndarray], dict2: Dict[str, np.ndarray]) -> bool:
    isEqual: bool = True
    if not len(dict1) == len(dict2):
        isEqual = False
    else:
        for key1, key2 in zip(dict1.keys(), dict2.keys()):
            if key1 != key2:
                isEqual = False
                break

        for arr1, arr2 in zip(dict1.values(), dict2.values()):
            if not np.array_equal(arr1, arr2):
                isEqual = False
                break

    return isEqual


class TestSampleView(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication(sys.argv)  # needed for properly running tests

    def testCreateSampleView(self) -> None:
        imgClf: MainWindow = MainWindow()
        multiView: MultiSampleView = MultiSampleView(imgClf)

        sampleViews: List['SampleView'] = multiView.getSampleViews()
        self.assertEqual(len(sampleViews), 0)

        newView: 'SampleView' = multiView.addSampleView()
        self.assertTrue(newView.isActive())
        sampleViews: List['SampleView'] = multiView.getSampleViews()
        self.assertEqual(len(sampleViews), 1)

        graphView: GraphView = newView.getGraphView()
        self.assertTrue(graphView._mainWin, imgClf)

        sample2: SampleView = multiView.addSampleView()
        self.assertEqual(len(multiView.getSampleViews()), 2)

    def testSetupSampleView(self) -> None:
        sampleView: SampleView = SampleView()
        fname, cube = "testName.npy", np.ones((3, 5, 5))
        sampleView.setUp(fname, cube)

        self.assertEqual(sampleView._name, fname.split('.npy')[0])
        self.assertTrue(sampleView.getGraphView()._origCube is cube)
        self.assertTrue(sampleView._specObj.getCube() is cube)

    def testGetSpectra(self) -> None:
        imgClf: MainWindow = MainWindow()
        multiView: MultiSampleView = MultiSampleView(imgClf)

        sample1: SampleView = multiView.addSampleView()
        sample1._name = 'Sample1'
        specObj1: SpectraObject = sample1._specObj
        specObj1.setCube(np.zeros((3, 10, 10)))
        sample1._classes2Indices = {"class1": set(np.arange(5)),
                                    "class2": set(np.arange(7))}  # the indices overlap here, but we only check that the amount is correct..

        sample2: SampleView = multiView.addSampleView()
        sample2._name = 'Sample2'
        specObj2: SpectraObject = sample2._specObj
        specObj2.setCube(np.ones((3, 10, 10)))
        sample2._classes2Indices = {"class1": set(np.arange(5)),
                                    "class2": set(np.arange(3)),
                                    "class3": set(np.arange(9))}

        # only get spectra of first sample
        sample1._activeBtn.setChecked(True)
        sample2._activeBtn.setChecked(False)

        spectraSample1: Dict[str, np.ndarray] = multiView.getLabelledSpectraFromActiveView()
        self.assertEqual(len(spectraSample1), 2)
        self.assertTrue("class1" in spectraSample1.keys() and "class2" in spectraSample1.keys())
        self.assertTrue(np.array_equal(spectraSample1["class1"].shape, np.array([5, 3])))
        self.assertTrue(np.array_equal(spectraSample1["class2"].shape, np.array([7, 3])))

        # only get spectra of second sample
        sample1._activeBtn.setChecked(False)
        sample2._activeBtn.setChecked(True)

        spectraSample2: Dict[str, np.ndarray] = multiView.getLabelledSpectraFromActiveView()
        self.assertEqual(len(spectraSample2), 3)
        self.assertTrue("class1" in spectraSample2.keys() and "class2" in spectraSample2.keys() and "class3" in spectraSample2.keys())
        self.assertTrue(np.array_equal(spectraSample2["class1"].shape, np.array([5, 3])))
        self.assertTrue(np.array_equal(spectraSample2["class2"].shape, np.array([3, 3])))
        self.assertTrue(np.array_equal(spectraSample2["class3"].shape, np.array([9, 3])))

        # not get both samples:
        allSpecs: Dict[Dict[str, :]] = multiView.getLabelledSpectraFromAllViews()
        self.assertEqual(len(allSpecs), 2)
        self.assertTrue(specDictEqual(allSpecs["Sample1"], spectraSample1))
        self.assertTrue(specDictEqual(allSpecs["Sample2"], spectraSample2))

    def test_getSpectraFromCube(self) -> None:
        cube: np.ndarray = np.zeros((3, 6, 5))
        indices: np.ndarray = np.arange(cube.shape[1] * cube.shape[2])
        # we set the cube values to the corresponding index
        for ind in indices:
            y, x = np.unravel_index(ind, cube.shape[1:])
            cube[:, y, x] = ind

        specs: np.ndarray = getSpectraFromIndices(indices, cube)
        self.assertEqual(specs.shape[0], len(indices))
        self.assertEqual(specs.shape[1], cube.shape[0])
        for i, ind in enumerate(indices):
            self.assertTrue(np.all(specs[i, :] == ind))

    def test_closeSample(self) -> None:
        def getSampleName(ind: int) -> str:
            return f"Sample {ind}"

        imgClf: MainWindow = MainWindow()
        multiView: MultiSampleView = MultiSampleView(imgClf)

        numSamplesOrig: int = 4
        for i in range(numSamplesOrig):
            newView: SampleView = multiView.addSampleView()
            newView._name = getSampleName(i)

        for i in reversed(range(numSamplesOrig)):
            multiView._viewClosed(getSampleName(i))
            self.assertEqual(len(multiView.getSampleViews()), i)
            remainingSampleNames = [view.getName() for view in multiView.getSampleViews()]
            self.assertTrue(getSampleName(i) not in remainingSampleNames)

    def test_saveSample(self) -> None:
        imgClf: MainWindow = MainWindow()

        classesSample1: Dict[str, Set[int]] = {'class1': {0, 1, 2, 3, 4},
                                               'class2': {5, 6, 7, 8}}
        classesSample2: Dict[str, Set[int]] = {'class1': {0, 1, 2, 3, 4, 5, 7},
                                               'class2': {8, 9, 10, 11},
                                               'class3': {12, 13, 14}}

        # Create MultiView and two SampleViews.
        multiView: MultiSampleView = MultiSampleView(imgClf)
        sample1: SampleView = multiView.addSampleView()
        sample1._name = 'Sample1'
        sample1._sampleData.filePath = os.path.join(r'FakeDir/Sample1.npy')
        sample1._classes2Indices = classesSample1

        sample2: SampleView = multiView.addSampleView()
        sample2._name = 'Sample2'
        sample2._sampleData.filePath = os.path.join(r'FakeDir/Sample2.npy')
        sample2._classes2Indices = classesSample2

        with tempfile.TemporaryDirectory() as tmpdirname:
            multiView.getSampleSaveDirectory = lambda: tmpdirname
            multiView._saveSampleView(sample1)
            savename: str = os.path.join(multiView.getSampleSaveDirectory(), sample1.getSaveFileName())
            self.assertTrue(os.path.exists(savename))
            with open(savename, "rb") as fp:
                savedData: 'Sample' = pickle.load(fp)
            self.assertDictEqual(savedData.classes2Indices, classesSample1)
            self.assertEqual(savedData.name, 'Sample1')
            self.assertEqual(savedData.filePath, r'FakeDir/Sample1.npy')

            multiView._saveSampleView(sample2)
            savename: str = os.path.join(multiView.getSampleSaveDirectory(), sample2.getSaveFileName())
            self.assertTrue(os.path.exists(savename))
            with open(savename, "rb") as fp:
                savedData: 'Sample' = pickle.load(fp)
            self.assertDictEqual(savedData.classes2Indices, classesSample2)
            self.assertEqual(savedData.name, 'Sample2')
            self.assertEqual(savedData.filePath, r'FakeDir/Sample2.npy')

    def test_loadFromSample(self) -> None:
        imgClf: MainWindow = MainWindow()
        sample: Sample = Sample()
        sample.name = 'Sample1'
        sample.classes2Indices = {'Background': {1, 2, 3, 4},
                                  'class2': {5, 6, 7, 8}}
        testCube: np.ndarray = np.random.rand(3, 10, 10)

        multiView: MultiSampleView = MultiSampleView(imgClf)
        self.assertTrue(len(multiView._sampleviews) == 0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            sample.filePath = os.path.join(tmpdirname, "testCube.npy")
            saveName: str = os.path.join(tmpdirname, sample.getFileHash() + '.pkl')
            with open(saveName, "wb") as fp:
                pickle.dump(sample, fp)
            np.save(sample.filePath, testCube)
            multiView.loadSampleViewFromFile(saveName)

        self.assertTrue(len(multiView._sampleviews) == 1)
        createdSample: SampleView = multiView._sampleviews[0]
        self.assertDictEqual(sample.__dict__, createdSample.getSampleData().__dict__)
        classCreator: ClassCreator = imgClf._clsCreator
        presentClasses: List[str] = [cls.name for cls in classCreator._classes]
        for cls in sample.classes2Indices.keys():
            self.assertTrue(cls in presentClasses)

        self.assertTrue(np.array_equal(createdSample._specObj.getCube(), testCube))
        self.assertTrue(np.array_equal(createdSample._graphView._origCube, testCube))

