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

import os.path
import tempfile
from unittest import TestCase
import sys
from PyQt5 import QtWidgets
from typing import List, Dict, TYPE_CHECKING, Set
import numpy as np
import pickle

from gui.HSIEvaluator import MainWindow
from gui.multisampleview import MultiSampleView, Sample
from gui.sampleview import SampleView
from gui.graphOverlays import GraphOverlays, ThresholdSelector
from spectraObject import SpectraObject, getSpectraFromIndices, WavelengthsNotSetError

if TYPE_CHECKING:
    from gui.classUI import ClassCreator
    from dataObjects import View


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

        graphView: GraphOverlays = newView.getGraphOverlayObj()
        self.assertTrue(graphView._mainWin, imgClf)

        _: SampleView = multiView.addSampleView()
        self.assertEqual(len(multiView.getSampleViews()), 2)

    def testSetupSampleView(self) -> None:
        sampleView: SampleView = SampleView()
        fname, cube, wavelenghts = "testName.npy", np.ones((3, 5, 5)), np.arange(3)
        sampleView.setUp(fname, cube, wavelenghts)

        self.assertEqual(sampleView._name, fname.split('.npy')[0])
        self.assertTrue(sampleView.getGraphOverlayObj()._origCube is cube)
        self.assertTrue(np.array_equal(sampleView.getWavelengths(), np.arange(3)))

    def testGetSpectra(self) -> None:
        imgClf: MainWindow = MainWindow()
        multiView: MultiSampleView = MultiSampleView(imgClf)

        sample1: SampleView = multiView.addSampleView()
        sample1._name = 'Sample1'
        specObj1: SpectraObject = sample1._sampleData.specObj
        specObj1.setCube(np.zeros((3, 10, 10)))
        sample1._classes2Indices = {"class1": set(np.arange(5)),
                                    "class2": set(np.arange(7))}  # the indices overlap here, but we only check that the amount is correct..

        sample2: SampleView = multiView.addSampleView()
        sample2._name = 'Sample2'
        specObj2: SpectraObject = sample2._sampleData.specObj
        specObj2.setCube(np.ones((3, 10, 10)))
        sample2._classes2Indices = {"class1": set(np.arange(5)),
                                    "class2": set(np.arange(3)),
                                    "class3": set(np.arange(9))}

        # only get spectra of first sample
        sample1._isActive = True
        sample2._isActive = False

        spectraSample1: Dict[str, np.ndarray] = multiView.getLabelledSpectraFromActiveView().getDictionary()

        self.assertEqual(len(spectraSample1), 2)
        self.assertTrue("class1" in spectraSample1.keys() and "class2" in spectraSample1.keys())
        self.assertTrue(np.array_equal(spectraSample1["class1"].shape, np.array([5, 3])))
        self.assertTrue(np.array_equal(spectraSample1["class2"].shape, np.array([7, 3])))

        # only get spectra of second sample
        sample1._isActive = False
        sample2._isActive = True

        spectraSample2: Dict[str, np.ndarray] = multiView.getLabelledSpectraFromActiveView().getDictionary()
        self.assertEqual(len(spectraSample2), 3)
        self.assertTrue("class1" in spectraSample2.keys() and "class2" in spectraSample2.keys() and "class3" in spectraSample2.keys())
        self.assertTrue(np.array_equal(spectraSample2["class1"].shape, np.array([5, 3])))
        self.assertTrue(np.array_equal(spectraSample2["class2"].shape, np.array([3, 3])))
        self.assertTrue(np.array_equal(spectraSample2["class3"].shape, np.array([9, 3])))

        # not get both samples:
        allSpecs: Dict[Dict[str, :]] = multiView.getLabelledSpectraFromAllViews().getSampleDictionary()
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
            multiView._closeSample(getSampleName(i))
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
        multiView: MultiSampleView = imgClf._multiSampleView
        sample1: SampleView = multiView.addSampleView()
        sample1._name = 'Sample1'
        sample1._sampleData.filePath = os.path.join(r'FakeDir/Sample1.npy')
        sample1._classes2Indices = classesSample1

        sample2: SampleView = multiView.addSampleView()
        sample2._name = 'Sample2'
        sample2._sampleData.filePath = os.path.join(r'FakeDir/Sample2.npy')
        sample2._classes2Indices = classesSample2

        # Test saving the individual samples
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

        # Test saving the view:
        # Set a preprocessing stack
        # TODO: REIMPLEMENT
        # preprocSelector: 'PreprocessingSelector' = imgClf._preprocSelector
        # preprocSelector._selected = preprocSelector._available  # just select them all
        # selectedNames: List[str] = [lbl.text() for lbl in preprocSelector._selected]

        with tempfile.TemporaryDirectory() as tmpdirname:
            multiView.getViewSaveDirectory = lambda: tmpdirname
            sample1.getSampleData().filePath = os.path.join(tmpdirname, "cube1.npy")
            sample2.getSampleData().filePath = os.path.join(tmpdirname, "cube2.npy")
            # create fake datacubes
            np.save(sample1.getSampleData().filePath, np.random.rand(3, 5, 5))
            np.save(sample2.getSampleData().filePath, np.random.rand(3, 5, 5))

            viewPath: str = os.path.join(tmpdirname, "NewView.view")
            imgClf._saveView(viewPath)
            self.assertTrue(os.path.exists(viewPath))

            with open(viewPath, "rb") as fp:
                savedView: View = pickle.load(fp)

            self.assertEqual(len(savedView.samples), 2)
            self.assertEqual(savedView.samples[0], sample1._sampleData)
            self.assertEqual(savedView.samples[1], sample2._sampleData)

    def test_loadFromSample(self) -> None:
        imgClf: MainWindow = MainWindow()
        sample: Sample = Sample()
        sample.name = 'Sample1'
        sample.classes2Indices = {'Background': {1, 2, 3, 4},
                                  'class2': {5, 6, 7, 8}}
        testCube: np.ndarray = np.random.rand(10, 5, 5)
        sample.specObj.setCube(testCube)

        multiView: MultiSampleView = imgClf._multiSampleView
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
        self.assertTrue(np.array_equal(createdSample._graphOverlays._origCube, testCube))

        # make sure that these objects are identical
        self.assertTrue(createdSample._sampleData.specObj == sample.specObj)
        self.assertDictEqual(createdSample._sampleData.particleHandler.__dict__, sample.particleHandler.__dict__)
        # We now set them specObjs to None. These are at different memory locations and would make the assertDictEqual fail
        createdSample._sampleData.specObj = None
        sample.specObj = None
        createdSample._sampleData.particleHandler = None
        sample.particleHandler = None
        self.assertDictEqual(sample.__dict__, createdSample.getSampleData().__dict__)
        classCreator: ClassCreator = imgClf._clsCreator
        presentClasses: List[str] = [cls.name for cls in classCreator._classes]
        for cls in sample.classes2Indices.keys():
            self.assertTrue(cls in presentClasses)

    def test_thresholdSelector(self) -> None:
        avgImg: np.ndarray = np.random.rand(10, 10)
        avgImg = avgImg*255
        avgImg = avgImg.astype(np.uint8)
        threshSelector: ThresholdSelector = ThresholdSelector(avgImg)  # just make sure nothing fails

    def test_getWavelenghts(self) -> None:
        class MockSampleGood1:
            def getWavelengths(self) -> np.ndarray:
                return np.zeros(10)

            def getName(self) -> str:
                return "good sample"


        class MockSampleGood2:
            def getWavelengths(self) -> np.ndarray:
                return np.zeros(12)

            def getName(self) -> str:
                return "good sample"

        class MockSampleBad:
            def getWavelengths(self) -> np.ndarray:
                raise WavelengthsNotSetError

            def getName(self) -> str:
                return "bad sample"

        multiView: MultiSampleView = MultiSampleView(None)
        multiView._sampleviews = [MockSampleGood1(), MockSampleBad()]
        self.assertTrue(np.array_equal(multiView.getWavelengths(), MockSampleGood1().getWavelengths()))

        multiView._sampleviews = [MockSampleBad()]
        self.assertRaises(AssertionError, multiView.getWavelengths)  # raise because no wavelengths set

        multiView._sampleviews = [MockSampleGood1(), MockSampleGood2()]
        self.assertRaises(AssertionError, multiView.getWavelengths)  # raise because inconsistent wavelengths

    def test_adjustWavelenghts(self) -> None:
        class MockMainWin:
            def setupConnections(self, sampleView: 'SampleView') -> None:
                pass

        numWavelengthsShort, numWavelengthsLong = 10, 12

        multiView: MultiSampleView = MultiSampleView(MockMainWin())
        sampleShort: SampleView = multiView.addSampleView()
        sampleLong: SampleView = multiView.addSampleView()
        wavelenghtsShort: np.ndarray = np.arange(numWavelengthsShort)

        sampleLong.setCube(np.random.rand(numWavelengthsLong, 10, 10), np.arange(numWavelengthsLong))
        # now we add setup a cube that has a shorter wavelength axis
        sampleShort.setCube(np.random.rand(numWavelengthsShort, 10, 10), wavelenghtsShort)

        # we have to call that here manually, signals don't work here.
        multiView._assertIdenticalWavelengths()
        self.assertTrue(np.array_equal(wavelenghtsShort, sampleShort.getWavelengths()))
        self.assertTrue(np.array_equal(wavelenghtsShort, sampleLong.getWavelengths()))
