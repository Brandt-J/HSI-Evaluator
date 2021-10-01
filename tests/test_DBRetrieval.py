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
GNU General Public License for more self.specDetails.

You should have received a copy of the GNU General Public License
along with this program, see COPYING.
If not, see <https://www.gnu.org/licenses/>.
"""

import sys
from PyQt5 import QtWidgets
from unittest import TestCase
import numpy as np
from typing import *
from multiprocessing import Queue

from database.database import uploadSpectra, SpecDetails
from gui.dbQueryWin import DatabaseQueryWindow, _convertSpecDictToCubeAndSelections, QueryGenerator
if TYPE_CHECKING:
    from database.database import DBConnection, DownloadedSpectrum


class TestSQLView(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)
        queryWin: DatabaseQueryWindow = DatabaseQueryWindow()

        # Upload some data
        cls.conn: 'DBConnection' = queryWin._dbConn
        testSampleName: str = "DBViewTestSample"
        cls.conn.createNewSample(testSampleName, "")

        specLength1, specLength2 = 10, 12
        cls.clsName1, cls.numCls1 = "DBViewTestClass1", 4
        cls.clsName2, cls.numCls2 = "DBViewTestClass2", 5
        cls.specDict1: Dict[str, np.ndaray] = {cls.clsName1: np.random.rand(cls.numCls1, specLength1)}
        cls.specDict2: Dict[str, np.ndaray] = {cls.clsName2: np.random.rand(cls.numCls2, specLength2)}
        cls.wavelengths1: np.ndarray = np.arange(specLength1)
        cls.wavelengths2: np.ndarray = np.arange(specLength2)

        cls.specDetails = SpecDetails(10, 0.5, 5.3, "SWIR_LM", "pristine", "unknown", testSampleName)
        uploadSpectra(cls.specDict1, cls.wavelengths1, cls.specDetails, Queue())
        uploadSpectra(cls.specDict2, cls.wavelengths2, cls.specDetails, Queue())

        cls.conn.disconnect()

    def testFetchSpectra(self) -> None:
        queryWin: DatabaseQueryWindow = DatabaseQueryWindow()
        self.conn: 'DBConnection' = queryWin._dbConn

        for clsName, wavelengths, numSpecs, specDict in zip([self.clsName1, self.clsName2],
                                                            [self.wavelengths1, self.wavelengths2],
                                                            [self.numCls1, self.numCls2],
                                                            [self.specDict1, self.specDict2]):

            queryWin._queryGen.getQuery = lambda: f"SELECT * FROM spectra WHERE assignment='{clsName}'"
            queryWin._fetch()
            downloadedSpecs: List['DownloadedSpectrum'] = queryWin._currentSpecs
            self.assertEqual(len(downloadedSpecs), numSpecs)
            self.assertTrue(_downloadedSpecsAreCorrect(downloadedSpecs, specDict[clsName]))
            for i in range(len(downloadedSpecs)):
                self.assertEqual(downloadedSpecs[i].className, clsName)
                self.assertTrue(np.array_equal(downloadedSpecs[i].wavelengths, wavelengths))

        self.conn.disconnect()

    def test_formatSpecDict(self):
        queryWin: DatabaseQueryWindow = DatabaseQueryWindow()
        self.conn: 'DBConnection' = queryWin._dbConn
        queryWin._queryGen.getQuery = lambda: f"SELECT * FROM spectra WHERE assignment='{self.clsName1}' OR assignment='{self.clsName2}'"
        queryWin._fetch()
        specDict, wavelenghts = queryWin._downloadedSpec2Dict()
        # wavelenghts1 is the shorter wavelength axis, so the spectra have to be mapped to that.
        assert len(self.wavelengths1) < len(self.wavelengths2)
        self.assertTrue(np.array_equal(self.wavelengths1, wavelenghts))

        self.assertEqual(len(specDict.keys()), 2)
        self.assertTrue(self.clsName1 in specDict.keys() and self.clsName2 in specDict.keys())
        self.assertEqual(specDict[self.clsName1].shape[0], self.numCls1)
        self.assertEqual(specDict[self.clsName2].shape[0], self.numCls2)

    def test_specDictToCube(self) -> None:
        wrongSpecDict: Dict[str, np.ndarray] = {"test1": np.random.rand(10, 8),
                                                "test2": np.random.rand(20, 7)}  # on purpose with different lengths of spectra
        self.assertRaises(AssertionError, _convertSpecDictToCubeAndSelections, (wrongSpecDict))

        numSpecsClass1, numSpecsClass2, numWavelengths = 10, 20, 8
        specDict: Dict[str, np.ndarray] = {"test1": np.random.rand(numSpecsClass1, numWavelengths),
                                           "test2": np.random.rand(numSpecsClass2, numWavelengths)}
        cube, classes2indices = _convertSpecDictToCubeAndSelections(specDict)
        cubeWidth: int = round((numSpecsClass1+numSpecsClass2)**0.5)
        cubeHeight: int = int(np.ceil((numSpecsClass1+numSpecsClass2) / cubeWidth))
        self.assertEqual(list(cube.shape), [numWavelengths, cubeHeight, cubeWidth])
        self.assertEqual(len(classes2indices), 2)
        self.assertEqual(len(classes2indices["test1"]), numSpecsClass1)
        self.assertEqual(len(classes2indices["test2"]), numSpecsClass2)

        i: int = 0
        for clsName, specs in specDict.items():
            for j in range(specs.shape[0]):
                y, x = np.unravel_index(i, (cubeHeight, cubeWidth))
                self.assertTrue(i in classes2indices[clsName])

                specInDict: np.ndarray = specs[j, :]
                specInCube: np.ndarray = cube[:, y, x]
                self.assertTrue(np.array_equal(specInCube, specInDict))

                i += 1

    @classmethod
    def tearDownClass(cls) -> None:
        cls.conn.connect()
        cursor = cls.conn._getCursor()
        wavelengthInd1: int = cls.conn._getIndexOfWavelenghtsWithoutUpload(cls.wavelengths1)
        wavelengthInd2: int = cls.conn._getIndexOfWavelenghtsWithoutUpload(cls.wavelengths2)
        cursor.execute(f"DELETE FROM spectra WHERE sample='{cls.specDetails.sampleName}'")
        cursor.execute(f"DELETE FROM samples WHERE sample_name='{cls.specDetails.sampleName}'")
        cursor.execute(f"DELETE FROM material_type WHERE material_name='{cls.clsName1}' OR material_name='{cls.clsName2}'")
        cursor.execute(f"DELETE FROM wavelengths WHERE id_wavelengths='{wavelengthInd1}' OR id_wavelengths='{wavelengthInd2}'")
        cls.conn._connection.commit()


class TestQueryGenerator(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)

    def test_queryGeneration(self) -> None:
        optnsDict: Dict[str, List[str]] = {"assignment": ["PE", "PMMA", "PET"],
                                           "particle_state": ["pristine", "weathered"],
                                           "size_class": ["unknown", "10-100", "100-500"]}
        queryGen: QueryGenerator = QueryGenerator(optnsDict)
        self.assertEqual(queryGen._optnCheckBoxes.keys(), optnsDict.keys())

        query: str = queryGen.getQuery()
        self.assertEqual(query, "NOTHING_SELECTED")  # in that case, everything

        for key in optnsDict.keys():
            self.assertEqual(len(optnsDict[key]), len(queryGen._optnCheckBoxes[key]))

        for checkbox in queryGen._optnCheckBoxes["assignment"]:
            if checkbox.text() in ["PE", "PMMA"]:
                checkbox.setChecked(True)

        for checkbox in queryGen._optnCheckBoxes["particle_state"]:
            if checkbox.text() in ["pristine"]:
                checkbox.setChecked(True)

        query: str = queryGen.getQuery()
        self.assertEqual(query,
                         "SELECT * FROM spectra WHERE (assignment='PE' OR assignment='PMMA') AND (particle_state='pristine')")


def _downloadedSpecsAreCorrect(downloadedSpecs: List['DownloadedSpectrum'], specArr: np.ndarray) -> bool:
    """
    Checks if a list of downloaded spectra contains the intensity data as indicated in the spec Array. The list can
    be out of order, hence we use that function here.
    :param downloadedSpecs: List of downloaded Spectra
    :param specArr: (NxM) array of N spectra with M wavelenghts
    """
    dataIsCorrect: bool = False
    if len(downloadedSpecs) == specArr.shape[0]:
        refSpecs: List[np.ndarray] = [specArr[i, :] for i in range(specArr.shape[0])]
        specsFound: int = 0
        for spec in downloadedSpecs:
            for i in range(len(refSpecs)):
                if np.array_equal(spec.intensities, refSpecs[i]):
                    specsFound += 1
                    del refSpecs[i]
                    break
        if specsFound == len(downloadedSpecs) and len(refSpecs) == 0:
            dataIsCorrect = True

    return dataIsCorrect

