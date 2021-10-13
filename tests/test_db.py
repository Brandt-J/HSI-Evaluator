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
from typing import List, Dict
from unittest import TestCase
import mysql.connector
from PyQt5 import QtWidgets
import numpy as np
from multiprocessing import Queue

from database.database import DBConnection, uploadSpectra, SpecDetails
from gui.dbWin import ClassEntry


class TestDatabase(TestCase):
    def testConnection(self):
        conn: DBConnection = DBConnection()
        self.assertTrue(conn._connection is None)
        conn.connect()
        self.assertEqual(type(conn._connection), mysql.connector.connection.MySQLConnection)
        conn.disconnect()
        self.assertTrue(conn._connection is None)

    def testUpload(self):
        def getNumSpectraOfSample(conn: DBConnection, samplename: 'str') -> int:
            cursor = conn._getCursor()
            cursor.execute(f"SELECT sample FROM spectra WHERE sample='{samplename}'")
            sampleNames: List[str] = [row[0] for row in cursor]
            numSpectra = len(sampleNames)
            if numSpectra > 0:
                self.assertEqual(sampleNames, [samplename]*numSpectra)
            return numSpectra

        specLength: int = 10
        nameCls1, numCls1 = "TestClass1", 5
        nameCls2, numCls2 = "TestClass2", 4
        specDict: Dict[str, np.ndaray] = {nameCls1: np.random.rand(numCls1, specLength),
                                          nameCls2: np.random.rand(numCls2, specLength)}
        wavelengths: np.ndarray = np.arange(specLength)
        details = SpecDetails(10, 0.5, 5.3, "SWIR_LM", "pristine", "unknown", "SQLTestSample", "white")
        connection: DBConnection = DBConnection()

        self.assertTrue(details.sampleName not in connection.getSampleNames())
        # Create the sample entry
        connection.createNewSample(details.sampleName, details.sampleName)
        self.assertTrue(details.sampleName in connection.getSampleNames())

        # Upload the spectra
        self.assertEqual(getNumSpectraOfSample(connection, details.sampleName), 0)
        uploadSpectra(specDict, wavelengths, details, Queue())

        # reconnect, to get a fresh connection that sees all the uploaded spectra (sometimes failed otherwise..)
        connection.disconnect()
        connection.connect()
        self.assertEqual(getNumSpectraOfSample(connection, details.sampleName), numCls2+numCls1)
        wavelengthInd: int = connection._getIndexOfWavelenghtsWithoutUpload(wavelengths)
        self.assertTrue(wavelengthInd != -1)

        # Cleanup...
        cursor = connection._getCursor()
        cursor.execute(f"DELETE FROM spectra WHERE sample='{details.sampleName}'")
        cursor.execute(f"DELETE FROM samples WHERE sample_name='{details.sampleName}'")
        cursor.execute(f"DELETE FROM material_type WHERE material_name='{nameCls1}'")
        cursor.execute(f"DELETE FROM material_type WHERE material_name='{nameCls2}'")
        cursor.execute(f"DELETE FROM wavelengths WHERE id_wavelengths='{wavelengthInd}'")
        connection._connection.commit()

        self.assertTrue(details.sampleName not in connection.getSampleNames())
        self.assertEqual(getNumSpectraOfSample(connection, details.sampleName), 0)
        self.assertEqual(connection._getIndexOfWavelenghtsWithoutUpload(wavelengths), -1)
        connection.disconnect()

    def testUploadWavelengths(self) -> None:
        def removeWavelengthsFromDB(ind: int, conn: DBConnection) -> None:
            cursor = conn._getCursor()
            cursor.execute(f"DELETE FROM wavelengths WHERE id_wavelengths='{ind}'")
            connection._connection.commit()

        lenSpec1, lenSpec2 = 10, 12
        wavelengths1: np.ndarray = np.random.rand(lenSpec1)
        wavelengths2: np.ndarray = np.random.rand(lenSpec2)
        connection: DBConnection = DBConnection()
        # Assert that wavelengths are not yet in DB
        self.assertEqual(connection._getIndexOfWavelenghtsWithoutUpload(wavelengths1), -1)
        self.assertEqual(connection._getIndexOfWavelenghtsWithoutUpload(wavelengths2), -1)

        # Now upload and make sure the indices exist
        connection._uploadWavelengths(wavelengths1)
        connection._uploadWavelengths(wavelengths2)

        connection.disconnect()  # Get a fresh connectionl
        connection.connect()

        ind1: int = connection._getIndexOfWavelenghtsWithoutUpload(wavelengths1)
        ind2: int = connection._getIndexOfWavelenghtsWithoutUpload(wavelengths2)
        self.assertTrue(ind1 != -1)
        self.assertTrue(ind2 != -1)

        removeWavelengthsFromDB(ind1, connection)
        removeWavelengthsFromDB(ind2, connection)
        self.assertEqual(connection._getIndexOfWavelenghtsWithoutUpload(wavelengths1), -1)
        self.assertEqual(connection._getIndexOfWavelenghtsWithoutUpload(wavelengths2), -1)

        connection.disconnect()  # Get a fresh connectionl
        connection.connect()

        # Now the combination of both. Directly get an index that is not -1
        ind1 = connection._getIndexOfWavelengths(wavelengths1)
        ind2 = connection._getIndexOfWavelengths(wavelengths2)
        self.assertTrue(ind1 != -1)
        self.assertTrue(ind2 != -1)

        # final cleanup
        removeWavelengthsFromDB(ind1, connection)
        removeWavelengthsFromDB(ind2, connection)
        self.assertEqual(connection._getIndexOfWavelenghtsWithoutUpload(wavelengths1), -1)
        self.assertEqual(connection._getIndexOfWavelenghtsWithoutUpload(wavelengths2), -1)
        connection.disconnect()


class TestDBUI(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)

    def test_ClassEntry(self):
        sqlClasses: List[str] = ['class1', 'class2', 'class3']
        clsEntry: ClassEntry = ClassEntry("newClass", sqlClasses)

        self.assertEqual(len(clsEntry._typeCombo), len(sqlClasses))
        entriesInComboBox: List[str] = []
        for i in range(clsEntry._typeCombo.count()):
            entriesInComboBox.append(clsEntry._typeCombo.itemText(i))

        self.assertEqual(entriesInComboBox, sqlClasses)
        self.assertTrue(clsEntry.getTargetName() in sqlClasses)

        clsEntry._newNameRadioBtn.setChecked(True)
        self.assertRaises(AssertionError, clsEntry.getTargetName)  # No new name indicated..
        clsEntry._lineEdit.setText("newClass")
        self.assertEqual(clsEntry.getTargetName(), "newClass")

