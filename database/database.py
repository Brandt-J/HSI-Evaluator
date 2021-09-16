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
import json
import os
from io import StringIO
from dataclasses import dataclass
import mysql.connector
from typing import Union, TYPE_CHECKING, List, Dict

import numpy as np
from readConfig import sqlLogin
from projectPaths import getAppFolder
from logger import getLogger

if TYPE_CHECKING:
    from logging import Logger
    from multiprocessing import Queue
    from gui.dbWin import SpecDetails


class DBConnection:
    def __init__(self):
        self._connection: Union[None, mysql.connector.connection.MySQLConnection] = None
        self._logger: 'Logger' = getLogger("SQL Connection")

    def connect(self) -> None:
        """
        Establishes a connection to the database. Raises a ConnectionError if any errors occur.
        """
        if self._connection is None:
            try:
                self._connection = mysql.connector.connect(**sqlLogin)
                self._logger.info("Successfully connected to database.")
            except mysql.connector.errors.ProgrammingError as e:
                self._logger.critical(f"Connection error: {e}")
                raise ConnectionError(e)

    def getClassNames(self) -> List[str]:
        """
        Returns a list of the material class names that are currently present in the database.
        """
        cursor = self._getCursor()
        cursor.execute("SELECT material_name FROM material_type")
        classes: List[str] = [row[0] for row in cursor]
        return classes

    def getSampleNames(self) -> List[str]:
        """
        Returns a list of the sample names that are currently present in the database.
        """
        cursor = self._getCursor()
        cursor.execute("SELECT sample_name FROM samples")
        classes: List[str] = [row[0] for row in cursor]
        return classes

    def getMaterialTypes(self) -> List[str]:
        """
        Returns a list of available material types.
        """
        cursor = self._getCursor()
        cursor.execute("SELECT spectrum_type FROM spec_type")
        types: List[str] = [row[0] for row in cursor]
        return types

    def getCommentOfSample(self, sampleName: str) -> str:
        """
        Returns the comment associated to the specified sample.
        """
        cursor = self._getCursor()
        cursor.execute("SELECT * FROM samples")
        sampleComment: Union[None, str] = None
        for row in cursor:
            name, comment = row[1], row[2]
            if name == sampleName:
                sampleComment = comment
                break
        if sampleComment is None:
            self._logger.critical(f"Sample '{sampleName}' was not found in Database.")
        assert sampleComment is not None, f"Sample {sampleName} was not found in Database."
        return sampleComment

    def createNewSample(self, sampleName: str, commentString: str) -> None:
        """
        Creates a new sample with the given information.
        :param sampleName: Name of the sample
        :param commentString: Comment to add to the sample in the database.
        """
        cursor = self._getCursor()
        cursor.execute(f"""INSERT INTO samples (sample_name, COMMENT)  VALUES ("{sampleName}", "{commentString}")""")
        self._connection.commit()
        self._logger.info(f"Created sample '{sampleName}' with comment '{commentString}' in SQL Database")

    def assertClassNameisPresent(self, className: str) -> None:
        """
        Makes sure that the given classname is present in the SQL database. If not, it will be added.
        """
        if className not in self._getMaterialTypes():
            self._createNewMaterialType(className)

    def uploadSpectrum(self, clsname: str, intensities: np.ndarray, wavelengths: np.ndarray,
                       spectraDetail: 'SpecDetails', directcommit: bool = True) -> None:
        """
        Uploads the given spectrum to the database.
        :param clsname: Material type name for the spectrum
        :param intensities: shape N np.array of intensities
        :param wavelengths: shape N np.array of corrensponding wavelenghts
        :param spectraDetail: SpecDetail object carrying all other details.
        :param directcommit: If False, the commit has to be done at a later stage, makes upload faster. If True, the commit
        will done directly after uploading the spectrum.
        """
        cursor = self._getCursor()
        specstring: str = specToString(wavelengths, intensities)
        sqlcommand = f"""INSERT INTO spectra (spec_type, assignment, specdata, num_accumulations, acquisition_time, pxScale, sample) 
                                        VALUES ("{spectraDetail.specType}", 
                                        "{clsname}", "{specstring}", "{spectraDetail.numAcc}", "{spectraDetail.accTime}",
                                        "{spectraDetail.resolution}", "{spectraDetail.sampleName}");"""
        cursor.execute(sqlcommand)
        if directcommit:
            self._connection.commit()

    def commit(self) -> None:
        """
        Commits the connection to actually update the remote database!
        """
        self._connection.commit()

    def disconnect(self) -> None:
        """
        Closes the connection to the SQL Database.
        """
        if self._connection is not None:
            self._connection.disconnect()
        self._connection = None
        self._logger.info("Disconnected from database.")

    def _getCursor(self):
        self._assertConnection()
        try:
            cursor = self._connection.cursor(buffered=True)  #
        except mysql.connector.errors.InternalError as e:
            self._logger.critical(f"Cursor to SQL database could not be retrieved:\n{e}")
            breakpoint()
            raise ConnectionError(f"Cursor to SQL database could not be retrieved:\n{e}")
        return cursor

    def _assertConnection(self) -> None:
        """
        Used for asserting that a valid connection is present.
        """
        if self._connection is None:
            self.connect()  # Will raise exception if not possible

    def _getMaterialTypes(self) -> List[str]:
        """
        Returns a list of all currently present material types in the database
        """
        cursor = self._getCursor()
        cursor.execute("SELECT material_name FROM material_type")
        return [row[0] for row in cursor]

    def _createNewMaterialType(self, typename: str) -> None:
        """
        Adds the given typename to the database.
        """
        cursor = self._getCursor()
        cursor.execute(f"""INSERT INTO material_type (material_name)  VALUES ("{typename}")""")
        self._connection.commit()
        self._logger.info(f"Created material type '{typename}' in SQL Database")


@dataclass
class SpecDetails:
    numAcc: int
    accTime: float
    resolution: float
    specType: str
    sampleName: str = ""


def uploadSpectra(spectraDict: Dict[str, np.ndarray], wavelengths: np.ndarray, spectraDetail: 'SpecDetails',
                  dataqueue: 'Queue') -> None:
    """
    Process to upload the spectra in the spectraDict to the SQL database, given the current spectra Details.
    :param spectraDict: Dictionary of spectra. Key: Classname in SQL DB, value: (NxM) array of N spectra with M wavelengths
    :param wavelengths: np.array (shape M) with the according wavelengths
    :param spectraDetail: SpecDetailStruct containing relevant meta-data
    :param dataqueue: DataQueue to use for updating upload status
    """
    logger: 'Logger' = getLogger("SQL Upload")
    conn: DBConnection = DBConnection()
    spectraUploaded: int = 0
    for clsname, spectra in spectraDict.items():
        logger.info(f"Uploading {len(spectra)} spectra for class '{clsname}'")
        conn.assertClassNameisPresent(clsname)
        for i in range(spectra.shape[0]):
            conn.uploadSpectrum(clsname, spectra[i, :], wavelengths, spectraDetail, directcommit=False)

            spectraUploaded += 1
            dataqueue.put(spectraUploaded)

    conn.commit()
    conn.disconnect()


def specToString(wavelengths: np.ndarray, intensities: np.ndarray) -> str:
    """
    Puts wavelengths and intensities into (Nx2) (N wavelengths, wavelengths in first column) array
    and converts into a bytestring.
    """
    assert len(wavelengths) == len(intensities)
    spectrum: np.ndarray = np.vstack((wavelengths, intensities)).transpose()
    fp = StringIO()
    np.savetxt(fp, spectrum)
    fp.seek(0)
    return fp.read()
