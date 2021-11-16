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
from io import StringIO
from dataclasses import dataclass
import mysql.connector
from typing import Union, TYPE_CHECKING, List, Dict

import numpy as np
from readConfig import sqlLogin
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

    def getParticleStates(self) -> List[str]:
        """
        Returns a list of particle states.
        """
        cursor = self._getCursor()
        cursor.execute("SELECT particle_state FROM particle_states")
        return [row[0] for row in cursor]

    def getParticleSizes(self) -> List[str]:
        """
        Returns a list of particle size classes
        """
        cursor = self._getCursor()
        cursor.execute("SELECT size_class FROM size_classes")
        return [row[0] for row in cursor]

    def getColors(self) -> List[str]:
        """
        Returns a list of colors
        """
        cursor = self._getCursor()
        cursor.execute("SELECT color_name FROM colors")
        return [row[0] for row in cursor]

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

    def fetchSpectraWithStatement(self, sqlStatement: str) -> List['DownloadedSpectrum']:
        cursor = self._getCursor()
        try:
            cursor.execute(sqlStatement)
        except Exception as e:
            self._logger.warning(f"Error on fetching with statement: {sqlStatement}:\n{e}")
            raise e
        else:  # only, when no exceptoin occurred
            wavelenghts: Dict[int, np.ndarray] = self._getWavelenghtAxes()
            spectra: List[DownloadedSpectrum] = []
            for row in cursor:
                intens, wavel = arrFromBytes(row[3]), wavelenghts[row[4]]
                assert len(intens) == len(wavel), f"Length spectrum ({len(intens)}) does not match length wavelengths ({len(wavel)})"
                spectra.append(DownloadedSpectrum(className=row[2],
                                                  intensities=intens,
                                                  wavelengths=wavel,
                                                  sample=row[8],
                                                  state=row[9],
                                                  size=row[10],
                                                  color=row[11]))
            return spectra

    def createNewSample(self, sampleName: str, commentString: str) -> None:
        """
        Creates a new sample with the given information.
        :param sampleName: Name of the sample
        :param commentString: Comment to add to the sample in the database.
        """
        if sampleName not in self.getSampleNames():
            cursor = self._getCursor()
            cursor.execute(f"""INSERT INTO samples (sample_name, COMMENT)  VALUES ("{sampleName}", "{commentString}")""")
            self._connection.commit()
            self._logger.info(f"Created sample '{sampleName}' with comment '{commentString}' in SQL Database")
        else:
            self._logger.warning(f"Did not create sample name {sampleName}, as it already exist in the database!")

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
        specstring: str = arrToString(intensities)
        wavelengthInd: int = self._getIndexOfWavelengths(wavelengths)
        assert wavelengthInd != -1, 'Failed finding wavelengths in database!'
        sqlcommand = f"""INSERT INTO spectra (spec_type, assignment, intensities, wavelengths, num_accumulations, acquisition_time, pxScale, sample, particle_state, size_class, color) 
                                        VALUES ("{spectraDetail.specType}", 
                                        "{clsname}", "{specstring}", "{wavelengthInd}", "{spectraDetail.numAcc}", "{spectraDetail.accTime}",
                                        "{spectraDetail.resolution}", "{spectraDetail.sampleName}", "{spectraDetail.particleState}", 
                                        "{spectraDetail.sizeClass}", "{spectraDetail.color}");"""
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

    def _getIndexOfWavelengths(self, wavelenghts: np.ndarray) -> int:
        """
        Takes a wavelength array and returns the unique index of it. If the array is not yet present in the DB, it will
        be uploaded.
        """
        ind: int = self._getIndexOfWavelenghtsWithoutUpload(wavelenghts)
        if ind == -1:  # i.e., not yet in db..
            self._uploadWavelengths(wavelenghts)
        ind = self._getIndexOfWavelenghtsWithoutUpload(wavelenghts)
        assert ind != -1
        return ind

    def _getIndexOfWavelenghtsWithoutUpload(self, wavelengths: np.ndarray) -> int:
        """
        Returns index of the wavelength array in the db. Returns -1 if wavelengths not yet in DB.
        """
        index: int = -1
        uploadedWavelengths: Dict[int, np.ndarray] = self._getWavelenghtAxes()
        for ind, arr in uploadedWavelengths.items():
            if np.array_equal(wavelengths, arr):
                index = ind
                break
        return index

    def _uploadWavelengths(self, wavelengths: np.ndarray) -> None:
        """
        Takes the wavelenghts array and uploads it to the DB.
        """
        cursor = self._getCursor()
        wavelstr: str = arrToString(wavelengths)
        cursor.execute(f"""INSERT into wavelengths (wavelengths) VALUES ("{wavelstr}")""")
        self._connection.commit()

    def _getWavelenghtAxes(self) -> Dict[int, np.ndarray]:
        """
        Returns a list dictionary all the wavelength axes present in the 'wavelengths' table of the db.
        Key: Unique index, Value: np.ndarray of wavelenght axis.
        """
        cursor = self._getCursor()
        cursor.execute("SELECT * FROM wavelengths")
        wavelengths: Dict[int, np.ndarray] = {}
        for row in cursor:
            index, data = row[0], arrFromBytes(row[1])
            wavelengths[index] = data
        return wavelengths


@dataclass
class SpecDetails:
    numAcc: int
    accTime: float
    resolution: float
    specType: str
    particleState: str = ""
    sizeClass: str = ""
    sampleName: str = ""
    color: str = ""


@dataclass
class DownloadedSpectrum:
    className: str
    wavelengths: np.ndarray
    intensities: np.ndarray
    sample: str
    state: str
    size: str
    color: str

    def getIntensitiesForOtherWavelengths(self, otherWavelengths: np.ndarray) -> np.ndarray:
        """
        Takes a new wavelength axis and returns an intensities array fitting this other wavelengths.
        """
        if np.array_equal(otherWavelengths, self.wavelengths):
            newIntensities: np.ndarray = self.intensities.copy()
        else:
            newIntensities: np.ndarray = np.zeros_like(otherWavelengths)
            for i in range(len(otherWavelengths)):
                closestInd: int = int(np.argmin(np.abs(self.wavelengths - otherWavelengths[i])))
                newIntensities[i] = self.intensities[closestInd]

        return newIntensities
    
    def groupSedimentName(self) -> None:
        """
        Convenience function for grouping sediment names. If the classname seems to be a sediment, then it will be
        renamed into just "sediment".
        """
        if self.className.lower().find("sediment") != -1:
            self.className = "Sediment"

    def abbreviatePolymer(self) -> None:
        """
        Convenience function for abbreviating polymer names.
        """
        self.className = self.className.replace("Polyethylene", "PE")
        self.className = self.className.replace("Polystyrene", "PS")
        self.className = self.className.replace("Polyurethane", "PUR")
        self.className = self.className.replace("Poly(ethylene terephthalate)", "PET")
        self.className = self.className.replace("Poly(methyl methacrylate)", "PMMA")
        self.className = self.className.replace("Poly(vinyl chloride)", "PVC")
        self.className = self.className.replace("Polyamide", "PA")
        self.className = self.className.replace("Polycarbonate", "PC")
        self.className = self.className.replace("Acrylonitrile butadiene styrene", "ABS")
        self.className = self.className.replace("Polypropylene", "PP")

    def getConcatenatedName(self) -> str:
        """
        Returns a name concatenating different properties
        """
        return '_'.join([self.className, self.state, self.color])
    

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
    assert spectraDetail.particleState in conn.getParticleStates(), f'{spectraDetail.particleState} was not yet uploaded to DB'
    assert spectraDetail.sizeClass in conn.getParticleSizes(), f'{spectraDetail.sizeClass} was not yet uploaded to DB'

    for clsname, spectra in spectraDict.items():
        logger.info(f"Uploading {len(spectra)} spectra for class '{clsname}'")
        conn.assertClassNameisPresent(clsname)
        for i in range(spectra.shape[0]):
            conn.uploadSpectrum(clsname, spectra[i, :], wavelengths, spectraDetail, directcommit=False)

            spectraUploaded += 1
            dataqueue.put(spectraUploaded)

    conn.commit()
    conn.disconnect()


def arrToString(array: np.ndarray) -> str:
    """
    Converts an np.ndarray into a bytestring.
    """
    arr = array[:, np.newaxis]
    fp = StringIO()
    np.savetxt(fp, arr)
    fp.seek(0)
    return fp.read()


def arrFromBytes(bytesObj: bytes) -> np.ndarray:
    """
    Takes a byte object as read from the SQL database and formats it into a numeric numpy array.
    """
    string: str = str(bytesObj)[2:]
    values: List[str] = string.split("\\n")[:-1]
    assert len(values) > 0, f'Invlaid conversion of bytes to value-list, please check! BytesObject is: {bytesObj}'
    values: List[float] = [float(val) for val in values]
    return np.array(values)
    