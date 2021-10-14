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
from copy import copy

from PyQt5 import QtWidgets, QtGui, QtCore
from typing import *
import numpy as np

from database.database import DBConnection, DownloadedSpectrum
if TYPE_CHECKING:
    from gui.sampleview import SampleView


class DatabaseQueryWindow(QtWidgets.QWidget):
    QueryFinished: QtCore.pyqtSignal = QtCore.pyqtSignal()
    AcceptResult: QtCore.pyqtSignal = QtCore.pyqtSignal(np.ndarray, np.ndarray, dict)  # cube, wavelenths, classes2ind

    def __init__(self):
        super(DatabaseQueryWindow, self).__init__()
        self.setWindowTitle("SQL Query Window")
        self._sampleView: Union[None, 'SampleView'] = None
        self._dbConn: DBConnection = DBConnection()
        self._currentSpecs: List[DownloadedSpectrum] = []

        self._queryGen: 'QueryGenerator' = QueryGenerator(self._getOptionsDict())

        self._checkComplex: QtWidgets.QCheckBox = QtWidgets.QCheckBox("Complex names")
        self._checkComplex.setChecked(False)
        self._checkGroupSediment: QtWidgets.QCheckBox = QtWidgets.QCheckBox("Group Sediments")
        self._checkGroupSediment.setChecked(True)

        self._btnFetch: QtWidgets.QPushButton = QtWidgets.QPushButton("Fetch")
        self._btnFetch.released.connect(self._fetch)
        self._lblFetchResult: QtWidgets.QLabel = QtWidgets.QLabel("No spectra fetched.")
        self._btnAccept: QtWidgets.QPushButton = QtWidgets.QPushButton("Accept")
        self._btnCancel: QtWidgets.QPushButton = QtWidgets.QPushButton("Cancel")
        self._btnAccept.released.connect(self._accept)
        self._btnCancel.released.connect(self.close)

        self._configureWidgets()
        self._createLayout()

    def setSampleView(self, sampleview: 'SampleView') -> None:
        """
        Sets sampleview reference.
        """
        self._sampleView = sampleview

    def _accept(self) -> None:
        """
        Accepts the currently retrieved data and reformats it into a spectra cube and the class selection.
        """
        if len(self._currentSpecs) > 0:
            specDict, wavelengths = self._downloadedSpec2Dict()
            cube, classes2Ind = _convertSpecDictToCubeAndSelections(specDict)
            self.AcceptResult.emit(cube, wavelengths, classes2Ind)
            self.close()
        else:
            QtWidgets.QMessageBox.about(self, "Info", "No spectra were fethched..")

    def _fetch(self) -> None:
        """
        Fetches spectra from the database accoring the current query and stores results into temporary spectra dict.
        """
        statement: str = self._queryGen.getQuery()
        if statement == "NOTHING_SELECTED":
            self._lblFetchResult.setText("Nothing selected")
        else:
            try:
                self._currentSpecs = self._dbConn.fetchSpectraWithStatement(statement)
            except Exception as e:
                self._currentSpecs = []
                self._lblFetchResult.setText(f"Fetching failed with error: {e}")
            else:
                types: np.ndarray = np.unique([spec.className for spec in self._currentSpecs])
                self._lblFetchResult.setText(f"Fetched {len(self._currentSpecs)} spectra of {len(types)} classes")

    def _configureWidgets(self) -> None:
        """
        Configures all used widgets.
        """
        for btn in [self._btnFetch, self._btnCancel, self._btnAccept]:
            btn.setFixedWidth(75)

    def _createLayout(self) -> None:
        """
        Creates the layout of the widget.
        """
        lblInfo: QtWidgets.QLabel = QtWidgets.QLabel("Define a selection of spectra to retrieve:")
        fontBold: QtGui.QFont = QtGui.QFont()
        fontBold.setBold(True)
        lblInfo.setFont(fontBold)
        lblInfo2: QtWidgets.QLabel = QtWidgets.QLabel("(If no entries are selected in a group, it's contents are ignored)")

        modClsNameGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox("Modify class names")
        modClsNameLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        modClsNameGroup.setLayout(modClsNameLayout)
        modClsNameLayout.addWidget(self._checkComplex)
        modClsNameLayout.addWidget(self._checkGroupSediment)
        modClsNameLayout.addStretch()

        fetchLayout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        fetchLayout.addWidget(self._btnFetch)
        fetchLayout.addWidget(self._lblFetchResult)
        fetchLayout.addStretch()

        btnLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        btnLayout.addWidget(self._btnAccept)
        btnLayout.addWidget(self._btnCancel)
        btnLayout.addStretch()

        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        layout.addWidget(lblInfo)
        layout.addWidget(lblInfo2)
        layout.addStretch()
        layout.addWidget(QtWidgets.QLabel(""))
        layout.addWidget(self._queryGen)
        layout.addWidget(modClsNameGroup)
        layout.addWidget(QtWidgets.QLabel(""))
        layout.addLayout(fetchLayout)
        layout.addStretch()
        layout.addLayout(btnLayout)
        self.setLayout(layout)

    def _downloadedSpec2Dict(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Reformats the currntly stored list of downloaded spectra into a dictionary with class-name as keys and
        intensity array as values. Also a unique wavelength axis is returned; all spectra are mapped to the shortes
        wavelength axis.
        :return Tuple[named SpectraDict, wavelenght array]
        """
        assert len(self._currentSpecs) > 0
        specsToProcess: List['DownloadedSpectrum'] = self._currentSpecs.copy()
        lenOfWavelengths: List[int] = [len(spec.wavelengths) for spec in specsToProcess]
        minInd: int = int(np.argmin(lenOfWavelengths))
        shortestWavelengths: np.ndarray = self._currentSpecs[minInd].wavelengths

        specDict: Dict[str, np.ndarray] = {}
        for spec in specsToProcess:
            remappedSpec: np.ndarray = spec.getIntensitiesForOtherWavelengths(shortestWavelengths)
            if self._checkGroupSediment.isChecked():
                spec.groupSedimentName()
            className: str = spec.getConcatenatedName() if self._checkComplex.isChecked() else spec.className
            if className not in specDict.keys():
                specDict[className] = remappedSpec
            else:
                presentSpectra: np.ndarray = specDict[className]
                specDict[className] = np.vstack((remappedSpec, presentSpectra))

        return specDict, shortestWavelengths

    def _getOptionsDict(self) -> Dict[str, List[str]]:
        """
        Retrieves an options dictionary from the SQL database.
        """
        optnDict: Dict[str, List[str]] = {"assignment": self._dbConn.getClassNames(),
                                          "sample": self._dbConn.getSampleNames(),
                                          "particle_state": self._dbConn.getParticleStates(),
                                          "size_class": self._dbConn.getParticleSizes(),
                                          "color": self._dbConn.getColors()}
        return optnDict

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.QueryFinished.emit()
        a0.accept()


class QueryGenerator(QtWidgets.QGroupBox):
    def __init__(self, optnsDict: Dict[str, List[str]]) -> None:
        """
        A widget for creating an SQL query. Takes a dictionary of options that are used for generating the query.
        Every key is a group of options, with the individual possible entries defined in the according value-Lists.
        """
        super(QueryGenerator, self).__init__()
        self._layout: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        self.setLayout(self._layout)

        self._fontBold: QtGui.QFont = QtGui.QFont()
        self._fontBold.setBold(True)
        self._optnCheckBoxes: Dict[str, List[QtWidgets.QCheckBox]] = {}

        self._queryPrefix: str = "SELECT * FROM spectra WHERE "
        assert len(optnsDict) > 0
        self._createLayoutAndCheckboxes(optnsDict)

    def _createLayoutAndCheckboxes(self, optionsDict: Dict[str, List[str]]) -> None:
        maxRowIndex: int = 0
        andColIndices: List[int] = []
        for columnIndex, (groupName, options) in enumerate(optionsDict.items()):
            headerLabel: QtWidgets.QLabel = QtWidgets.QLabel(groupName)
            headerLabel.setFont(self._fontBold)

            self._layout.addWidget(headerLabel, 0, 2*columnIndex, QtCore.Qt.AlignCenter)
            self._optnCheckBoxes[groupName] = []
            for rowIndex, optnName in enumerate(options, start=1):
                newCheckBox: QtWidgets.QCheckBox = QtWidgets.QCheckBox(optnName)
                self._optnCheckBoxes[groupName].append(newCheckBox)
                self._layout.addWidget(newCheckBox, rowIndex, 2*columnIndex, QtCore.Qt.AlignLeft)

            maxRowIndex = max([rowIndex, maxRowIndex])
            if columnIndex < len(optionsDict) - 1:
                andColIndices.append(2*columnIndex + 1)

        for columnIndex in andColIndices:
            self._layout.addWidget(QtWidgets.QLabel("AND"), 1, columnIndex, maxRowIndex-1, 1)

    def getQuery(self) -> str:
        """
        Returns the currently defined query.
        """
        checkedOptions: Dict[str, List[str]] = {}
        if len(self._optnCheckBoxes) > 0:
            for groupName, entries in self._optnCheckBoxes.items():
                checkedEntries: List[str] = [box.text() for box in entries if box.isChecked()]
                if len(checkedEntries) > 0:
                    checkedOptions[groupName] = checkedEntries
        if len(checkedOptions) > 0:
            query = copy(self._queryPrefix)
            for i, (groupName, options) in enumerate(checkedOptions.items()):
                query += "("
                for j, entry in enumerate(options):
                    query += groupName + f"='{entry}'"
                    if j < len(options)-1:
                        query += " OR "
                query += ")"

                if i < len(checkedOptions)-1:
                    query += " AND "
        else:
            query = "NOTHING_SELECTED"

        return query


def _convertSpecDictToCubeAndSelections(specDict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Set[int]]]:
    """
    Takes a spectra dictionary and converts it into a spec "cube" and the according class selections.
    All arrays in the dictionary have to have the same axis-1 length (i.e., num wavelengths).
    :param specDict: The spectra dict to convert (keys: class names, values: (NxM) arrays of N spectra with M wavelengths
    :return Tuple: KxMxN cube of MxN spectra with K wavelengths, classes2Indices dictonary
    """
    numWavenums: Set[int] = set([specArr.shape[1] for specArr in specDict.values()])
    assert len(numWavenums) == 1, 'The given spectra have multiple number of wavelengths.'
    numWavenums: int = list(numWavenums)[0]

    allSpecs: Union[None, np.ndarray] = None
    allClasses: List[str] = []
    for name, spectra in specDict.items():
        numSpecs: int = spectra.shape[0]
        allClasses += [name]*numSpecs
        if allSpecs is None:
            allSpecs = spectra
        else:
            allSpecs = np.vstack((allSpecs, spectra))

    assert allSpecs.shape[0] == len(allClasses)
    numSpecsTotal: int = len(allClasses)
    cubeWidth: int = int(round(numSpecsTotal**0.5))
    cubeHeight: int = int(np.ceil(numSpecsTotal / cubeWidth))

    classes2Ind: Dict[str, Set[int]] = {}
    cube: np.ndarray = np.zeros((numWavenums, cubeHeight, cubeWidth))

    i: int = 0
    for clsName, specArr in specDict.items():
        for j in range(specArr.shape[0]):
            if clsName not in classes2Ind.keys():
                classes2Ind[clsName] = {i}
            else:
                classes2Ind[clsName].add(i)

            y, x = np.unravel_index(i, (cubeHeight, cubeWidth))
            cube[:, y, x] = specArr[j, :]
            i += 1

    return cube, classes2Ind
    

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    queryWin: DatabaseQueryWindow = DatabaseQueryWindow()
    queryWin.show()
    app.exec_()
