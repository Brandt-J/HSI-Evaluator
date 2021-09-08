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
from typing import List, Union, TYPE_CHECKING, Dict
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from multiprocessing import Process, Queue
import difflib

from database.database import DBConnection, uploadSpectra, SpecDetails
from logger import getLogger

if TYPE_CHECKING:
    from gui.sampleview import SampleView
    from logging import Logger


class DBUploadWin(QtWidgets.QWidget):
    UploadFinished: QtCore.pyqtSignal = QtCore.pyqtSignal()

    def __init__(self):
        super(DBUploadWin, self).__init__()
        self.setWindowTitle("Upload to SQL")
        self._conn: DBConnection = DBConnection()

        self._logger: 'Logger' = getLogger("SQL-DB-GUI")
        self._sampleview: Union[None, 'SampleView'] = None
        self._sampleEntry: Union[None, SampleEntry] = None
        self._detailsEntry: Union[None, 'SpecDetailEntry'] = None

        self._uploadBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("Upload")
        self._uploadBtn.setFixedWidth(150)
        self._uploadBtn.released.connect(self._uploadData)

        self._progressWindow: ProgressWindow = ProgressWindow()
        self._timer: QtCore.QTimer = QtCore.QTimer()
        self._timer.setSingleShot(False)
        self._timer.timeout.connect(self._checkOnUpload)
        self._process: Process = Process()
        self._queue: Queue = Queue()

        self._classEntries: List['ClassEntry'] = []
        self._layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

    def setSampleView(self, sampleView: 'SampleView') -> None:
        self._sampleview = sampleView

    def recreateLayout(self) -> None:
        try:
            self._conn.connect()
            sqlClassNames: List[str] = self._conn.getClassNames()
            specTypes: List[str] = self._conn.getMaterialTypes()
        except ConnectionError as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not connect to SQL Database:\n{e}")
            self.close()
            return

        self._clearLayoutAndWidgets()

        self._sampleEntry = SampleEntry(self._conn)
        self._detailsEntry = SpecDetailEntry(specTypes)

        clsScrollArea: QtWidgets.QScrollArea = QtWidgets.QScrollArea()
        clsGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox("Select Classes to upload")
        clsGroup.setLayout(QtWidgets.QVBoxLayout())
        if self._sampleview is not None:
            classes: Dict[str, np.ndarray] = self._sampleview.getAllLabelledSpectra()
            for cls in classes.keys():
                newClsEntry: ClassEntry = ClassEntry(cls, sqlClassNames)
                self._classEntries.append(newClsEntry)
                clsGroup.layout().addWidget(newClsEntry)

        clsScrollArea.setWidget(clsGroup)

        self._layout.addWidget(self._sampleEntry)
        self._layout.addWidget(self._detailsEntry)
        self._layout.addWidget(clsScrollArea)
        self._layout.addWidget(self._uploadBtn)
        self._conn.disconnect()

    def _uploadData(self) -> None:
        """
        Prompts for uploading the data and does so, if confirmed.
        """
        ret = QtWidgets.QMessageBox.question(self, "Continue", "Do you want to upload the selection?",
                                             QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                             QtWidgets.QMessageBox.Yes)
        if ret == QtWidgets.QMessageBox.Yes:
            self._logger.info("Starting upload.")
            try:
                self._conn.connect()
            except ConnectionError as e:
                self._logger.critical(f"Error on connecting to database: {e}")
                QtWidgets.QMessageBox.critical(self, "Error", f"Error on connecting to database:\n{e}")
                return

            sampleName: str = self._sampleEntry.getSampleName()
            self._assertSampleInDB(sampleName)
            specdetails: SpecDetails = self._detailsEntry.getDetails()
            specdetails.sampleName = sampleName

            specDict: Dict[str, np.ndarray] = self._getSpecsToUpload()
            wavelengths: np.ndarray = self._sampleview.getWavelengths()
            numSpecs: int = int(sum([len(specSet) for specSet in specDict.values()]))
            self._progressWindow.setupToSample(sampleName, numSpecs)
            self._progressWindow.show()
            self._timer.start(10)

            self._process = Process(target=uploadSpectra, args=(specDict, wavelengths, specdetails, self._queue))
            self._process.start()
            self.setDisabled(True)

    def _checkOnUpload(self) -> None:
        while not self._queue.empty():
            i = self._queue.get()  # counter of finished samples
            self._progressWindow.setValue(i)
            if self._progressWindow.isFinished():
                self._finishUpload()
                break

    def _finishUpload(self) -> None:
        self._progressWindow.hide()
        self._process.join()
        self._queue.close()
        self._timer.stop()
        self.setEnabled(True)
        QtWidgets.QMessageBox.about(self, "Upload Done.", "Finished spectra upload without errors.")

    def _assertSampleInDB(self, samplename: str) -> None:
        """
        Makes sure that the given samplename is present in the SQL Database. If it's not yet there, a new entry will
        be created.
        :param samplename: Name of the sample
        """
        presentnames: List[str] = self._conn.getSampleNames()
        if samplename not in presentnames:
            self._conn.createNewSample(samplename, self._sampleEntry.getNewSampleComment())

    def _getSpecsToUpload(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary of spectra to upload. Keys are class names as to be used in database.
        """
        allSpectra: Dict[str, np.ndarray] = self._sampleview.getAllLabelledSpectra()
        specsToUpload: Dict[str, np.ndarray] = {}
        for clsEntry in self._classEntries:
            if clsEntry.isSelected():
                origname, targetName = clsEntry.getOrigignalName(), clsEntry.getTargetName()
                specsToUpload[targetName] = allSpectra[origname]
        return specsToUpload

    def _clearLayoutAndWidgets(self) -> None:
        """
        Clears the layout content and the respective widgets.
        """
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if item is not None:
                self._layout.removeWidget(item.widget())

        self._classEntries = []
        self._sampleEntry = None
        self._detailsEntry = None

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.UploadFinished.emit()
        self._conn.disconnect()
        event.accept()


class ClassEntry(QtWidgets.QGroupBox):
    """
    Entry for mapping a selected class in the sample to a present class in the Database.
    """
    def __init__(self, classname: str, sqlClasses: List[str]):
        """
        :param classname: Name of the selected class.
        :param sqlClasses: List of class names available in SQL Database.
        """
        super(ClassEntry, self).__init__()
        self._nameLabel: QtWidgets.QLabel = QtWidgets.QLabel(classname)
        font: QtGui.QFont = QtGui.QFont()
        font.setBold(True)
        self._nameLabel.setFont(font)

        self._selectCheckbox: QtWidgets.QCheckBox = QtWidgets.QCheckBox("Upload")
        self._selectCheckbox.stateChanged.connect(self._enableDisableSelection)
        self._typeCombo: QtWidgets.QComboBox = QtWidgets.QComboBox()
        self._typeCombo.addItems(sqlClasses)
        self._setTypeComboToMostLikelyClass()
        self._lineEdit: QtWidgets.QLineEdit = QtWidgets.QLineEdit()
        self._presentRadioBtn: QtWidgets.QRadioButton = QtWidgets.QRadioButton("Use present class")
        self._newNameRadioBtn: QtWidgets.QRadioButton = QtWidgets.QRadioButton("Create new class")
        self._presentRadioBtn.setChecked(True)
        for btn in [self._presentRadioBtn, self._newNameRadioBtn]:
            btn.toggled.connect(self._enableDisableWidgets)

        layout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._selectCheckbox)

        self.contentGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox()
        self.contentGroup.setFlat(True)
        contentLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        self.contentGroup.setLayout(contentLayout)

        contentLayout.addWidget(self._nameLabel)
        contentLayout.addWidget(QtWidgets.QLabel(f"\tMap to:"))

        choiceLayout: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        choiceLayout.addWidget(self._presentRadioBtn, 0, 0)
        choiceLayout.addWidget(self._typeCombo, 0, 1)
        choiceLayout.addWidget(self._newNameRadioBtn, 1, 0)
        choiceLayout.addWidget(self._lineEdit, 1, 1)
        contentLayout.addLayout(choiceLayout)

        layout.addWidget(self.contentGroup)
        self._enableDisableWidgets()
        self._enableDisableSelection()

    def isSelected(self) -> bool:
        """
        Returns whether or not the class is selected for upload.
        """
        return self._selectCheckbox.isChecked()

    def getOrigignalName(self) -> str:
        """
        Returns the original name of the class the widget represents.
        """
        return self._nameLabel.text()

    def getTargetName(self) -> str:
        """
        Returns the name of the "target class name", i.e., the sql database class name that should be used.
        Can be a new name or an alread present one.
        """
        if self._presentRadioBtn.isChecked():
            name: str = self._typeCombo.currentText()
        else:
            name: str = self._lineEdit.text()
        assert len(name) > 0, "The new indicated name must be at least 1 character long."
        return name

    def _setTypeComboToMostLikelyClass(self) -> None:
        """
        Convenience function to set the class text to the one that is most likely to be the best match to the
        given class.
        """
        clsName: str = self._nameLabel.text()
        availableClasses: List[str] = []
        for i in range(self._typeCombo.count()):
            availableClasses.append(self._typeCombo.itemText(i))
        bestMatches: List[str] = difflib.get_close_matches(clsName, availableClasses, n=1, cutoff=0.0)
        if len(bestMatches) > 0:
            self._typeCombo.setCurrentText(bestMatches[0])

    def _enableDisableWidgets(self) -> None:
        """
        Disables the interactive elements according to radio btn selection.
        """
        new: bool = self._newNameRadioBtn.isChecked()
        self._typeCombo.setEnabled(not new)
        self._lineEdit.setEnabled(new)

    def _enableDisableSelection(self) -> None:
        """
        Disables everything but the checkbox if required.
        """
        self.contentGroup.setEnabled(self._selectCheckbox.isChecked())


class SampleEntry(QtWidgets.QGroupBox):
    """
    Groupbox for showing and setting sample details.
    """
    def __init__(self, sqlConn: DBConnection):
        super(SampleEntry, self).__init__("Select Sample")
        self._conn: DBConnection = sqlConn
        self._presentSampleRadio: QtWidgets.QRadioButton = QtWidgets.QRadioButton("Present Sample")
        self._newSampleRadio: QtWidgets.QRadioButton = QtWidgets.QRadioButton("New Sample")
        self._newSampleRadio.setChecked(True)
        for radioBtn in [self._presentSampleRadio, self._newSampleRadio]:
            radioBtn.toggled.connect(self._enableDisableWidgets)

        self._sampleComboBox: QtWidgets.QComboBox = QtWidgets.QComboBox()
        try:
            sampleNames: List[str] = self._conn.getSampleNames()
        except ConnectionError as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error on retrieving sample names:\n{e}")
            return

        self._sampleComboBox.addItems(sampleNames)
        self._sampleComboBox.currentTextChanged.connect(self._updateSampleComment)

        self._newNameLineEdit: QtWidgets.QLineEdit = QtWidgets.QLineEdit()
        self._newCommentLineEdit: QtWidgets.QLineEdit = QtWidgets.QLineEdit()
        self._presentCommentLabel: QtWidgets.QLabel = QtWidgets.QLabel()

        self._enableDisableWidgets()
        if self._presentSampleRadio.isChecked():
            self._updateSampleComment(self._sampleComboBox.currentText())
        self._createLayout()

    def getSampleName(self) -> str:
        """
        Returns the name of the desired sample.
        """
        if self._newSampleRadio.isChecked():
            name: str = self._newNameLineEdit.text()
        else:
            name: str = self._sampleComboBox.currentText()
        return name

    def getNewSampleComment(self) -> str:
        """
        Returns the comment for the new sample
        """
        return self._newCommentLineEdit.text()

    @QtCore.pyqtSlot(str)
    def _updateSampleComment(self, sampleName: str) -> None:
        """
        Updates UI to display the according sample comment string.
        """
        assert self._presentSampleRadio.isChecked()
        try:
            comment: str = self._conn.getCommentOfSample(sampleName)
        except ConnectionError as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error on updating sample selection:\n{e}")
            return

        except AssertionError as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error on updating sample selection:\n{e}")
            return

        self._presentCommentLabel.setText(comment)

    def _enableDisableWidgets(self) -> None:
        """
        Enable/Disable widgets according to the current sample type selection.
        """
        new: bool = self._newSampleRadio.isChecked()
        self._sampleComboBox.setEnabled(not new)
        self._presentCommentLabel.setEnabled(not new)
        self._newNameLineEdit.setEnabled(new)
        self._newCommentLineEdit.setEnabled(new)

    def _createLayout(self) -> None:
        radioLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        radioLayout.addWidget(self._newSampleRadio)
        radioLayout.addWidget(self._presentSampleRadio)

        newGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox("New Sample")
        newGroupLayout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        newGroup.setLayout(newGroupLayout)
        newGroupLayout.addWidget(QtWidgets.QLabel("Sample Name:"))
        newGroupLayout.addWidget(self._newNameLineEdit)
        newGroupLayout.addWidget(QtWidgets.QLabel("Comment:"))
        newGroupLayout.addWidget(self._newCommentLineEdit)
        newGroup.setFixedWidth(300)

        presentGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox("Present Sample")
        presentGroupLayout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        presentGroup.setLayout(presentGroupLayout)
        presentGroupLayout.addWidget(QtWidgets.QLabel("Sample to use:"))
        presentGroupLayout.addWidget(self._sampleComboBox)
        presentGroupLayout.addWidget(QtWidgets.QLabel("Comment:"))
        presentGroupLayout.addWidget(self._presentCommentLabel)
        presentGroup.setFixedWidth(300)

        detailLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        detailLayout.addWidget(newGroup)
        detailLayout.addWidget(presentGroup)

        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        layout.addLayout(radioLayout)
        layout.addLayout(detailLayout)


class SpecDetailEntry(QtWidgets.QGroupBox):
    """
    Groupbox for defining spectra accumulation details.
    """
    def __init__(self, specTypes: List[str]):
        """
        :param specTypes: List of possible spectra types.s
        """
        super(SpecDetailEntry, self).__init__("Accumulation Details")
        self._specTypeCombo: QtWidgets.QComboBox = QtWidgets.QComboBox()
        self._specTypeCombo.addItems(specTypes)

        self._numAccumSpinbox: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
        self._numAccumSpinbox.setMinimum(1)
        self._numAccumSpinbox.setMaximum(1000)
        self._numAccumSpinbox.setValue(1)

        self._accTimeSpinbox: QtWidgets.QDoubleSpinBox = QtWidgets.QDoubleSpinBox()
        self._accTimeSpinbox.setMinimum(0.01)
        self._accTimeSpinbox.setMaximum(1e5)
        self._accTimeSpinbox.setValue(90.0)

        self._resSpinBox: QtWidgets.QDoubleSpinBox = QtWidgets.QDoubleSpinBox()
        self._resSpinBox.setMinimum(0.1)
        self._resSpinBox.setMaximum(100.0)
        self._resSpinBox.setValue(7.0)

        for widget in [self._specTypeCombo, self._resSpinBox, self._accTimeSpinbox, self._numAccumSpinbox]:
            widget.setFixedWidth(150)

        layout: QtWidgets.QFormLayout = QtWidgets.QFormLayout()
        layout.addRow("Spectra Type", self._specTypeCombo)
        layout.addRow("Number Accumulations", self._numAccumSpinbox)
        layout.addRow("Accumulation time (ms)", self._accTimeSpinbox)
        layout.addRow("Pixel resolution (µm/px)", self._resSpinBox)
        self.setLayout(layout)

    def getDetails(self) -> 'SpecDetails':
        details: SpecDetails = SpecDetails(
            self._numAccumSpinbox.value(),
            self._accTimeSpinbox.value(),
            self._resSpinBox.value(),
            self._specTypeCombo.currentText()
        )
        return details


class ProgressWindow(QtWidgets.QWidget):
    def __init__(self):
        super(ProgressWindow, self).__init__()
        self._progressbar: QtWidgets.QProgressBar = QtWidgets.QProgressBar()
        self._progressbar.setFixedWidth(500)
        self._infoLabel: QtWidgets.QLabel = QtWidgets.QLabel()
        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._progressbar)
        layout.addWidget(self._infoLabel, QtCore.Qt.AlignCenter)

    def setupToSample(self, sampleName: str, numSpectra: int) -> None:
        """
        Setup the progresswindow to a new sample.
        :param sampleName: Name of the current samples.
        :param numSpectra: Number of spectra (in total, for all samples to upload).
        """
        self.setWindowTitle(f"Uploading {sampleName}")
        self._progressbar.setValue(0)
        self._progressbar.setMaximum(numSpectra)
        self._updateInfoLabel()

    # def increment(self) -> None:
    #     """
    #     Increments the progressbar.
    #     """
    #     self._progressbar.setValue(self._progressbar.value()+1)
    #     self._updateInfoLabel()

    def setValue(self, newValue: int) -> None:
        """
        Sets a new status value
        """
        self._progressbar.setValue(newValue)
        self._updateInfoLabel()

    def isFinished(self) -> None:
        return self._progressbar.value() == self._progressbar.maximum()

    def _updateInfoLabel(self) -> None:
        """
        Updates the label to indicate progress.
        """
        numDone, maxNum = self._progressbar.value(), self._progressbar.maximum()
        self._infoLabel.setText(f"Finished {numDone} of {maxNum} spectra.")
