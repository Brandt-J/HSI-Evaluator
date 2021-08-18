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
from PyQt5 import QtWidgets, QtGui, QtCore
from typing import List, TYPE_CHECKING

from preprocessors import getPreprocessors
if TYPE_CHECKING:
    from preprocessors import Preprocessor


class PreprocessingSelector(QtWidgets.QGroupBox):
    ProcessorStackUpdated: QtCore.pyqtSignal = QtCore.pyqtSignal()

    def __init__(self):
        super(PreprocessingSelector, self).__init__()
        self.setWindowTitle("Define Preprocessing")
        
        self._layout: QtWidgets.QGridLayout = QtWidgets.QGridLayout()

        self._available: List[SelectableLabel] = self._getPreprocessorLabels()
        self._selected: List[SelectableLabel] = []

        self._addBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("-Add->")
        self._addBtn.released.connect(self._select)
        self._removeBtn: QtWidgets.QPushButton = QtWidgets.QPushButton("<-Remove-")
        self._removeBtn.released.connect(self._deselect)

        self.setLayout(self._layout)
        self._recreateLayout()

    def _getPreprocessorLabels(self) -> List['SelectableLabel']:
        labelList: List['SelectableLabel'] = []
        for processor in getPreprocessors():
            labelList.append(SelectableLabel(processor.label))
        return labelList

    def getPreprocessors(self) -> List['Preprocessor']:
        """
        Returns a list of the currently selected preprocessors.
        """
        selectedProcessors: List['Preprocessor'] = []
        availableProc: List['Preprocessor'] = getPreprocessors()
        for lbl in self._selected:
            for proc in availableProc:
                if proc.label == lbl.text():
                    selectedProcessors.append(proc)
                    break
        return selectedProcessors

    def getPreprocessorNames(self) -> List[str]:
        """
        Returns a list of the currently selected preprocessor names. Used for storing
        """
        return [lbl.text() for lbl in self._selected]

    def selectPreprocessors(self, processorNames: List[str]) -> None:
        """
        Takes a list of processor names and sets the current selection to that.
        :param processorNames: List of processor Names
        """
        self._selected = []
        self._available = self._getPreprocessorLabels()
        for name in processorNames:
            for label in self._available:
                if label.text() == name:
                    self._selected.append(label)
                    self._available.remove(label)
                    break
        self._recreateLayout()

    def _recreateLayout(self) -> None:
        self._layout.addWidget(QtWidgets.QLabel("Avalable"), 0, 0, QtCore.Qt.AlignCenter)
        self._layout.addWidget(QtWidgets.QLabel("Selected"), 0, 2, QtCore.Qt.AlignCenter)

        btnLayout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        btnLayout.addStretch()
        btnLayout.addWidget(self._removeBtn)
        btnLayout.addWidget(self._addBtn)
        btnLayout.addStretch()

        self._layout.addWidget(self._getLabelScrollArea(self._available), 1, 0)
        self._layout.addLayout(btnLayout, 1, 1)
        self._layout.addWidget(self._getLabelScrollArea(self._selected), 1, 2)

        self._deselectAllLabels()

    def _select(self) -> None:
        selected: List['SelectableLabel'] = []
        for label in self._available:
            if label.isSelected():
                self._selected.append(label)
                selected.append(label)
                
        for label in selected:
            self._available.remove(label)
        
        self._recreateLayout()
        self.ProcessorStackUpdated.emit()

    def _deselect(self) -> None:
        deselected: List['SelectableLabel'] = []
        for label in self._selected:
            if label.isSelected():
                self._available.append(label)
                deselected.append(label)
        
        for label in deselected:
            self._selected.remove(label)

        self._recreateLayout()
        self.ProcessorStackUpdated.emit()

    def _getLabelScrollArea(self, labelList: List['SelectableLabel']) -> QtWidgets.QScrollArea:
        group: QtWidgets.QGroupBox = QtWidgets.QGroupBox()
        group.setStyleSheet("QGroupBox{border:0;}")
        group.setMinimumWidth(150)
        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()

        for label in labelList:
            layout.addWidget(label)

        group.setLayout(layout)
        area: QtWidgets.QScrollArea = QtWidgets.QScrollArea()
        area.setWidget(group)
        return area

    def _deselectAllLabels(self) -> None:
        for lbl in self._selected + self._available:
            lbl.deselect()


class SelectableLabel(QtWidgets.QLabel):
    def __init__(self, label: str):
        super(SelectableLabel, self).__init__()
        self.setText(label)
        self._isSelected: bool = False
        self._font = QtGui.QFont()
        self._setStyle()
        self.setFixedHeight(30)
        self.setMinimumWidth(50)
        self.setMaximumWidth(200)
        self.setToolTip(label)

    def isSelected(self) -> bool:
        return self._isSelected

    def deselect(self) -> None:
        self._isSelected = False
        self._setStyle()

    def mousePressEvent(self, event) -> None:
        self._isSelected = not self._isSelected
        self._setStyle()

    def _setStyle(self) -> None:
        if self._isSelected:
            self.setStyleSheet("QLabel{border: 1px solid black;"
                               "background-color: #b3d08b;"
                               "border-radius: 7}")
            self._font.setBold(True)
        else:
            self._font.setBold(False)
            self.setStyleSheet("QLabel{border: 1px solid gray;"
                               "background-color: white;"
                               "border-radius: 7}")
        self.setFont(self._font)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    editor: PreprocessingSelector = PreprocessingSelector()
    editor.show()
    app.exec_()
