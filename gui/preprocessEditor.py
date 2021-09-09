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
from PyQt5 import QtWidgets, QtCore
from typing import List, TYPE_CHECKING

from preprocessing.preprocessors import getPreprocessors
from gui.nodegraph.nodegraph import NodeGraph
if TYPE_CHECKING:
    from preprocessing.preprocessors import Preprocessor


class PreprocessingSelector(QtWidgets.QGroupBox):
    ProcessorStackUpdated: QtCore.pyqtSignal = QtCore.pyqtSignal()

    def __init__(self):
        super(PreprocessingSelector, self).__init__()
        self.setWindowTitle("Define Preprocessing")
        
        self._nodeGraph: NodeGraph = NodeGraph()

        self._layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self._layout.addWidget(self._nodeGraph)
        self.setLayout(self._layout)

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
        raise NotImplementedError


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    editor: PreprocessingSelector = PreprocessingSelector()
    editor.show()
    app.exec_()
