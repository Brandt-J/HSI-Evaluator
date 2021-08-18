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

from PyQt5 import QtCore, QtWidgets
import os


def getAppFolder() -> str:
    """
    Returns a writable locatione, specific for the Imec Evaluator App.
    """
    app = QtWidgets.QApplication.instance()
    if app is None:
        # if it does not exist then a QApplication is created
        app = QtWidgets.QApplication([])

    app.setApplicationName("HSI Evaluator")
    appFolder: str = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.AppLocalDataLocation)
    os.makedirs(appFolder, exist_ok=True)
    return appFolder