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
import numpy as np
from PyQt5 import QtGui, QtCore
from typing import Tuple


def addBlueGradientToPainter(painter: QtGui.QPainter) -> QtGui.QPainter:
    """
    Adds a gradient from darkBlue to blue to the painter and returns it.
    """
    painter.setPen(QtCore.Qt.darkBlue)
    painter.setBrush(QtCore.Qt.blue)
    return painter


def getIOGradient(gradient: QtGui.QLinearGradient, selected: bool, color1: Tuple[int, int, int],
                  color2: Tuple[int, int, int]) -> QtGui.QLinearGradient:
    """
    Gets a linear gradient for the input_output_area of nodes.
    :param gradient: The gradient template to use (already has the correct size)
    :param selected: Whether or not the node is selected
    :param color1: Start (input) color, reflecting the input data type
    :param color2: End (output) color, reflecting the output data tupe
    """
    # convert for easier data handling
    color1: np.ndarray = np.array(color1, dtype=np.float)
    color2: np.ndarray = np.array(color2, dtype=np.float)

    if not selected:
        color1 *= 0.9
        color2 *= 0.8

    color1: QtGui.QColor = QtGui.QColor(int(color1[0]), int(color1[1]), int(color1[2]))
    color2: QtGui.QColor = QtGui.QColor(int(color2[0]), int(color2[1]), int(color2[2]))

    gradient.setColorAt(0, color1)
    gradient.setColorAt(1, color2)

    # if selected:
    #     gradient.setColorAt(0, QtCore.Qt.gray)
    #     gradient.setColorAt(0.8, QtCore.Qt.lightGray)
    #     gradient.setColorAt(1, QtCore.Qt.white)
    # else:
    #     gradient.setColorAt(0, QtCore.Qt.darkGray)
    #     gradient.setColorAt(0.8, QtCore.Qt.gray)
    #     gradient.setColorAt(1, QtCore.Qt.lightGray)

    return gradient