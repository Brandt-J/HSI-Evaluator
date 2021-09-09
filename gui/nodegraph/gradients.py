"""
GEPARD - Gepard-Enabled PARticle Detection
Copyright (C) 2018  Lars Bittrich and Josef Brandt, Leibniz-Institut für
Polymerforschung Dresden e. V. <bittrich-lars@ipfdd.de>

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

from PyQt5 import QtGui, QtCore


def addBlueGradientToPainter(painter: QtGui.QPainter) -> QtGui.QPainter:
    painter.setPen(QtCore.Qt.darkBlue)
    painter.setBrush(QtCore.Qt.blue)
    return painter
