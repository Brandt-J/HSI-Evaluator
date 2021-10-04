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
from typing import *
import numpy as np

if TYPE_CHECKING:
    from particles import Particle


def getContourItemForParticle(particle: 'Particle') -> 'ParticleContour':
    contour: ParticleContour = ParticleContour()
    contour.setupParticle(particle)
    return contour


class ParticleContour(QtWidgets.QGraphicsObject):
    """
    Graphics Object for displaying a particle contour
    """
    def __init__(self):
        super(ParticleContour, self).__init__()
        self.setZValue(1)
        self.setPos(0, 0)
        self.brect = QtCore.QRectF(0, 0, 1, 1)
        self.particle: Union[None, 'Particle'] = None
        self.polygon: Union[None, QtGui.QPolygonF] = None
        self.color = QtGui.QColor(180, 255, 180, 200)

    def setupParticle(self, particle: 'Particle') -> None:
        """
        Calculates the bounding rect (needed for drawing the QGraphicsView) and converts the contourdata to a polygon.
        :return:
        """
        self.particle = particle
        contourData = particle.getContour()
        self.polygon = QtGui.QPolygonF()
        x0 = contourData[:, 0, 0].min()
        x1 = contourData[:, 0, 0].max()
        y0 = contourData[:, 0, 1].min()
        y1 = contourData[:, 0, 1].max()
        for point in contourData:
            self.polygon.append(QtCore.QPointF(point[0, 0], point[0, 1]))

        self.brect.setCoords(x0, y0, x1, y1)

    def boundingRect(self):
        return self.brect

    def setColor(self, color: QtGui.QColor):
        """
        sets color of contour
        :param QtGui.QColor color:
        :return:
        """
        self.color = color

    def paint(self, painter, option, widget):
        if self.polygon is not None:
            painter.setPen(QtCore.Qt.green)
            painter.setBrush(self.color)
            painter.drawPolygon(self.polygon)
