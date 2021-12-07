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

if TYPE_CHECKING:
    from particles import Particle
    from gui.classUI import ClassInterpretationParams


def getContourItemForParticle(particle: 'Particle', classInterpParams: 'ClassInterpretationParams') -> 'ParticleContour':
    contour: ParticleContour = ParticleContour()
    contour.setupParticle(particle, classInterpParams)
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
        self.setAcceptHoverEvents(True)
        self._particleID: int = -1
        self._polygon: Union[None, QtGui.QPolygonF] = None
        self._color = QtGui.QColor(180, 255, 180, 200)
        self._alpha: float = 0.75
        self._partilcleInfo: ParticleInfo = ParticleInfo()
        self._partilcleInfo.setParentItem(self)
        self._infoHasToBeVisible: bool = False

    def setupParticle(self, particle: 'Particle', classInterpParams: 'ClassInterpretationParams') -> None:
        """
        Calculates the bounding rect (needed for drawing the QGraphicsView) and converts the contourdata to a polygon.
        :return:
        """
        self._particleID = particle.getID()
        contourData = particle.getContour()
        self._polygon = QtGui.QPolygonF()
        x0 = contourData[:, 0, 0].min()
        x1 = contourData[:, 0, 0].max()
        y0 = contourData[:, 0, 1].min()
        y1 = contourData[:, 0, 1].max()
        for point in contourData:
            self._polygon.append(QtCore.QPointF(point[0, 0], point[0, 1]))

        self.brect.setCoords(x0, y0, x1, y1)

        self._partilcleInfo.setPos(x0 + (x1-x0)/2, y0 + (y1-y0)/2)
        self._partilcleInfo.setClassName(particle.getAssignment(classInterpParams))

    def hoverEnterEvent(self, event) -> None:
        self._partilcleInfo.show()

    def hoverLeaveEvent(self, event) -> None:
        if not self._infoHasToBeVisible:
            self._partilcleInfo.hide()

    def infoIsVisible(self) -> bool:
        return self._partilcleInfo.isVisible()

    def setAssignment(self, assignment: str) -> None:
        self._partilcleInfo.setClassName(assignment)

    def boundingRect(self):
        return self.brect

    def getParticleID(self) -> int:
        """
        Returns the particle's id.
        """
        return self._particleID

    def setColor(self, color: Tuple[int, int, int]) -> None:
        """
        Sets color of contour
        :param color: Tuple[R: int, G: int, B: int]
        :return:
        """
        self._color = QtGui.QColor(color[0], color[1], color[2])

    def setInfoVisibility(self, visible: bool) -> None:
        """
        Sets visibility of the particle info overlay.
        """
        self._partilcleInfo.setVisible(visible)
        self._infoHasToBeVisible = visible

    def paint(self, painter, option, widget):
        if self._polygon is not None:
            painter.setPen(QtCore.Qt.green)
            painter.setBrush(self._color)
            painter.setOpacity(self._alpha)
            painter.drawPolygon(self._polygon)


class ParticleInfo(QtWidgets.QGraphicsItem):
    """
    Small overlay indicating the particle's class.
    """
    margin: int = 3
    pixelsize: int = 7

    def __init__(self):
        super(ParticleInfo, self).__init__()
        self._cls: str = ""
        self._width: int = 0
        self._height: int = 0
        self._font: QtGui.QFont = QtGui.QFont()
        self._font.setPixelSize(self.pixelsize)
        self.hide()

    def setClassName(self, className: str) -> None:
        self._cls = className
        fontMetric: QtGui.QFontMetrics = QtGui.QFontMetrics(self._font)
        self._width = (fontMetric.boundingRect(className).width())*2 + 2*self.margin
        self._height = fontMetric.boundingRect(className).height() + 2*self.margin

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(-self.margin, -self.margin, self._width, self._height)

    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        if self._cls:
            painter.setPen(QtCore.Qt.white)
            painter.drawText(0, self.margin+self.pixelsize, self._cls)
