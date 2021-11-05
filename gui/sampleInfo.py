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

from PyQt5 import QtWidgets, QtCore, QtGui
from typing import *


class SampleInfo(QtWidgets.QGraphicsItem):
    """
    Container GraphicsItem for displaying sample info and configuring use during training/inference
    """
    Spacing: int = 20
    
    def __init__(self, sampleName: str):
        super(SampleInfo, self).__init__()
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)
        self._elements: List['SampleInfoElement'] = []

        self._sampleField: SampleNameField = SampleNameField(sampleName)
        self._editBtn: EditNameButton = EditNameButton()
        self._checkTrain: CheckBoxElement = CheckBoxElement("Training")
        self._checkTest: CheckBoxElement = CheckBoxElement("Inference")

        self._postitionElements()

    def isCheckedForTraining(self) -> bool:
        return self._checkTrain.isChecked()

    def isCheckedForInference(self) -> bool:
        return self._checkTest.isChecked()

    def getChangeNameBtn(self) -> 'EditNameButton':
        return self._editBtn

    def setName(self, newName: str) -> None:
        self._sampleField.setName(newName)
        self._postitionElements()
        self.update()

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, 0, self._getTotalWidth(), self._getMaxHeight() + 2*self.Spacing)

    def _postitionElements(self) -> None:
        self._elements = []
        for element in [self._sampleField, self._editBtn, self._checkTrain, self._checkTest]:
            self._addElement(element)

    def _addElement(self, newElement: 'SampleInfoElement') -> None:
        xPos: int = self.Spacing
        yPos: int = self.Spacing
        if len(self._elements) > 0:
            xPos += self._getTotalWidth() - 2*self.Spacing  # ignore the outer spacings
            yPos += int(round((self._getMaxHeight() - newElement.Height) / 2))
        newElement.setPos(xPos, yPos)
        newElement.setParentItem(self)
        self._elements.append(newElement)
        self.update()

    def _getMaxHeight(self) -> int:
        if len(self._elements) > 0:
            height: int = max([element.Height for element in self._elements])
        else:
            height: int = 0
        return height

    def _getMaxWidth(self) -> int:
        if len(self._elements) > 0:
            width: int = max([element.Width for element in self._elements])
        else:
            width: int = 0
        return width

    def _getTotalWidth(self) -> int:
        return sum([elem.Width for elem in self._elements]) + (len(self._elements)+1) * self.Spacing

    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        rect: QtCore.QRectF = self.boundingRect()
        painter.setPen(QtCore.Qt.GlobalColor.white)
        grad: QtGui.QLinearGradient = QtGui.QLinearGradient(0, 0, rect.width(), rect.height())
        grad.setColorAt(0, QtCore.Qt.GlobalColor.darkGray)
        grad.setColorAt(1, QtCore.Qt.GlobalColor.black)
        painter.setBrush(grad)
        painter.setOpacity(0.5)

        path: QtGui.QPainterPath = QtGui.QPainterPath()
        path.addRoundedRect(rect, 5, 5)

        painter.drawPath(path)


class SampleInfoElement(QtWidgets.QGraphicsObject):
    Width: int = 20
    Height: int = 20


class SampleNameField(SampleInfoElement):
    Width: int = 150
    Height: int = 15

    def __init__(self, name: str):
        super(SampleNameField, self).__init__()
        self._name: str = name
        self._font: QtGui.QFont = QtGui.QFont()
        self._font.setPixelSize(self.Height)
        self._font.setBold(True)

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, 0, self.Width, self.Height)

    def setName(self, newName: str) -> None:
        self._name = newName
        fontMetric: QtGui.QFontMetrics = QtGui.QFontMetrics(self._font)
        self.Width = fontMetric.boundingRect(newName).width() + 20
        self.update()

    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        painter.setPen(QtCore.Qt.GlobalColor.white)
        painter.setFont(self._font)
        painter.drawText(0, self.Height, self._name)


class EditNameButton(SampleInfoElement):
    Width = 25
    Height = 25
    colors: Dict[str, QtCore.Qt.GlobalColor] = {"Normal": QtCore.Qt.GlobalColor.gray,
                                                "Hovered": QtCore.Qt.GlobalColor.lightGray,
                                                "Clicked": QtCore.Qt.GlobalColor.darkGray}

    ButtonClicked: QtCore.pyqtSignal = QtCore.pyqtSignal()

    def __init__(self):
        super(EditNameButton, self).__init__()
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)

        self._isHovered: bool = False
        self._isClicked: bool = False

        style = QtWidgets.QWidget().style()
        self._icon: QtGui.QIcon = style.standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogResetButton'))
        self.setToolTip("Edit the sample's name.")

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, 0, self.Width, self.Height)

    def hoverEnterEvent(self, event) -> None:
        self._isHovered = True
        self.update()

    def hoverLeaveEvent(self, event) -> None:
        self._isHovered = False
        self.update()

    def mousePressEvent(self, event) -> None:
        self._isClicked = True
        self.update()
    
    def mouseReleaseEvent(self, event) -> None:
        self._isClicked = False
        self.ButtonClicked.emit()
        self.update()
    
    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        painter.setPen(QtCore.Qt.GlobalColor.black)
        if self._isClicked:
            painter.setBrush(self.colors["Clicked"])
        else:
            if self._isHovered:
                painter.setBrush(self.colors["Hovered"])
            else:
                painter.setBrush(self.colors["Normal"])

        path = QtGui.QPainterPath()
        path.addRoundedRect(self.boundingRect(), 3, 3)
        painter.drawPath(path)
        iconSize: int = int(round(self.Width * 0.8))
        offset: int = int(round((self.Width - iconSize) / 2))
        self._icon.paint(painter, QtCore.QRect(offset, offset, iconSize, iconSize))


class CheckBoxElement(SampleInfoElement):
    Width = 75
    Height = 15

    def __init__(self, label: str):
        super(CheckBoxElement, self).__init__()
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)

        self._label: str = label
        self._isChecked: bool = True

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, 0, self.Width, self.Height)

    def isChecked(self) -> bool:
        return self._isChecked

    def mousePressEvent(self, event) -> None:
        self._isChecked = not self._isChecked
        self.update()

    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        spacing: int = 5
        textLength: int = self.Width - self.Height - spacing
        checkBoxStart: int = self.Width - self.Height
        checkBoxSize: int = self.Height

        painter.setPen(QtCore.Qt.GlobalColor.white)
        painter.drawText(QtCore.QRectF(0, 0, textLength, self.Height), self._label)

        painter.setPen(QtCore.Qt.GlobalColor.black)
        painter.setBrush(QtCore.Qt.GlobalColor.white)
        painter.drawRect(checkBoxStart, 0, checkBoxSize, checkBoxSize)
        if self._isChecked:
            painter.drawLine(QtCore.QLine(checkBoxStart, 0, self.Width, self.Height))
            painter.drawLine(QtCore.QLine(checkBoxStart, self.Height, self.Width, 0))
