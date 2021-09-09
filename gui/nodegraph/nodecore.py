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

from PyQt5 import QtWidgets, QtCore, QtGui
from typing import *
from enum import Enum, auto
import numpy as np
if TYPE_CHECKING:
    from gui.nodegraph.nodegraph import NodeGraph
    from logging import Logger


class BaseNode(QtWidgets.QGraphicsWidget):
    """
    Base Node Class for Spectra Processing.
    """
    label = 'BaseNode'
    defaultParams: dict = {}

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(BaseNode, self).__init__()
        self._parentGraph: 'NodeGraph' = nodeGraphParent
        self._inputs: List['Input'] = []
        self._outputs: List['Output'] = []
        self._bodywidget: Union[None, QtWidgets.QWidget] = None
        self._logger: 'Logger' = logger
        self._showPreviewBtn: bool = True
        self.isStartNode: bool = False
        self.id: int = -1
        self._alpha: float = 1.0

        self.setPos(pos.x(), pos.y())
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        self._layout: QtWidgets.QGraphicsLinearLayout = QtWidgets.QGraphicsLinearLayout()
        self._layout.setOrientation(QtCore.Qt.Vertical)
        self.setLayout(self._layout)
        self._origLayoutSize: QtCore.QSize = QtCore.QSize()
        self._pen: QtGui.QPen = QtGui.QPen(QtCore.Qt.black, 2)
        self._ioGradient: QtGui.QLinearGradient = None
        self._bodyGradient: QtGui.QLinearGradient = None
        self._inputStop: int = np.nan  # Vertical position where inputArea stops
        self._bodyHeight: int = np.nan  # Vertical height of body group

    def _populateLayoutAndCreateIO(self) -> None:
        def getGroupBox(layout: QtWidgets.QLayout) -> QtWidgets.QGroupBox:
            _group: QtWidgets.QGroupBox = QtWidgets.QGroupBox()
            _group.setFlat(True)
            _group.setStyleSheet("border: 0;")
            _group.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            _group.setLayout(layout)
            return _group

        scene: QtWidgets.QGraphicsScene = self._parentGraph.scene()
        inputGroup: QtWidgets.QGroupBox = getGroupBox(QtWidgets.QGridLayout())
        rowOffset = 1 if np.all([inp.isConnected for inp in self._inputs]) else 0
        for i, inp in enumerate(self._inputs):
            if not inp.isConnected():
                inputGroup.layout().addWidget(inp.getWidget(), 0, i)

            inputGroup.layout().addWidget(inp.getLabelWidget(), rowOffset, i)

        bodyGroup: QtWidgets.QGroupBox = getGroupBox(QtWidgets.QVBoxLayout())
        bodyGroup.layout().addWidget(getTransparentQLabel(self.label, bold=True))
        if self._bodywidget is not None:
            bodyGroup.layout().addWidget(self._bodywidget)

        outputGroup: QtWidgets.QGroupBox = getGroupBox(QtWidgets.QHBoxLayout())
        for i, outp in enumerate(self._outputs):
            outputGroup.layout().addWidget(outp.getLabelWidget())

        self._layout.addItem(scene.addWidget(inputGroup))
        self._layout.addItem(scene.addWidget(bodyGroup))
        self._layout.addItem(scene.addWidget(outputGroup))
        for i in range(3):
            self._layout.setItemSpacing(i, 0)
        self._origLayoutSize = self.preferredSize()
        self._updateGradients()
        self._addInputOutputItems()
        if self._showPreviewBtn:
            self._addPreviewButton()

        self._inputStop = int(inputGroup.rect().height() + self._layout.spacing())
        self._bodyHeight = int(bodyGroup.geometry().height() + self._layout.spacing())

    def toDict(self) -> dict:
        """
        Used for saving the nodes configuration
        """
        inDict: dict = {}
        for i, inp in enumerate(self._inputs):
            if inp.isConnected():
                connectedNodeID = inp.getConnectedNode().id
                connectedOutputID = inp.getConnectedOutputID()
                inDict[i] = [connectedNodeID, connectedOutputID]

        nodeDict: dict = {'label': self.label,
                          'id': self.id,
                          'params': {},
                          'inputs': inDict}
        return nodeDict

    def fromDict(self, configDict: dict) -> dict:
        """
        Used to restore configuration from the dict
        """
        pass

    def invalidateCache(self) -> None:
        """
        Will be called after particle detection. Can be overloaded to erase any temporarily cached results.
        """
        pass

    def pushPreview(self) -> None:
        """
        Create a preview image and send it to the detection preview window.
        """
        if self._isConnectedToRGBInput():
            prevImg: np.ndarray = self._getPreviewImage()
            if prevImg is not None:
                self.previewGenerated.emit(prevImg)

    def _isConnectedToRGBInput(self) -> bool:
        return recursiveIsConnectedToRGBInputNode(self)

    def _getPreviewImage(self) -> np.ndarray:
        return None

    def select(self) -> None:
        self._updateGradients(selected=True)
        self.update()

    def deselect(self) -> None:
        self._updateGradients(selected=False)
        self.update()

    def disable(self) -> None:
        """
        Sets the node disabled, when a detection is running.
        """
        if self._bodywidget is not None:
            self._bodywidget.setDisabled(True)
        self._alpha = 0.5
        self.update()

    def enable(self) -> None:
        """
        Sets the node enabled, when a detection is finished.
        """
        if self._bodywidget is not None:
            self._bodywidget.setDisabled(False)
        self._alpha = 1.0
        self.update()

    def _updateGradients(self, selected: bool = False) -> None:
        self._ioGradient = self._getIOGradient(selected)
        self._bodyGradient = self._getBodyGradient(selected)

    def _addInputOutputItems(self) -> None:
        for inp in self._inputs:
            xPos = inp.getLabelWidget().pos().x()  # i.e., lower corner
            xPos += (inp.getLabelWidget().size().width() + inp.diameter) / 2
            inp.setParentItem(self)
            inp.setPos(xPos, 0)
            inp.capConnection.connect(self._parentGraph.capConnectionFrom)

        for outp in self._outputs:
            diameter = outp.diameter
            xPos = outp.getLabelWidget().pos().x()  # i.e., lower corner
            xPos += (outp.getLabelWidget().sizeHint().width() + diameter) / 2
            yPos = int(self.boundingRect().height() - diameter)
            outp.setParentItem(self)
            outp.setPos(xPos, yPos)
            outp.dragConnection.connect(self._startDragConnection)

    def _addPreviewButton(self) -> None:
        prevBtn: PreviewButton = PreviewButton(self)
        prevBtn.setParentItem(self)
        diameter = prevBtn.diameter
        margin = 5
        nodeWidth = self.preferredSize().width()
        prevBtn.setPos(nodeWidth - diameter - margin, margin)

    def _getIOGradient(self, selected: bool = False) -> QtGui.QLinearGradient:
        grad: QtGui.QLinearGradient = QtGui.QLinearGradient(0, 0, self.preferredWidth(), 0)
        if selected:
            grad.setColorAt(0, QtCore.Qt.gray)
            grad.setColorAt(0.8, QtCore.Qt.lightGray)
            grad.setColorAt(1, QtCore.Qt.white)
        else:
            grad.setColorAt(0, QtCore.Qt.darkGray)
            grad.setColorAt(0.8, QtCore.Qt.gray)
            grad.setColorAt(1, QtCore.Qt.lightGray)
        return grad

    def _getBodyGradient(self, selected: bool = False) -> QtGui.QLinearGradient:
        grad: QtGui.QLinearGradient = QtGui.QLinearGradient(0, 0, self.preferredWidth(), 0)
        if selected:
            grad.setColorAt(0, QtGui.QColor(230, 255, 230))
            grad.setColorAt(0.2, QtGui.QColor(180, 255, 180))
            grad.setColorAt(1, QtGui.QColor(128, 128, 128))
        else:
            grad.setColorAt(0, QtGui.QColor(200, 255, 200))
            grad.setColorAt(0.2, QtGui.QColor(128, 255, 128))
            grad.setColorAt(1, QtGui.QColor(64, 128, 64))
        return grad

    def _startDragConnection(self, slot: QtWidgets.QGraphicsObject) -> None:
        if type(slot) == Input:
            self._parentGraph.dragConnectionTo(slot)
        elif type(slot) == Output:
            self._parentGraph.dragConnectionFrom(slot)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self._parentGraph.updateScene()
        return super(BaseNode, self).itemChange(change, value)

    def mousePressEvent(self, event) -> None:
        self._parentGraph.selectNode(self)

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, 0, self._origLayoutSize.width(), self._origLayoutSize.height())

    def getOutput(self, outputName: str = '') -> object:
        raise NotImplementedError

    def getOutputs(self) -> List['Output']:
        return self._outputs

    def getIndexOfOutput(self, output: 'Output') -> int:
        assert output in self._outputs, f'Output {output} not in list of outputs of Node {self.label}'
        return self._outputs.index(output)

    def getInputs(self) -> List['Input']:
        return self._inputs

    def getNodeGraph(self) -> 'NodeGraph':
        return self._parentGraph

    def getInputOrOutputAtPos(self, scenepos: QtCore.QPointF) -> Union['Input', 'Output']:
        inOrOut: Union['Input', 'Output'] = None
        pos: QtCore.QPointF = self.mapFromScene(scenepos)  # relative to self
        if self.boundingRect().contains(pos):
            for inout in self._inputs + self._outputs:
                if inout.boundingRect().contains(pos - inout.pos()):
                    inOrOut = inout
                    break

        return inOrOut

    def paint(self, painter, option, widget) -> None:
        painter.setPen(self._pen)
        painter.setOpacity(self._alpha)
        painter.setBrush(self._ioGradient)
        path = QtGui.QPainterPath()
        path.addRoundedRect(self.boundingRect(), 10, 10)
        painter.drawPath(path)

        painter.setBrush(self._bodyGradient)
        rect = QtCore.QRect(0, self._inputStop, int(self.preferredWidth()), self._bodyHeight)
        painter.drawRect(rect)


class Input(QtWidgets.QGraphicsObject):
    """
    Object representing an Node Input.
    """
    diameter: float = 10  # px diameter of widget
    capConnection: QtCore.pyqtSignal = QtCore.pyqtSignal(QtWidgets.QGraphicsObject)

    def __init__(self, name: str, dataTypes: List['DataType']):
        """
        :param name: Name of the input to display
        :param dataType: Type of the input.
        """
        super(Input, self).__init__()
        self.name: str = name
        self._types: List[DataType] = dataTypes
        self._connectedOutput: Union[None, 'Output'] = None
        self._qlabel: QtWidgets.QLabel = getTransparentQLabel(name)
        self.setZValue(1)

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, 0, self.diameter, self.diameter)

    def getValue(self):
        val: object = None
        if self._connectedOutput is not None:
            val = self._connectedOutput.getValue()

        return val

    def mousePressEvent(self, event) -> None:
        if self.isConnected():
            self.capConnection.emit(self)

    def isConnected(self) -> bool:
        return self._connectedOutput is not None

    def getWidget(self) -> QtWidgets.QWidget:
        return self._widget

    def getLabelWidget(self) -> QtWidgets.QLabel:
        return self._qlabel

    def getDataType(self) -> 'DataType':
        return self._type

    def getConnectedNode(self) -> 'BaseNode':
        otherNode: 'BaseNode' = None
        if self.isConnected():
            otherNode = self._connectedOutput.getNode()
        return otherNode

    def getConnectedOutputID(self) -> int:
        outID = -1
        if self.isConnected():
            outID = self.getConnectedNode().getIndexOfOutput(self._connectedOutput)
        return outID

    def acceptOutputConnection(self, otherOutput: 'Output') -> None:
        assert otherOutput.getVarType() == self._type
        self._connectedOutput = otherOutput

    def disconnect(self) -> None:
        self._connectedOutput = None

    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        painter = self._type.formatPainter(painter)
        painter.drawEllipse(self.boundingRect())


class Output(QtWidgets.QGraphicsObject):
    dragConnection: QtCore.pyqtSignal = QtCore.pyqtSignal(QtWidgets.QGraphicsObject)
    diameter: float = 10  # px diameter of widget

    def __init__(self, parentNode: 'BaseNode', name: str, dataType: 'DataType'):
        super(Output, self).__init__()
        self.name: str = name
        self._node: 'BaseNode' = parentNode
        self._type: DataType = dataType
        self._qlabel: QtWidgets.QLabel = getTransparentQLabel(name)
        self.setZValue(1)

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, 0, self.diameter, self.diameter)

    def mousePressEvent(self, event) -> None:
        self.dragConnection.emit(self)

    def getValue(self) -> object:
        return self._node.getOutput(self.name)

    def getVarType(self) -> 'DataType':
        return self._type

    def getLabelWidget(self) -> QtWidgets.QLabel:
        return self._qlabel

    def getNode(self) -> 'BaseNode':
        return self._node

    def connectTo(self, otherInput: 'Input') -> bool:
        success: bool = False
        if otherInput.getDataType() == self._type:
            otherInput.acceptOutputConnection(self)
            success = True
        return success

    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        painter = self._type.formatPainter(painter)
        painter.drawEllipse(self.boundingRect())


class ConnectionWire(QtWidgets.QGraphicsItem):
    def __init__(self, start: Union[Input, Output], end: Union[Input, Output, QtCore.QPointF] = None):
        super(ConnectionWire, self).__init__()
        self.setPos(0, 0)
        self._start: Union[Input, Output] = start
        self._end: Union[Input, Output, QtCore.QPointF] = end
        self._pen: QtGui.QPen = QtGui.QPen()
        self._pen.setColor(QtCore.Qt.black)
        self._pen.setWidth(2)

    def boundingRect(self) -> QtCore.QRectF:
        start: QtCore.QPointF = getCenterOf(self._start)
        end: QtCore.QPointF = self._getEndPos()
        return QtCore.QRectF(start, end).normalized().adjusted(-1, -1, 1, 1)

    def setEnd(self, end: Union[QtWidgets.QGraphicsObject, QtCore.QPointF]) -> None:
        self._end = end

    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        painter.setPen(self._pen)

        start: QtCore.QPointF = getCenterOf(self._start)
        end: QtCore.QPointF = self._getEndPos()
        control1: QtCore.QPointF = QtCore.QPointF()
        control1.setX(start.x())
        control1.setY(start.y() + (end.y() - start.y()) / 3)

        control2: QtCore.QPointF = QtCore.QPointF()
        control2.setX(end.x())
        control2.setY(end.y() - (end.y() - start.y()) / 3)

        cubicPath = QtGui.QPainterPath(start)
        cubicPath.cubicTo(control1, control2, end)
        painter.drawPath(cubicPath)

    def _getEndPos(self) -> QtCore.QPointF:
        pos: QtCore.QPointF = QtCore.QPointF()
        if type(self._end) in [Output, Input]:
            pos = getCenterOf(self._end)
        elif type(self._end) == QtCore.QPointF:
            pos = self._end
        elif type(self._end) == QtCore.QPoint:
            pos = QtCore.QPointF(self._end)
        else:
            raise TypeError(f'Type of end of line not understood: {type(self._end)}')
        return pos

    def getInput(self) -> 'Input':
        inp: 'Input' = self._start
        if type(self._end) == Input:
            inp = self._end
        return inp

    def getOutput(self) -> 'Output':
        out: 'Output' = self._end
        if type(self._start) == Output:
            out = self._start
        return out


class PreviewButton(QtWidgets.QGraphicsObject):
    diameter: float = 20  # px diameter of widget

    def __init__(self, parentNode: 'BaseNode'):
        super(PreviewButton, self).__init__()
        self._node: 'BaseNode' = parentNode
        self.setZValue(1)

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, 0, self.diameter, self.diameter)

    def mousePressEvent(self, event) -> None:
        self._node.pushPreview()

    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        brect: QtCore.QRectF = self.boundingRect()
        painter.setPen(QtCore.Qt.black)
        painter.setBrush(QtCore.Qt.darkGray)
        painter.drawEllipse(brect)

        painter.setPen(QtCore.Qt.white)
        painter.drawText(brect, QtCore.Qt.AlignCenter, 'P')


class DataType(Enum):
    """
    The variations of (spectral) data that can be accepted or returned by a node
    """
    CONTINUOUS = auto()
    DISCRETE = auto()


def recursiveIsConnectedToRGBInputNode(node: 'BaseNode') -> bool:
    """
    Tests, if the node is connected to the RGB Input Node.
    """

    def inputGoesToRGBNode(inp: 'Input') -> bool:
        goesToInput: bool = False
        connected = inp.getConnectedNode()
        if connected is not None:
            goesToInput = connected.isStartNode
        return goesToInput

    connectedToInput: bool = False
    for inp in node.getInputs():
        if inp.isConnected():
            connectedToInput = inputGoesToRGBNode(inp)
            if not connectedToInput:
                connectedToInput = recursiveIsConnectedToRGBInputNode(inp.getConnectedNode())
            if connectedToInput:
                break

    return connectedToInput


def getTransparentQLabel(text: str, bold: bool = False) -> QtWidgets.QLabel:
    _label = QtWidgets.QLabel(text)
    _label.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
    if bold:
        font: QtGui.QFont = QtGui.QFont()
        font.setBold(True)
        _label.setFont(font)
    return _label


def getCenterOf(inOutObj: Union['Input', 'Output']) -> QtCore.QPointF:
    center: QtCore.QPointF = QtCore.QPointF(inOutObj.pos().x() + inOutObj.diameter / 2,
                                            inOutObj.pos().y() + inOutObj.diameter / 2)
    return inOutObj.parentItem().mapToScene(center)
