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
from enum import Enum
import numpy as np

if TYPE_CHECKING:
    from preprocessing.preprocessors import Preprocessor
    from logging import Logger
    from gui.nodegraph.nodegraph import NodeGraph


class BaseNode(QtWidgets.QGraphicsWidget):
    """
    Base Node Class for Spectra Processing.
    """
    label = 'BaseNode'
    defaultParams: dict = {}
    isRequiredAndUnique: bool = False  # if yes, this node will be created automatically and cannot be deleted or added again.
    ParamsChanged: QtCore.pyqtSignal = QtCore.pyqtSignal()

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(BaseNode, self).__init__()
        self._parentGraph: 'NodeGraph' = nodeGraphParent
        self._inputs: List['Input'] = []
        self._outputs: List['Output'] = []
        self._bodywidget: Union[None, QtWidgets.QWidget] = None
        self._logger: 'Logger' = logger
        self._preprocessor: Union[None, 'Preprocessor'] = None

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
        self._bodyGradient: Union[None, QtGui.QLinearGradient] = None

    def _populateLayoutAndCreateIO(self) -> None:
        def getGroupBox(layout: QtWidgets.QLayout) -> QtWidgets.QGroupBox:
            _group: QtWidgets.QGroupBox = QtWidgets.QGroupBox()
            _group.setFlat(True)
            _group.setStyleSheet("border: 0;")
            _group.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            _group.setLayout(layout)
            return _group

        scene: QtWidgets.QGraphicsScene = self._parentGraph.scene()
        bodyGroup: QtWidgets.QGroupBox = getGroupBox(QtWidgets.QVBoxLayout())
        bodyGroup.layout().addWidget(getTransparentQLabel(self.label, bold=True))
        if self._bodywidget is not None:
            bodyGroup.layout().addWidget(self._bodywidget)

        self._layout.addItem(scene.addWidget(bodyGroup))
        self._origLayoutSize = self.preferredSize()
        self._updateGradients()
        self._addInputOutputItems()

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

    def fromDict(self, paramsDict: dict) -> None:
        """
        Used to restore configuration from the dict. Only the "params" sub-dictionary is passed in.
        """
        pass

    def invalidateCache(self) -> None:
        """
        Can be overloaded to erase any temporarily cached results.
        """
        pass

    def isConnectedToInput(self) -> bool:
        return recursiveIsConnectedToInputNode(self)

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
        self._bodyGradient = self._getBodyGradient(selected)

    def _addInputOutputItems(self) -> None:
        middleX: float = self.preferredWidth()/2 - Input.diameter/2
        for inp in self._inputs:
            # xPos += (inp.getLabelWidget().size().width() + inp.diameter) / 2
            inp.setParentItem(self)
            inp.setPos(middleX, 0)
            inp.capConnection.connect(self._parentGraph.capConnectionFrom)

        for outp in self._outputs:
            diameter = outp.diameter
            yPos = int(self.boundingRect().height() - diameter)
            outp.setParentItem(self)
            outp.setPos(middleX, yPos)
            outp.dragConnection.connect(self._startDragConnection)

    def _getBodyGradient(self, selected: bool = False) -> QtGui.QLinearGradient:
        grad: QtGui.QLinearGradient = QtGui.QLinearGradient(0, 0, self.preferredWidth(), 0)
        if selected:
            if self.isRequiredAndUnique:
                grad.setColorAt(0, QtGui.QColor(255, 233, 230))
                grad.setColorAt(0.2, QtGui.QColor(255, 180, 180))
                grad.setColorAt(1, QtGui.QColor(128, 128, 128))
            else:
                grad.setColorAt(0, QtGui.QColor(230, 255, 0))
                grad.setColorAt(0.2, QtGui.QColor(180, 255, 0))
                grad.setColorAt(1, QtGui.QColor(128, 128, 0))
        else:
            if self.isRequiredAndUnique:
                grad.setColorAt(0, QtGui.QColor(255, 200, 200))
                grad.setColorAt(0.2, QtGui.QColor(255, 128, 128))
                grad.setColorAt(1, QtGui.QColor(128, 100, 100))
            else:
                grad.setColorAt(0, QtGui.QColor(200, 255, 0))
                grad.setColorAt(0.2, QtGui.QColor(128, 255, 0))
                grad.setColorAt(1, QtGui.QColor(64, 128, 0))
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

    def getPreprocessor(self) -> Union[None, 'Preprocessor']:
        return self._preprocessor

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
        painter.setBrush(self._bodyGradient)

        path = QtGui.QPainterPath()
        path.addRoundedRect(self.boundingRect(), 10, 10)
        painter.drawPath(path)


class Input(QtWidgets.QGraphicsObject):
    """
    Object representing an Node Input.
    """
    diameter: float = 15  # px diameter of widget
    capConnection: QtCore.pyqtSignal = QtCore.pyqtSignal(QtWidgets.QGraphicsObject)

    def __init__(self, name: str, dataTypes: List['DataType']):
        """
        :param name: Name of the input to display
        :param dataTypes: Type that the input can accept.
        """
        super(Input, self).__init__()
        self.name: str = name
        self._types: List[DataType] = dataTypes
        self._connectedOutput: Union[None, 'Output'] = None
        self._color: QtGui.QColor = QtCore.Qt.black
        self.setZValue(1)
        self._setInputColor()

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

    def isCompatibleToOutput(self, output: 'Output') -> bool:
        """
        Checks, whether the input is compatible to the given output.
        :param output: The output object to check for.
        """
        isCompatible: bool = False
        if output.getDataType() in self._types:
            isCompatible = True
        return isCompatible

    def getCompatibleDataTypes(self) -> List['DataType']:
        """
        Returns the datatypes this input is compatible to.
        """
        return self._types

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
        assert otherOutput.getDataType() in self._types
        self._connectedOutput = otherOutput

    def disconnect(self) -> None:
        self._connectedOutput = None

    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        painter.setPen(QtCore.Qt.black)
        painter.setBrush(self._color)
        painter.drawEllipse(self.boundingRect())

    def _setInputColor(self) -> None:
        """
        Sets a color that is then used for drawing, representing the datatypes.
        """
        if len(self._types) == 1:
            rgb: np.ndarray = self._types[0].getColor().astype(int)
            self._color: QtGui.QColor = QtGui.QColor(rgb[0], rgb[1], rgb[2])
        else:
            rgb: np.ndarray = np.zeros(3)
            for dtype in self._types:
                rgb += dtype.getColor()
            rgb: np.ndarray = rgb / len(self._types)
            rgb = rgb.astype(int)
            self._color: QtGui.QColor = QtGui.QColor(rgb[0], rgb[1], rgb[2])


class Output(QtWidgets.QGraphicsObject):
    dragConnection: QtCore.pyqtSignal = QtCore.pyqtSignal(QtWidgets.QGraphicsObject)
    diameter: float = 15  # px diameter of widget

    def __init__(self, parentNode: 'BaseNode', name: str, dataType: 'DataType'):
        super(Output, self).__init__()
        self.name: str = name
        self._node: 'BaseNode' = parentNode
        self._type: DataType = dataType
        rgb: np.ndarray = dataType.getColor()
        self._color: QtGui.QColor = QtGui.QColor(rgb[0], rgb[1], rgb[2])
        self.setZValue(1)

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, 0, self.diameter, self.diameter)

    def mousePressEvent(self, event) -> None:
        self.dragConnection.emit(self)

    def getValue(self) -> object:
        return self._node.getOutput(self.name)

    def getDataType(self) -> 'DataType':
        return self._type

    def getNode(self) -> 'BaseNode':
        return self._node

    def connectTo(self, otherInput: 'Input') -> bool:
        success: bool = False
        if otherInput.isCompatibleToOutput(self):
            otherInput.acceptOutputConnection(self)
            success = True
        return success

    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        painter.setPen(QtCore.Qt.black)
        painter.setBrush(self._color)
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
    CONTINUOUS = 0
    DISCRETE = 1

    def getColor(self) -> np.ndarray:
        """
        Returns an RGB color representing the data type.
        """
        color: np.ndarray = np.zeros(3)
        if self.value == 0:
            color = np.array([150, 255, 150])
        elif self.value == 1:
            color = np.array([150, 150, 255])
        return color


def recursiveIsConnectedToInputNode(node: 'BaseNode') -> bool:
    """
    Tests, if the node is connected to the RGB Input Node.
    """

    def inputGoesToStartNode(inp: 'Input') -> bool:
        goesToInput: bool = False
        connected = inp.getConnectedNode()
        if connected is not None:
            goesToInput = connected.isStartNode
        return goesToInput

    connectedToInput: bool = False
    for inp in node.getInputs():
        if inp.isConnected():
            connectedToInput = inputGoesToStartNode(inp)
            if not connectedToInput:
                connectedToInput = recursiveIsConnectedToInputNode(inp.getConnectedNode())
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
