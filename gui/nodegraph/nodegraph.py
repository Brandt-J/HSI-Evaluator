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
import os
from copy import copy
import json

from logger import getLogger
from projectPaths import getAppFolder
from gui.nodegraph.nodes import *
from gui.nodegraph.nodecore import *

if TYPE_CHECKING:
    from preprocessing.preprocessors import Preprocessor


class NodeGraph(QtWidgets.QGraphicsView):
    NewSpecsForSpecPlot: QtCore.pyqtSignal = QtCore.pyqtSignal(np.ndarray)
    NewSpecsForScatterPlot: QtCore.pyqtSignal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        """
        NodeGraph Object for Spectra Preprocessing.
        """
        super(NodeGraph, self).__init__()
        self.setMinimumSize(500, 250)
        scene = BackgroundScene(self)
        scene.setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)
        scene.setBackgroundBrush(QtCore.Qt.gray)
        self.setScene(scene)

        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self._zoom = 0

        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self._logger: 'Logger' = getLogger("NodeGraph")

        self._nodes: List['BaseNode'] = []  # Nodes, EXCEPT the start and end-Node
        self._connections: List['ConnectionWire'] = []
        self._selectedNode: Union[None, 'BaseNode'] = None

        self._dragFromOut: Union[None, 'Output'] = None
        self._dragFromIn: Union[None, 'Input'] = None
        self._tempConnection: Union[None, 'ConnectionWire'] = None
        self._contextMenu: QtWidgets.QMenu = QtWidgets.QMenu()
        self._addNodesMenu: QtWidgets.QMenu = QtWidgets.QMenu("Add Node")
        for nodeClassName, nodeClass in nodeTypes.items():
            if not nodeClass.isRequiredAndUnique:
                self._addNodesMenu.addAction(nodeClassName)

        self._contextMenu.addMenu(self._addNodesMenu)
        self._contextMenu.addSeparator()
        self._contextMenu.addAction("Save Graph")
        self._contextMenu.addAction("Load Graph")

        self.verticalScrollBar().valueChanged.connect(self.updateScene)

        self._inputNode: NodeStart = NodeStart(self, self._logger)
        self._nodeScatterPlot: NodeScatterPlot = NodeScatterPlot(self, self._logger)
        self._nodeSpecPlot: NodeSpecPlot = NodeSpecPlot(self, self._logger)
        self._nodeClf: NodeClassification = NodeClassification(self, self._logger)

        for i, node in enumerate(self._getRequiredNodes()):
            node.id = i
        self._addRequiredNodes()
        self._addMinimalSetup()

        self._fitToWindow()

    def getGraphConfig(self) -> List[dict]:
        """
        Returns a list of configurations representing the current node setup.
        """
        nodeDicts: List[dict] = []
        for node in self._nodes + self._getRequiredNodes():
            nodeDicts.append(node.toDict())
        return nodeDicts

    def applyGraphConfig(self, configList: List[dict]) -> None:
        """
        Takes a list of node configurations and sets up the current graph accordingly.
        """
        self._deleteAllNodesAndConnections()
        self._createNodesFromConfig(configList)
        self._createConnectionsFromConfig(configList)

    def setInputSpecta(self, spectra: np.ndarray) -> None:
        """
        Sets the spectra for the input node
        :param spectra: (NxM) array of N spectra with M wavelenghts
        """
        self._inputNode.setSpectra(spectra)

    def updatePlotNodes(self) -> None:
        """
        Runs the nodegraph so that both resultplots get updated (if connected).
        """

        if self._nodeSpecPlot.isConnectedToInput():
            preprocSpecs: np.ndarray = self._nodeSpecPlot.getOutput()
            self.NewSpecsForSpecPlot.emit(preprocSpecs)

        if self._nodeScatterPlot.isConnectedToInput():
            preprocSpecs: np.ndarray = self._nodeScatterPlot.getOutput()
            self.NewSpecsForScatterPlot.emit(preprocSpecs)

    def selectNode(self, node: 'BaseNode') -> None:
        if self._selectedNode is not None:
            self._selectedNode.deselect()
        node.select()
        self._selectedNode = node

    def getPreprocessors(self) -> List['Preprocessor']:
        """
        Returns a stack of Preprocessors for the current classficiation setup.
        """
        preprocStack: List['Preprocessor'] = []


        return preprocStack

    def clearNodeCache(self) -> None:
        """
        Clears the cache of all nodes.
        """
        for node in self._nodes:
            node.invalidateCache()

    def _createNodesFromConfig(self, nodeConfigs: List[dict]) -> None:
        """
        Takes a list of nodeconfigs and creates the nodes.
        """
        requiredNodeTypes = [type(node) for node in self._getRequiredNodes()]
        for config in nodeConfigs:
            nodeType: Type['BaseNode'] = nodeTypes[config["label"]]
            if nodeType not in requiredNodeTypes:
                newNode: 'BaseNode' = self._addNode(nodeType)
                newNode.id = config["id"]
                newNode.fromDict(config["params"])

    def _saveConfig(self, path: str) -> None:
        """
        Saves the current graph configuration to file
        :param path: The filepath where to save the config to.
        """
        with open(path, "w") as fp:
            json.dump(self.getGraphConfig(), fp)

    def _loadConfig(self, path: str) -> None:
        """
        Loads a configuration from file and sets the graph accordinly.
        :param path: The filepath where to load the config from.
        """
        assert os.path.exists(path), f"Cannot load config from {path}.\nFile does not exist."
        with open(path, "r") as fp:
            nodeConfigs: List[dict] = json.load(fp)
        self.applyGraphConfig(nodeConfigs)

    def _createConnectionsFromConfig(self, nodeConfigs: List[dict]) -> None:
        """
        Takes a list of nodeconfigs and creates the coresponding connections.
        ATTENTION: Nodes have to be already created using the _createNodesFromConfig Method!!!
        """
        for config in nodeConfigs:
            for inputID, ids in config["inputs"].items():
                inputID = int(inputID)
                connectedNodeID, outputID = int(ids[0]), int(ids[1])

                node: 'BaseNode' = self._getNodeOfID(int(config["id"]))
                inp: 'Input' = node.getInputs()[inputID]
                connectedNode: 'BaseNode' = self._getNodeOfID(connectedNodeID)
                output: 'Output' = connectedNode.getOutputs()[outputID]
                self._addConnection(inp, output)

    def _addRequiredNodes(self) -> None:
        """
        Adds the required nodes to the graph and positions them at reasonable places.
        """
        for node in self._getRequiredNodes():
            self.scene().addItem(node)
        self._nodeClf.setPos(0, 500)
        self._nodeSpecPlot.setPos(400, 500)
        self._nodeScatterPlot.setPos(200, 500)

    def _addMinimalSetup(self) -> None:
        """
        Creates a minimal working setup and connects nodes.
        """
        nodeDeriv: NodeSmoothDeriv = cast(NodeSmoothDeriv, self._addNode(NodeSmoothDeriv, QtCore.QPointF(0, 100)))
        self._addConnection(nodeDeriv._inputs[0], self._inputNode._outputs[0])

        nodeDimRed: NodeDimReduct = cast(NodeDimReduct, self._addNode(NodeDimReduct, QtCore.QPointF(0, 300)))
        self._addConnection(nodeDimRed._inputs[0], nodeDeriv._outputs[0])

        self._addConnection(self._nodeScatterPlot._inputs[0], nodeDimRed._outputs[0])
        self._addConnection(self._nodeSpecPlot._inputs[0], nodeDeriv._outputs[0])
        self._addConnection(self._nodeClf._inputs[0], nodeDimRed._outputs[0])

    def _deselectNode(self) -> None:
        self._selectedNode.deselect()
        self._selectedNode = None

    def _executeContextMenu(self, pos: QtCore.QPoint) -> None:
        action: QtWidgets.QAction = self._contextMenu.exec_(pos)
        if action:
            action = action.text()
            if action == "Save Graph":
                graphName, ok = QtWidgets.QInputDialog.getText(self, "Input Name",
                                                               "What Name should the procedure be saved at?")
                if ok:
                    savePath: str = os.path.join(self._getGraphSaveFolder(), graphName + '.graph')
                    self._saveConfig(savePath)

            elif action == "Load Graph":
                graphName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "What procedure to load?",
                                                                     self._getGraphSaveFolder(), "*.graph")
                if graphName:
                    loadPath: str = os.path.join(self._getGraphSaveFolder(), graphName)
                    self._loadConfig(loadPath)

            elif action in nodeTypes:
                nodeClass: Type['BaseNode'] = nodeTypes[action]
                pos: QtCore.QPointF = self.mapToScene(self.mapFromGlobal(pos))
                self._addNode(nodeClass, pos)

    def _getGraphSaveFolder(self) -> str:
        """
        Returns a folder in which graphs can be saved.
        """
        path: str = os.path.join(getAppFolder(), "PreprocessingGraphs")
        os.makedirs(path, exist_ok=True)
        return path

    def _getClassificationPath(self) -> List['BaseNode']:
        """
        Gets the path of Nodes starting from the Spectra Input node to the Classification Node. Returns an empty list
        if there is no valid connection.
        """
        def allNodesVisited(visited: List['BaseNode']) -> bool:
            return len(visited) == len(self._nodes) + len(self._getRequiredNodes())

        nodePath: List['BaseNode'] = []
        if self._nodeClf.isConnectedToInput():
            startReached: bool = False
            newNodeFound: bool = True
            visitedNodes: List['BaseNode'] = []

            paths: List[List['BaseNode']] = [[self._nodeClf]]
            while not startReached and not allNodesVisited(visitedNodes) and newNodeFound:
                newNodeFound = False
                for curPath in paths:
                    curNode: 'BaseNode' = curPath[-1]
                    linkedNodes: List['BaseNode'] = self._getConnectedNodes(curNode)
                    if len(linkedNodes) > 0:
                        node: 'BaseNode' = linkedNodes[0]
                        if node not in visitedNodes:
                            curPath.append(node)
                            visitedNodes.append(node)
                            newNodeFound = True
                            if type(node) == NodeStart:
                                startReached = True
                                nodePath = curPath[::-1]
                                len(curPath)
                                break
                            linkedNodes.remove(node)

                        # attach a copy of the unfinished path for each remaining linked node
                        for _ in range(len(linkedNodes)):
                            paths.append(copy(curPath))

        return nodePath

    def _getNodeOfID(self, nodeID: int) -> 'BaseNode':
        wantedNode: 'BaseNode' = None
        for node in self._nodes + self._getRequiredNodes():
            if node.id == nodeID:
                wantedNode = node
                break
        assert wantedNode is not None, f'Node of id {nodeID} does not exist'
        return wantedNode

    def _addNode(self, nodeClass: Type['BaseNode'], pos: QtCore.QPointF = None) -> 'BaseNode':
        assert nodeClass in nodeTypes.values(), f"Requested nodeType {nodeClass} does not exist."
        if pos is None:
            inputHeight: float = self._inputNode.preferredHeight()
            minSpace: float = 50
            pos: QtCore.QPointF = QtCore.QPointF(0, inputHeight + minSpace)
            for node in self._nodes:
                nodeHeight = node.preferredHeight()
                diff: QtCore.QPointF = pos - node.pos()
                if diff.manhattanLength() < minSpace + nodeHeight:
                    pos += QtCore.QPointF(0, minSpace + nodeHeight)

        newNode: 'BaseNode' = nodeClass(self, self._logger, pos=pos)
        newNode.id = self._getNewNodeID()

        self._nodes.append(newNode)
        self.scene().addItem(newNode)
        return newNode



    def _deleteAllNodesAndConnections(self) -> None:
        """
        Deletes all nodes but the required ones, and also all connections.
        """
        for node in reversed(self._nodes):
            self._deleteNode(node)
        for wire in reversed(self._connections):
            self._removeConnection(wire)

    def _deleteNode(self, node: 'BaseNode') -> None:
        if node.isRequiredAndUnique:
            QtWidgets.QMessageBox.about(self, "Warning", "This node is required and cannot be removed.")

        else:
            assert node in self._nodes, f'Requested not present node to delete: {node}'
            for inp in node.getInputs():
                if inp.isConnected():
                    self.capConnectionFrom(inp, drawNewConnection=False)

            for otherNode in self._nodes + self._getRequiredNodes():
                for inp in otherNode.getInputs():
                    if inp.getConnectedNode() is node:
                        self.capConnectionFrom(inp, drawNewConnection=False)

            self.scene().removeItem(node)
            self._nodes.remove(node)

    def _getNewNodeID(self) -> int:
        """Get's a new unique Node id"""
        curIDs: List[int] = [node.id for node in self._nodes + self._getRequiredNodes()]
        i: int = 0
        while True:
            if i in curIDs:
                i += 1
            else:
                break
        return i

    def _getRequiredNodes(self) -> List['BaseNode']:
        """
        Gets a list of the nodes that are always required for the nodegraph
        """
        return [self._inputNode, self._nodeSpecPlot, self._nodeScatterPlot, self._nodeClf]

    def disableAllNodes(self) -> None:
        for node in self._nodes + [self._resultNode, self._inputNode]:
            node.disable()

    def enableAllNodes(self) -> None:
        for node in self._nodes + [self._resultNode, self._inputNode]:
            node.enable()

    def dragConnectionTo(self, inputObj: 'Input') -> None:
        self._dragFromIn = inputObj
        self._tempConnection = ConnectionWire(inputObj, inputObj)
        self.scene().addItem(self._tempConnection)

    def dragConnectionFrom(self, output: 'Output') -> None:
        self._dragFromOut = output
        self._tempConnection = ConnectionWire(output, output)
        self.scene().addItem(self._tempConnection)

    def capConnectionFrom(self, inputObj: 'Input', drawNewConnection: bool = True) -> None:
        wire: 'ConnectionWire' = self._getConnectionOfInputObj(inputObj)
        inputObj.disconnect()
        output: 'Output' = wire.getOutput()
        self._removeConnection(wire)
        if self._tempConnection is None and drawNewConnection:
            self.dragConnectionFrom(output)

    def updateScene(self) -> None:
        self.scene().update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Delete and self._selectedNode is not None:
            self._deleteNode(self._selectedNode)
        else:
            super(NodeGraph, self).keyPressEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.RightButton:
            self._executeContextMenu(self.mapToGlobal(event.pos()))
        else:
            if event.button() == QtCore.Qt.LeftButton and self._selectedNode is not None:
                self._deselectNode()
            super(NodeGraph, self).mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._tempConnection is not None:
            self._tempConnection.setEnd(self.mapToScene(event.pos()))
            self.scene().update()
        else:
            super(NodeGraph, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._tempConnection is not None:
            inpuOrOutput: Union[Input, Output] = self._getOverInputOutput(self.mapToScene(event.pos()))
            self._tryConnectingNodes(inpuOrOutput)
            self._destroyTempConnection()
        else:
            super(NodeGraph, self).mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.25
            self._zoom += 1
        else:
            factor = 0.8
            self._zoom -= 1

        if self._zoom > 0:
            self.scale(factor, factor)
        elif self._zoom == 0:
            self._fitToWindow()
        else:
            self._zoom = 0

    def _fitToWindow(self, marginFactor: float = 0.3):
        """
        Fits the window to show the entire sample.
        :param marginFactor: The view is scaled, so that there is an empty margin by this factor around the
        visible elements in the scene.
        :return:
        """
        brect = self.scene().itemsBoundingRect()
        self.fitInView(-brect.width() * marginFactor,
                       -brect.height() * marginFactor,
                       brect.width() * (1+(2*marginFactor)),
                       brect.height()*(1+(2*marginFactor)),
                       QtCore.Qt.KeepAspectRatio)

    def _tryConnectingNodes(self, inOrOut: Union[Input, Output]) -> None:
        if inOrOut is not None:
            inp, outp = None, None
            if type(inOrOut) == Input and self._dragFromOut is not None:
                inp, outp = inOrOut, self._dragFromOut
            elif type(inOrOut) == Output and self._dragFromIn is not None:
                inp, outp = self._dragFromIn, inOrOut

            if inp is not None and outp is not None:
                if inp.isCompatibleToOutput(outp):
                    if inp.isConnected():
                        self.capConnectionFrom(inp)
                    self._addConnection(inp, outp)
                else:
                    QtWidgets.QMessageBox.warning(self, "Incompatible Sockets",
                                                  f"cannot connect {inp.getCompatibleDataTypes()} to {outp.getDataType()}")

    def _addConnection(self, inp: 'Input', outp: 'Output') -> None:
        connectSucess: bool = outp.connectTo(inp)
        assert connectSucess, f'Could not input type {inp.getCompatibleDataTypes()} to output type {outp.getDataType()}'
        newConn: ConnectionWire = ConnectionWire(inp, outp)
        self._connections.append(newConn)
        self.scene().addItem(newConn)

    def _removeConnection(self, connectionWire: 'ConnectionWire') -> None:
        self._connections.remove(connectionWire)
        self.scene().removeItem(connectionWire)

    def _destroyTempConnection(self) -> None:
        self.scene().removeItem(self._tempConnection)
        self._tempConnection = None
        self._dragFromIn = self._dragFromOut = None
        self.scene().update()

    def _getOverInputOutput(self, scenePos: QtCore.QPointF) -> Union[None, Input, Output]:
        """
        Determines, if an input, an output or neither (None) is found at the given scene Position.
        """
        isOver: Union[None, Input, Output] = None
        for node in self._nodes + self._getRequiredNodes():
            isOver = node.getInputOrOutputAtPos(scenePos)
            if isOver is not None:
                break
        return isOver

    def _getConnectionOfInputObj(self, inputObj: 'Input') -> 'ConnectionWire':
        wire: 'ConnectionWire' = None
        for curWire in self._connections:
            if curWire.getInput() is inputObj:
                wire = curWire
                break
        assert wire is not None, 'Wire to requested input could not be found.'
        return wire

    def _getConnectedNodes(self, node: 'BaseNode') -> List['BaseNode']:
        nodes: List['BaseNode'] = []
        for inp in node.getInputs():
            if inp.isConnected():
                nodes.append(inp.getConnectedNode())
        return nodes


class BackgroundScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent: QtWidgets.QGraphicsView):
        super(BackgroundScene, self).__init__(parent)
        self._vscrollbar: QtWidgets.QScrollBar = parent.verticalScrollBar()
        self._hscrollbar: QtWidgets.QScrollBar = parent.horizontalScrollBar()
        self._minX: int = 0
        self._minY: int = 0
        self._gridSize: int = 50

    def drawBackground(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        painter.setBrush(QtCore.Qt.gray)
        painter.setPen(QtCore.Qt.gray)
        painter.drawRect(rect)
        painter.setPen(QtCore.Qt.darkGray)

        if rect.left() < self._minX:
            self._minX = rect.left()
        if rect.top() < self._minY:
            self._minY = rect.top()

        x, y = int(self._minX), int(self._minY)
        while y < self._getTotalHeight(rect.height()):
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)
            y += self._gridSize

        while x < self._getTotalWidth(rect.width()):
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
            x += self._gridSize

    def _getTotalHeight(self, rectHeight: float) -> int:
        scrollHeight = self._vscrollbar.maximum() - self._vscrollbar.minimum() + self._vscrollbar.pageStep()
        return int(max([rectHeight, scrollHeight]))

    def _getTotalWidth(self, rectWidth: float) -> int:
        scrollWidth = self._hscrollbar.maximum() - self._hscrollbar.minimum() + self._hscrollbar.pageStep()
        return int(max([rectWidth, scrollWidth]))
