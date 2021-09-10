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
from copy import copy
import json
import os

from logger import getLogger
from gui.nodegraph.nodes import *
from gui.nodegraph.nodecore import *


class NodeGraph(QtWidgets.QGraphicsView):
    def __init__(self):
        """
        NodeGraph Object for Spectra Preprocessing.
        """
        super(NodeGraph, self).__init__()
        self.setMinimumSize(700, 300)
        scene = BackgroundScene(self)
        scene.setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)
        scene.setBackgroundBrush(QtCore.Qt.gray)
        self.setScene(scene)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self._logger: 'Logger' = getLogger("NodeGraph")

        self._nodes: List['BaseNode'] = []  # Nodes, EXCEPT the start and end-Node
        self._connections: List['ConnectionWire'] = []
        self._selectedNode: 'BaseNode' = None

        self._dragFromOut: 'Output' = None
        self._dragFromIn: 'Input' = None
        self._tempConnection: 'ConnectionWire' = None
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
        self._inputNode.id = 0
        self._nodeScatterPlot: NodeScatterPlot = NodeScatterPlot(self, self._logger)
        self._nodeSpecPlot: NodeSpecPlot = NodeSpecPlot(self, self._logger)
        self._nodeClf: NodeClassification = NodeClassification(self, self._logger)

        self._addRequiredNodes()
        self._addNode(NodeSNV, QtCore.QPointF(50, 50))
        self._addNode(NodeDimReduct, QtCore.QPointF(100, 100))

    def _addRequiredNodes(self) -> None:
        """
        Adds the required nodes to the graph and positions them at reasonable places.
        """
        for node in self._getRequiredNodes():
            self.scene().addItem(node)
        self._nodeClf.setPos(0, 400)
        self._nodeSpecPlot.setPos(200, 400)
        self._nodeScatterPlot.setPos(400, 400)

    def saveConfig(self, path: str) -> None:
        nodeConfig: List[dict] = [node.toDict() for node in self._nodes + [self._inputNode, self._resultNode]]
        with open(path, "w") as fp:
            json.dump(nodeConfig, fp)

    def loadConfig(self, path: str) -> None:
        raise NotImplementedError
        # self._deleteAllNodesAndConnections()
        # with open(path, "r") as fp:
        #     nodeConfigs: List[dict] = json.load(fp)

        # self._createNodesFromConfig(nodeConfigs)
        # self._createConnectionsFromConfig(nodeConfigs)

    def _createNodesFromConfig(self, nodeConfigs: List[dict]) -> None:
        """
        Takes a list of nodeconfigs and creates the nodes.
        """
        raise NotImplementedError
        # for config in nodeConfigs:
        #     nodeType: Type['BaseNode'] = nodeTypes[config["label"]]
        #     if nodeType not in [NodeRGBInput, NodeProcessContours]:
        #         newNode: 'BaseNode' = self._addNode(nodeType)
        #         newNode.id = config["id"]
        #         newNode.fromDict(config)

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

    def selectNode(self, node: 'BaseNode') -> None:
        if self._selectedNode is not None:
            self._selectedNode.deselect()
        node.select()
        self._selectedNode = node

    def _deselectNode(self) -> None:
        self._selectedNode.deselect()
        self._selectedNode = None

    def getDetectionRunning(self) -> bool:
        return self.detectionRunning

    def runOnPreview(self) -> None:
        self._logger.debug('Starting particle detection on preview image')
        self.prepareForDetection()
        numNodes: int = self.getNumberOfNodes()
        if numNodes == -1:
            QtWidgets.QMessageBox.warning(self, "Invalid Node Setup",
                                          "There is no connection between input and measure points node.\n"
                                          "Please fix node setup and retry,")
        else:
            contours, measPoints = self._resultNode.getOutput()
            prevSize: int = self._detectParent.getPreviewSize()
            overlayImg: np.ndarray = np.zeros((prevSize, prevSize, 3), dtype=np.uint8)

            for cnt in contours:
                cv2.drawContours(overlayImg, [cnt], -1, [128, 255, 128], thickness=1)  # outline
                cv2.drawContours(overlayImg, [cnt], -1, [0, 255, 0], thickness=-1)  # fill

            for measpoints in measPoints:
                for point in measpoints:
                    cv2.drawMarker(overlayImg, (point.x, point.y), [128, 128, 255], markerType=cv2.MARKER_SQUARE,
                                   markerSize=5, thickness=3)

            self._detectParent.updatePreviewOverlay(overlayImg)
            for node in self._nodes:
                node.invalidateCache()

    def runOnFullImage(self, dataset: 'DataSet', logger: 'Logger' = None) -> bool:
        if logger is not None:
            oldLogger = self._logger
            self._logger = logger

        success: bool = False
        self._logger.info(f'Starting particle detection on full image, detectunning: {self.detectionRunning}')
        fullimage = self._detectParent.getFullImage()
        self.prepareForDetection(fullimage)
        try:
            self._resultNode.getOutput(dataset=dataset, fullimg=fullimage)
            success = True
        except Exception as e:
            self._logger.critical(f"Running detection on fullimage failed: {e}")

        if logger is not None:
            self._logger = oldLogger
        return success

    def prepareForDetection(self, fullimg: np.ndarray = None) -> None:
        if fullimg is None:
            self._setSeedsToNodes(preview=True)
            prevImg: np.ndarray = self._detectParent.getPreviewImage()
            self._inputNode.setInputImage(prevImg)
        else:
            self._setSeedsToNodes(preview=False)
            self._inputNode.setInputImage(fullimg)

    @QtCore.pyqtSlot(np.ndarray)
    def _showNodePreview(self, prevImg: np.ndarray) -> None:
        self._detectParent.updatePreviewOverlay(prevImg)

    def _executeContextMenu(self, pos: QtCore.QPoint) -> None:
        action: QtWidgets.QAction = self._contextMenu.exec_(pos)
        if action:
            action = action.text()
            if action == "Save Graph":
                if self._detectParent is not None:
                    savePath: str = self._detectParent.getDetectGraphSavePath(default=False)
                    if savePath is not None:
                        self.saveConfig(savePath)

            elif action == "Load Graph":
                if self._detectParent is not None:
                    loadPath: str = self._detectParent.getDetectGraphLoadPath(default=False)
                    if os.path.exists(loadPath):
                        self.loadConfig(loadPath)

            elif action == "Load Default Watershed":
                self._getBasicWatershedSetup()

            elif action == "Load Default Neural Net":
                self._getNeuralNetSetup()

            elif action in nodeTypes:
                nodeClass: Type['BaseNode'] = nodeTypes[action]
                pos: QtCore.QPointF = self.mapToScene(self.mapFromGlobal(pos))
                self._addNode(nodeClass, pos)

    def _setSeedsToNodes(self, preview: bool) -> None:
        """
        If a watershed node is in the nodegraph, it gets passed references to the seedpoints from preview or fullimage,
        according to the preview-flag.
        """
        for node in self._nodes:
            if type(node) in [NodeWatershed, NodeThreshold]:
                if preview:
                    prevWidget: 'DetectionPreview' = self._detectParent.getPreviewWiget()
                    seedPoints: np.ndarray = np.array(prevWidget.seedpoints)
                    seedDeletePoints: np.ndarray = np.array(prevWidget.seeddeletepoints)
                else:
                    seedPoints: np.ndarray = self._detectParent.getDatasetSeedPoints()
                    seedDeletePoints: np.ndarray = self._detectParent.getDatasetSeedDeletePoints()

                node.setSeedPoints(seedPoints, seedDeletePoints)

    def getPixelScale(self) -> float:
        return self._detectParent.getPixelScale()

    def getDetectionScale(self) -> float:
        return self._detectParent.getDetectScaleFactor()

    def getPreviewSize(self) -> int:
        """Gets value of width and height of preview image. It's always squared..."""
        return self._detectParent.getPreviewSize()

    def getPreviewSourceImage(self) -> np.ndarray:
        return self._detectParent.getPreviewImage()

    def getNumberOfNodes(self) -> int:
        """
        Returns the number of nodes in the current node pipeline, i.e., between input and output (ProcessContours) node.
        If "-1" is returned, there is no connection
        """
        nodePath: List['BaseNode'] = self._getNodePath()
        if len(nodePath) == 0:
            numNodes = -1
        else:
            numNodes = len(nodePath)
        return numNodes

    def _getNodePath(self) -> List['BaseNode']:
        """
        Gets the path of Nodes starting from the RGB Input node to the Process Contours Node. Returns an empty list
        if there is no valid connection.
        """
        def allNodesVisited(visited: List['BaseNode']) -> bool:
            return len(visited) == len(self._nodes) + 2

        nodePath: List['BaseNode'] = []
        startReached: bool = False
        newNodeFound: bool = True
        visitedNodes: List['BaseNode'] = []

        paths: List[List['BaseNode']] = [[self._resultNode]]
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
                        if type(node) == NodeRGBInput:
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
        for node in self._nodes + [self._inputNode, self._resultNode]:
            if node.id == nodeID:
                wantedNode = node
                break
        assert wantedNode is not None, f'Node of id {nodeID} does not exist'
        return wantedNode

    def _getBasicWatershedSetup(self) -> None:
        self._deleteAllNodesAndConnections()

        self._addNode(NodeGrayscale)
        self._addNode(NodeContrastCurve)
        self._addNode(NodeBlur)
        self._addNode(NodeThreshold)
        self._addNode(NodeWatershed)

        self._addConnection(self._nodes[0].getInputs()[0], self._inputNode.getOutputs()[0])
        for i in range(len(self._nodes) - 1):
            self._addConnection(self._nodes[i + 1].getInputs()[0], self._nodes[i].getOutputs()[0])
        self._addConnection(self._resultNode.getInputs()[0], self._nodes[-1].getOutputs()[0])

        watershednode: NodeWatershed = cast(NodeWatershed, self._nodes[-1])
        self._resultNode.setPos(0, watershednode.pos().y() + watershednode.preferredHeight() + 50)
        self._detectParent.enableAutoUpdateDetection()

    def _getNeuralNetSetup(self) -> None:
        self._deleteAllNodesAndConnections()

        neuralNetNode: NodeNeuralNet = cast(NodeNeuralNet, self._addNode(NodeNeuralNet))

        self._addConnection(neuralNetNode.getInputs()[0], self._inputNode.getOutputs()[0])
        self._addConnection(self._resultNode.getInputs()[0], neuralNetNode.getOutputs()[0])

        self._resultNode.setPos(0, neuralNetNode.pos().y() + neuralNetNode.preferredHeight() + 50)
        self._detectParent.disableAutoUpdateDetection()

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
        for node in reversed(self._nodes):
            self._deleteNode(node)
        for wire in reversed(self._connections):
            self._removeConnection(wire)

    def _deleteNode(self, node: 'BaseNode') -> None:
        if node.isRequiredAndUnique:
            QtWidgets.QMessageBox.about(self, "Warning", "This node is required and cannot be removed.")

        else:

            assert node in self._nodes, f'Requested not present node to delete: {node}'
            if self._detectParent is not None:
                node.detectionState.disconnect()
                node.parameterChanged.disconnect()

            for inp in node.getInputs():
                if inp.isConnected():
                    self.capConnectionFrom(inp, drawNewConnection=False)

            for otherNode in self._nodes:
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
                                                  f"cannot connect {inp.getCompatibleDataTypes()} to {outp.getCompatibleDataTypes()}")

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
