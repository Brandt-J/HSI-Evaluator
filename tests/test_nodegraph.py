import unittest
import os
import logging
import tempfile
import copy
from PyQt5 import QtWidgets
import sys

from gui.nodegraph.nodes import *
from gui.nodegraph.nodegraph import NodeGraph


class TestNodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)

    def setUp(self) -> None:
        self.logger: logging.Logger = logging.getLogger('TestLogger')
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.DEBUG)
        self.nodegraph: NodeGraph = NodeGraph()

    def tearDown(self) -> None:
        del self.logger
        del self.nodegraph

    def test_connectNodes(self) -> None:
        startNode: NodeStart = self.nodegraph._inputNode
        scatterPlot: NodeScatterPlot = self.nodegraph._nodeScatterPlot

        self.nodegraph._addConnection(scatterPlot._inputs[0], startNode._outputs[0])

    # def test_getNumberOfNodes(self):
    #     numNodes = self.nodegraph.getNumberOfNodes()
    #     self.assertEqual(numNodes, 5)
    #
    #     # Now add a number node to the Threshold node's low-input
    #     self.nodegraph._addNode(NodeNumber)
    #     numNode: NodeNumber = self.nodegraph._nodes[-1]
    #     threshNode: NodeThreshold = self.nodegraph._nodes[1]
    #     self.nodegraph._addConnection(threshNode._inputs[1], numNode._outputs[0])
    #     numNodes = self.nodegraph.getNumberOfNodes()
    #     self.assertEqual(numNodes, 5)
    #
    #     # And now another one to the high-input
    #     self.nodegraph._addNode(NodeNumber)
    #     newNodeNumber: NodeNumber = self.nodegraph._nodes[-1]
    #     self.nodegraph._addConnection(threshNode._inputs[2], newNodeNumber._outputs[0])
    #     numNodes = self.nodegraph.getNumberOfNodes()
    #     self.assertEqual(numNodes, 5)
    #
    #     # And now add another not connected node:
    #     self.nodegraph._addNode(NodeMath)
    #     numNodes = self.nodegraph.getNumberOfNodes()
    #     self.assertEqual(numNodes, 5)
    #
    #     # Now disconnect the start node, it should return -1:
    #     self.nodegraph.capConnectionFrom(self.nodegraph._nodes[0]._inputs[0])
    #     numNodes = self.nodegraph.getNumberOfNodes()
    #     self.assertEqual(numNodes, -1)
    #
    # def test_isConnectedToStartAndGetNodePath(self):
    #     nodePath: List['BaseNode'] = self.nodegraph._getNodePath()
    #     expectedPath: List['BaseNode'] = [self.inpNode, self.grayNode, self.threshNode, self.watershedNode, self.resultNode]
    #     self.assertEqual(nodePath, expectedPath)
    #
    #     self.nodegraph._deleteNode(self.threshNode)
    #     self.assertTrue(self.grayNode._isConnectedToRGBInput())
    #     self.assertFalse(self.watershedNode._isConnectedToRGBInput())
    #     nodePath = self.nodegraph._getNodePath()
    #     self.assertEqual(nodePath, [])

    # def test_getNodeDict(self):
    #     self.createMinimalWatershedSetup()
    #     threshDict: dict = self.threshNode.toDict()
    #     self.assertEqual(threshDict["label"], self.threshNode.label)
    #     self.assertEqual(threshDict["id"], self.threshNode.id)
    #
    #     inputs: dict = threshDict["inputs"]
    #     self.assertEqual(list(inputs.keys()), [0])  # the first input has a connection
    #     self.assertEqual(inputs[0], [self.grayNode.id, 0])  # it is connected to the first output of the gray node, specified by its id
    #
    #     params: dict = threshDict["params"]
    #     self.assertEqual(params["threshLow"], self.threshNode._threshLow.value())
    #     self.assertEqual(params["threshHigh"], self.threshNode._threshHigh.value())
    #
    # def test_deleteALlNodes(self):
    #     self.createMinimalWatershedSetup()
    #     self.nodegraph._deleteAllNodesAndConnections()
    #     self.assertEqual(len(self.nodegraph._nodes), 0)
    #     self.assertEqual(len(self.nodegraph._connections), 0)
    #
    # def test_saveLoadNodeConfig(self) -> None:
    #     self.createMinimalWatershedSetup()
    #
    #     with tempfile.TemporaryDirectory() as tmpDir:
    #         origThresh = self.threshNode._threshLow.value()
    #         origNodePath: List['BaseNode'] = copy.copy(self.nodegraph._getNodePath())
    #         self.threshNode._threshLow.setValue(2*origThresh)
    #         savePath: str = os.path.join(tmpDir, 'nodeConfig.txt')
    #         self.nodegraph.saveConfig(savePath)
    #         self.assertTrue(os.path.exists(savePath))
    #
    #         self.threshNode._threshLow.setValue(0)  # set to something else..
    #         self.nodegraph.loadConfig(savePath)
    #         self.assertEqual(len(origNodePath), self.nodegraph.getNumberOfNodes())
    #         for origNode, newNode in zip(origNodePath, self.nodegraph._getNodePath()):
    #             self.assertEqual(type(origNode), type(newNode))
    #             self.assertEqual(origNode.id, newNode.id)
    #             if type(newNode) == NodeThreshold:
    #                 self.assertEqual(newNode._threshLow.value(), 2*origThresh)
    #
    #
