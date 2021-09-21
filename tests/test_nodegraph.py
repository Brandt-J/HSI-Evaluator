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
        specPlot: NodeSpecPlot = self.nodegraph._nodeSpecPlot

        self.nodegraph._addConnection(specPlot._inputs[0], startNode._outputs[0])

    def test_isConnectedToStartAndGetNodePath(self):
        nodeSmoothDeriv: BaseNode = self.nodegraph._nodes[0]
        self.assertTrue(type(nodeSmoothDeriv) == NodeSmoothDeriv)
        nodeDimReduct: BaseNode = self.nodegraph._nodes[1]
        self.assertTrue(type(nodeDimReduct) == NodeDimReduct)

        expectedPath: List['BaseNode'] = [self.nodegraph._inputNode,
                                          nodeSmoothDeriv,
                                          nodeDimReduct,
                                          self.nodegraph._nodeClf]

        nodePath: List['BaseNode'] = self.nodegraph._getClassificationPath()
        self.assertEqual(nodePath, expectedPath)

        self.nodegraph._deleteNode(nodeDimReduct)
        self.assertTrue(nodeSmoothDeriv.isConnectedToInput())
        self.assertFalse(self.nodegraph._nodeClf.isConnectedToInput())
        nodePath = self.nodegraph._getClassificationPath()
        self.assertEqual(nodePath, [])

    def test_getNodeDict(self):
        nodeDimReduct: BaseNode = self.nodegraph._nodes[1]
        nodeSmoothDeriv: BaseNode = self.nodegraph._nodes[0]
        self.assertTrue(type(nodeSmoothDeriv) == NodeSmoothDeriv)
        self.assertTrue(type(nodeDimReduct) == NodeDimReduct)

        nodeDict: dict = nodeDimReduct.toDict()
        self.assertEqual(nodeDict["label"], nodeDimReduct.label)
        self.assertEqual(nodeDict["id"], nodeDimReduct.id)

        inputs: dict = nodeDict["inputs"]
        self.assertEqual(list(inputs.keys()), [0])  # the first input has a connection
        self.assertEqual(inputs[0], [nodeSmoothDeriv.id, 0])  # it is connected to the first output of the gray node, specified by its id

        params: dict = nodeDict["params"]
        self.assertEqual(params["numComps"], nodeDimReduct._numcompSpin.value())
        self.assertEqual(params["pcaChecked"], nodeDimReduct._pcaBtn.isChecked())

    def test_deleteAllNodes(self):
        self.nodegraph._deleteAllNodesAndConnections()
        self.assertEqual(len(self.nodegraph._nodes), 0)
        self.assertEqual(len(self.nodegraph._connections), 0)

    def test_saveLoadNodeConfig(self) -> None:
        smoothNode: NodeSmoothDeriv = cast(NodeSmoothDeriv, self.nodegraph._nodes[0])
        self.assertTrue(type(smoothNode) == NodeSmoothDeriv)
        origWinSize = smoothNode._winSizeSpin.value()
        origNodePath: List['BaseNode'] = copy.copy(self.nodegraph._getClassificationPath())
        smoothNode._winSizeSpin.setValue(2*origWinSize)

        with tempfile.TemporaryDirectory() as tmpDir:
            savePath: str = os.path.join(tmpDir, 'nodeConfig.txt')
            self.nodegraph._saveConfig(savePath)
            self.assertTrue(os.path.exists(savePath))

            smoothNode._winSizeSpin.setValue(0)  # set to something else..
            self.nodegraph._loadConfig(savePath)

        loadedNodePath: List['BaseNode'] = self.nodegraph._getClassificationPath()
        self.assertEqual(len(origNodePath), len(loadedNodePath))

        for origNode, newNode in zip(origNodePath, loadedNodePath):
            self.assertEqual(type(origNode), type(newNode))
            self.assertEqual(origNode.id, newNode.id)
            if type(newNode) == NodeSmoothDeriv:
                self.assertEqual(newNode._winSizeSpin.value(), 2*origWinSize)