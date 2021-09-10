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

from gui.nodegraph.nodecore import *
if TYPE_CHECKING:
    from gui.nodegraph.nodegraph import NodeGraph
    from logging import Logger


class NodeStart(BaseNode):
    label = 'Spectra'
    isRequiredAndUnique = True

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeStart, self).__init__(nodeGraphParent, logger, pos)
        self.isStartNode = True
        self._outputs = [Output(self, 'Spectra', DataType.CONTINUOUS)]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str == '') -> float:
        return self._inputs[0].getValue()


class NodeScatterPlot(BaseNode):
    label = 'Scatter Plot'
    isRequiredAndUnique = True

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeScatterPlot, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Scatter Plot', [DataType.DISCRETE])]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> object:
        return None


class NodeSpecPlot(BaseNode):
    label = 'Spectra Plot'
    isRequiredAndUnique = True

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeSpecPlot, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Spectra Plot', [DataType.CONTINUOUS])]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> object:
        return None


class NodeClassification(BaseNode):
    label = 'Classification'
    isRequiredAndUnique = True

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeClassification, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Spectra Plot', [DataType.CONTINUOUS, DataType.DISCRETE])]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> object:
        return None


class NodeDimReduct(BaseNode):
    label = 'Dimensionality Reduction'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeDimReduct, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Spectra', [DataType.CONTINUOUS, DataType.DISCRETE])]
        self._outputs = [Output(self, 'Spectra', DataType.DISCRETE)]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> object:
        inputSpectra: np.ndarray = self._inputs[0].getValue()
        return inputSpectra


class NodeSNV(BaseNode):
    label = 'Standard Normal Variate'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeSNV, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Spectra', [DataType.CONTINUOUS])]
        self._outputs = [Output(self, 'Spectra', DataType.CONTINUOUS)]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> object:
        inputSpectra: np.ndarray = self._inputs[0].getValue()
        return inputSpectra


class NodeNormalize(BaseNode):
    label = 'Normalize'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeNormalize, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Spectra', [DataType.CONTINUOUS])]
        self._outputs = [Output(self, 'Spectra', DataType.CONTINUOUS)]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> object:
        inputSpectra: np.ndarray = self._inputs[0].getValue()
        return inputSpectra


nodeTypes: List[Type['BaseNode']] = [val for key, val in locals().items() if key.startswith('Node')]
nodeTypes: Dict[str, Type['BaseNode']] = {node.label: node for node in nodeTypes}
