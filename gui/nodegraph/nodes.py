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

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeStart, self).__init__(nodeGraphParent, logger, pos)
        self.isStartNode = True
        self._outputs = [Output(self, 'Spectra', DataType.CONTINUOUS)]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str == '') -> float:
        return self._inputs[0].getValue()


class NodeScatterPlot(BaseNode):
    label = 'Scatter Plot'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeScatterPlot, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Scatter Plot', [DataType.CONTINUOUS, DataType.DISCRETE])]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> object:
        return None


class NodeSpecPlot(BaseNode):
    label = 'Spectra Plot'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeSpecPlot, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Spectra Plot', [DataType.CONTINUOUS])]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> object:
        return None


class NodeClassification(BaseNode):
    label = 'Classification'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeClassification, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Spectra Plot', [DataType.CONTINUOUS, DataType.DISCRETE])]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> object:
        return None


class NodeSNV(BaseNode):
    label = 'SNV'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeSNV, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Spectra Plot', [DataType.CONTINUOUS])]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> object:
        inputSpectra: np.ndarray = self._inputs[0].getValue()
        return inputSpectra


class NodeNormalize(BaseNode):
    label = 'Normalize'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeNormalize, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Spectra Plot', [DataType.CONTINUOUS])]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> object:
        inputSpectra: np.ndarray = self._inputs[0].getValue()
        return inputSpectra

# Here we define a dictionary of node classes that can be added to the nodegraph by the user.
# Some "unique" ones have to be expluded, as they are in the nodegraph by default and cannot be there more
# than one time.
addableNodeTypes: List[Type['BaseNode']] = [val for key, val in locals().items()
                                            if key.startswith('Node') and key not in ['NodeStart',
                                                                                      'NodeScatterPlot',
                                                                                      'NodeSpecPlot',
                                                                                      'NodeClassification']]

addableNodeTypes: Dict[str, Type['BaseNode']] = {node.label: node for node in addableNodeTypes}