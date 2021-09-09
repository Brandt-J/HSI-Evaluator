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

from gui.nodegraph.nodecore import *
if TYPE_CHECKING:
    from gui.nodegraph.nodegraph import NodeGraph
    from logging import Logger


class StartNode(BaseNode):
    label = 'Spectra'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(StartNode, self).__init__(nodeGraphParent, logger, pos)
        self.isStartNode = True
        self._outputs = [Output(self, '', DataType.CONTINUOUS)]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str == '') -> float:
        return self._inputs[0].getValue()


class NodeScatterPlot(BaseNode):
    label = 'PCA'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeScatterPlot, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input(self, 'Spectraaa', DataType.SPECTRA_GENERAL)]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> object:
        return None


class NodeSpecPlot(BaseNode):
    label = 'Spectra Plot'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeSpecPlot, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input(self, 'Spectraaa', VariableType.SPECTRA_CONT)]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> object:
        return None
