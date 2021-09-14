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
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


from gui.nodegraph.nodecore import *
from preprocessing.processing import *

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
        self._spectra: Union[None, np.ndarray] = None

    def setSpectra(self, spectra: np.ndarray) -> None:
        self._spectra = spectra

    def getOutput(self, outputName: str == '') -> np.ndarray:
        assert self._spectra is not None, "Spectra were not yet set on Input Node!"
        return self._spectra


class NodeScatterPlot(BaseNode):
    label = 'Scatter Plot'
    isRequiredAndUnique = True

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeScatterPlot, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Scatter Plot', [DataType.DISCRETE])]
        self._cachedSpectra: Union[None, np.ndarray] = None
        self._populateLayoutAndCreateIO()

    def isConnected(self) -> bool:
        return self._inputs[0].isConnected()

    def getOutput(self, outputName: str = '') -> np.ndarray:
        if self._cachedSpectra is not None:
            specs: np.ndarray = self._cachedSpectra
        else:
            specs: np.ndarray = self._inputs[0].getValue()
        return specs


class NodeSpecPlot(BaseNode):
    label = 'Spectra Plot'
    isRequiredAndUnique = True

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeSpecPlot, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Spectra Plot', [DataType.CONTINUOUS])]
        self._cachedSpectra: Union[None, np.ndarray] = None
        self._populateLayoutAndCreateIO()

    def isConnected(self) -> bool:
        return self._inputs[0].isConnected()

    def getOutput(self, outputName: str = '') -> np.ndarray:
        if self._cachedSpectra is not None:
            specs: np.ndarray = self._cachedSpectra
        else:
            specs: np.ndarray = self._inputs[0].getValue()
        return specs


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

        self._pcaBtn: QtWidgets.QRadioButton = QtWidgets.QRadioButton("PCA")
        self._pcaBtn.setChecked(True)
        self._tsneBtn: QtWidgets.QRadioButton = QtWidgets.QRadioButton("t-SNE")
        self._numcompSpin: QtWidgets.QSpinBox = QtWidgets.QSpinBox()

        self._pcaBtn.toggled.connect(lambda: self.ParamsChanged.emit())
        self._tsneBtn.toggled.connect(lambda: self.ParamsChanged.emit())
        self._numcompSpin.valueChanged.connect(lambda: self.ParamsChanged.emit())
        self._numcompSpin.setMinimum(2)
        self._numcompSpin.setMaximum(100)
        self._numcompSpin.setValue(3)

        btnLayout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        btnLayout.addWidget(self._pcaBtn)
        btnLayout.addWidget(self._tsneBtn)

        self._bodywidget: QtWidgets.QGroupBox = QtWidgets.QGroupBox()
        layout: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        self._bodywidget.setLayout(layout)
        layout.addLayout(btnLayout, 0, 0, 1, 2)
        layout.addWidget(QtWidgets.QLabel("Num Components:"), 1, 0)
        layout.addWidget(self._numcompSpin)

        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> object:
        inputSpectra: np.ndarray = self._inputs[0].getValue()
        numComps: int = self._numcompSpin.value()
        if self._pcaBtn.isChecked():
            pca: PCA = PCA(n_components=numComps)
            outSpecs: np.ndarray = pca.fit_transform(inputSpectra)
        else:
            if numComps > 3:
                QtWidgets.QMessageBox.about(self._parentGraph, "Info", "Num Components cannot be greater than 3 with t-SNE.\n"
                                                                       "Calculating only three components.")
                numComps = 3
            tsne: TSNE = TSNE(n_components=numComps)
            outSpecs: np.ndarray = tsne.fit_transform(inputSpectra)
        return outSpecs


class NodeSNV(BaseNode):
    label = 'Standard Normal Variate'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeSNV, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Spectra', [DataType.CONTINUOUS])]
        self._outputs = [Output(self, 'Spectra', DataType.CONTINUOUS)]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> np.ndarray:
        inputSpectra: np.ndarray = self._inputs[0].getValue()
        return snv(inputSpectra)


class NodeNormalize(BaseNode):
    label = 'Normalize'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeNormalize, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Spectra', [DataType.CONTINUOUS])]
        self._outputs = [Output(self, 'Spectra', DataType.CONTINUOUS)]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> np.ndarray:
        inputSpectra: np.ndarray = self._inputs[0].getValue()
        return normalizeIntensities(inputSpectra)


class NodeDetrend(BaseNode):
    label = 'Detrend'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeDetrend, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Spectra', [DataType.CONTINUOUS])]
        self._outputs = [Output(self, 'Spectra', DataType.CONTINUOUS)]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> np.ndarray:
        inputSpectra: np.ndarray = self._inputs[0].getValue()
        return detrend(inputSpectra)


class NodeMSC(BaseNode):
    label = 'Mult. Scatt. Corr.'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeMSC, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Spectra', [DataType.CONTINUOUS])]
        self._outputs = [Output(self, 'Spectra', DataType.CONTINUOUS)]
        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> np.ndarray:
        inputSpectra: np.ndarray = self._inputs[0].getValue()
        return msc(inputSpectra)


class NodeSmoothDeriv(BaseNode):
    label = 'Derivative/Smooth'

    def __init__(self, nodeGraphParent: 'NodeGraph', logger: 'Logger', pos: QtCore.QPointF = QtCore.QPointF()):
        super(NodeSmoothDeriv, self).__init__(nodeGraphParent, logger, pos)
        self._inputs = [Input('Spectra', [DataType.CONTINUOUS])]
        self._outputs = [Output(self, 'Spectra', DataType.CONTINUOUS)]

        self._derivSpin: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
        self._derivSpin.setMinimum(0)
        self._derivSpin.setMaximum(5)
        self._derivSpin.setValue(1)

        self._winSizeSpin: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
        self._winSizeSpin.setMinimum(1)
        self._winSizeSpin.setMaximum(21)
        self._winSizeSpin.setValue(5)

        self._bodywidget: QtWidgets.QGroupBox = QtWidgets.QGroupBox()
        layout: QtWidgets.QFormLayout = QtWidgets.QFormLayout()
        layout.addRow("Derivative Order:", self._derivSpin)
        layout.addRow("Window Size:", self._winSizeSpin)
        self._bodywidget.setLayout(layout)

        self._populateLayoutAndCreateIO()

    def getOutput(self, outputName: str = '') -> np.ndarray:
        inputSpectra: np.ndarray = self._inputs[0].getValue()
        return deriv_smooth(inputSpectra, self._derivSpin.value(), self._winSizeSpin.value())



nodeTypes: List[Type['BaseNode']] = [val for key, val in locals().items() if key.startswith('Node')]
nodeTypes: Dict[str, Type['BaseNode']] = {node.label: node for node in nodeTypes}
