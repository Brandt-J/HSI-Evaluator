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
import random
import json
from PyQt5 import QtWidgets, QtGui, QtCore
import pickle
import os
from typing import List, TYPE_CHECKING, Dict, Set, Union, Optional, Tuple
import numpy as np

from gui.sampleview import SampleView
from logger import getLogger
from projectPaths import getAppFolder
from spectraObject import SpectraCollection, WavelengthsNotSetError
from dataObjects import Sample
from legacyConvert import assertUpToDateSample

if TYPE_CHECKING:
    from gui.HSIEvaluator import MainWindow
    from logging import Logger


class MultiSampleView(QtWidgets.QGraphicsView):
    """
    Container class for showing multiple sampleviews in a ScrollArea.
    """
    def __init__(self, mainWinParent: 'MainWindow'):
        super(MultiSampleView, self).__init__()
        self._setUpScene()
        self._mainWinParent: 'MainWindow' = mainWinParent
        self._sampleviews: List['SampleView'] = []
        self._logger: 'Logger' = getLogger('MultiSampleView')
        screenRes: QtCore.QSize = QtWidgets.QApplication.primaryScreen().size()
        multiViewWidth: int = int(round(screenRes.width() * 0.5))
        self.setMinimumWidth(multiViewWidth)

        self._startDrag: Optional[QtCore.QPoint] = None
        self._draggedView: Optional['SampleView'] = None

        self.setMouseTracking(True)

    def addSampleView(self) -> 'SampleView':
        """
        Adds a new sampleview and sets up the graphView properly
        :return: the new sampleview
        """
        newView: 'SampleView' = SampleView()
        newView.setMainWindowReferences(self._mainWinParent)
        newView.Activated.connect(self._viewActivated)
        newView.Closed.connect(self._closeSample)
        newView.WavelenghtsChanged.connect(self._assertIdenticalWavelengths)
        newView.activate()

        newView.getGraphOverlayObj().ParticlesChanged.connect(self._updateElementsFromSample)
        self._mainWinParent.setupConnections(newView)

        newPos: QtCore.QPointF = self._getNewViewPosition()
        self._sampleviews.append(newView)
        self._addElementsFromSample(newView)
        newView.setPos(newPos)
        newView.addToolsGroup()
        
        self.ensureVisible(newView.boundingRect())
        self._logger.debug("New Sampleview added")
        return newView

    def updateClassificationResults(self):
        for sampleView in self._sampleviews:
            sampleView.updateClassImageInGraphView()
            sampleView.updateParticlesInGraphUI()

    @QtCore.pyqtSlot()
    def toggleSampleToolbars(self) -> None:
        for sample in self._sampleviews:
            sample.toggleToolarVisibility()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MiddleButton:
            self._startDrag = event.pos()

        elif event.modifiers() == QtCore.Qt.ShiftModifier:
            self._draggedView = self._getViewAtPositon(event.pos())
            if self._draggedView is not None:
                self._draggedView.activate()
                self._startDrag = self.mapToScene(event.pos())
        else:
            for view in self._sampleviews:
                view.mousePressEvent(event)
            super(MultiSampleView, self).mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._startDrag is not None:
            p0 = event.pos()
            if self._draggedView is None:
                move = self._startDrag - p0
                self.horizontalScrollBar().setValue(move.x() + self.horizontalScrollBar().value())
                self.verticalScrollBar().setValue(move.y() + self.verticalScrollBar().value())
            else:
                p0 = self.mapToScene(p0)
                move = self._startDrag - p0
                self._draggedView.setX(self._draggedView.x() - move.x())
                self._draggedView.setY(self._draggedView.y() - move.y())
            self._startDrag = p0
        else:
            for view in self._sampleviews:
                view.mouseMoveEvent(event)
            super(MultiSampleView, self).mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._startDrag is not None:
            self._startDrag = None
            if self._draggedView is not None:
                self._draggedView = None
        else:
            super(MultiSampleView, self).mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        factor = 1.01**(event.angleDelta().y()/8)
        self.scale(factor, factor)
        event.accept()

    def _setUpScene(self) -> QtWidgets.QGraphicsScene:
        scene = QtWidgets.QGraphicsScene(self)
        scene.setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)
        scene.setBackgroundBrush(QtCore.Qt.darkGray)
        self.setScene(scene)
        self.setCacheMode(QtWidgets.QGraphicsView.CacheBackground)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.BoundingRectViewportUpdate)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        return scene

    def _getViewAtPositon(self, pos: QtCore.QPoint) -> Union[None, 'SampleView']:
        """
        Checks if a sampleview is present at the current position and returns it.
        Otherwise, returns None.
        """
        pos: QtCore.QPointF = self.mapToScene(pos)
        presentView: Union[None, 'SampleView'] = None
        for view in self._sampleviews:
            if view.x() <= pos.x() <= view.x() + view.boundingRect().width():
                if view.y() <= pos.y() <= view.y() + view.boundingRect().height():
                    presentView = view
                    break
        return presentView

    def _getNewViewPosition(self) -> QtCore.QPointF:
        spacing: int = 50
        viewEnds: List[float] = [view.y() + view.boundingRect().height() for view in self._sampleviews]
        if len(viewEnds) == 0:
            newY = 0
        else:
            newY = max(viewEnds) + spacing
        newPt: QtCore.QPointF = QtCore.QPointF(0, newY)
        return newPt

    def _assertIdenticalWavelengths(self) -> None:
        """
        Asserts that all samples have identical wavelenghts. If multiple wavelength axes are present, the shortest
        axis is used.
        """
        shortestWavelenghts, shortestWavelengthsLength = None, np.inf
        for sample in self._sampleviews:
            try:
                curWavelenghts: np.ndarray = sample.getWavelengths()
            except WavelengthsNotSetError:
                pass  # we just skip it here
            else:
                if len(curWavelenghts) < shortestWavelengthsLength:
                    shortestWavelengthsLength = len(curWavelenghts)
                    shortestWavelenghts = curWavelenghts

        if shortestWavelenghts is not None:
            for sample in self._sampleviews:
                try:
                    sample.getSpecObj().remapToWavelenghts(shortestWavelenghts)
                except WavelengthsNotSetError:
                    pass  # We can safely ignore that here.

    def loadSampleViewFromFile(self, fpath: str) -> None:
        """Loads the sample configuration from the file and creates a sampleview accordingly"""
        with open(fpath, "rb") as fp:
            loadedSampleData: 'Sample' = pickle.load(fp)
        self._createNewSampleFromSampleData(loadedSampleData)
        
    def createListOfSamples(self, sampleList: List['Sample']) -> None:
        """Creates a list of given samples and replaces the currently opened with that."""
        self._logger.info("Closing all samples, opening the following new ones..")
        for sample in self._sampleviews:
            self._removeElementsFromSample(sample)

        self._sampleviews = []
        for sample in sampleList:
            self._createNewSampleFromSampleData(sample)
            self._logger.info(f"Creating sample {sample.name}")

    def positonSampleViewsAsSaved(self) -> None:
        """
        Called when loading a view. The loaded samples are positioned in the viewport as they where when the view
        was saved.
        """
        for sample in self._sampleviews:
            pos: Union[None, Tuple[float, float]] = sample.getSampleData().viewCoordinates
            if pos is not None:
                sample.setPos(QtCore.QPointF(pos[0], pos[1]))

    def _createNewSampleFromSampleData(self, sampleData: Sample) -> None:
        """
        Creates a new sample and configures it according to the provided sample data object.
        Also handles legacy conversion.
        """
        newSampleData: 'Sample' = Sample()
        loadedSampleData = assertUpToDateSample(sampleData)
        newSampleData.__dict__.update(loadedSampleData.__dict__)

        newView: SampleView = self.addSampleView()
        newView.setSampleData(newSampleData)

        newView.setupFromSampleData()
        self._mainWinParent.updateClassCreatorClasses()

    def saveSamples(self) -> None:
        """
        Saves all the loaded samples individually.
        """
        for sample in self._sampleviews:
            self._saveSampleView(sample)

    def getSampleViews(self) -> List['SampleView']:
        """Returns a list of all samples"""
        return self._sampleviews.copy()

    def getActiveSample(self) -> 'SampleView':
        """
        Returns the currently active sample.
        """
        activeSample: Union[None, 'SampleView'] = None
        for sample in self._sampleviews:
            if sample.isActive():
                activeSample = sample
                break
        assert activeSample is not None
        return activeSample

    def getWavelengths(self) -> np.ndarray:
        """
        Returns the wavelength axis.
        """
        wavelenghts: Union[None, np.ndarray] = None
        for sample in self._sampleviews:
            try:
                sampleWavelengths: np.ndarray = sample.getWavelengths()
            except WavelengthsNotSetError:
                self._logger.warning(f"No wavelengths set in sample {sample.getName()}")
            else:
                if wavelenghts is None:
                    wavelenghts = sampleWavelengths
                else:
                    assert np.array_equal(wavelenghts, sampleWavelengths)

        assert wavelenghts is not None
        return wavelenghts

    def getClassNamesFromAllSamples(self) -> Set[str]:
        """
        Returns the class names that are used from all the samples that are currently loaded.
        """
        clsNames: List[str] = []
        for sample in self._sampleviews:
            clsNames += list(sample.getClassNames())
        return set(clsNames)

    def getLabelledSpectraFromActiveView(self) -> SpectraCollection:
        """
        Gets the labelled Spectra, in form of a dictionary, from the active sampleview
        :return: SpectraCollection with all the daata
        """
        specColl: SpectraCollection = SpectraCollection()
        for view in self._sampleviews:
            if view.isActive():
                spectra: Dict[str, np.ndarray] = view.getVisibleLabelledSpectra()
                specColl.addSpectraDict(spectra, view.getName())
                break
        return specColl

    def getLabelledSpectraFromAllViews(self) -> SpectraCollection:
        """
        Gets the labelled Spectra, in form of a dictionary, from the all sampleviews
        :return: SpectraCollectionObject
        """
        specColl: SpectraCollection = SpectraCollection()
        for view in self._sampleviews:
            specColl.addSpectraDict(view.getVisibleLabelledSpectra(), view.getName())
        return specColl

    def closeAllSamples(self) -> None:
        """
        Closes all opened sample views.
        """
        for sample in self._sampleviews:
            self._removeElementsFromSample(sample)

    @QtCore.pyqtSlot(str)
    def _updateElementsFromSample(self, samplename: str) -> None:
        for sample in self._sampleviews:
            if sample.getName() == samplename:
                self._removeElementsFromSample(sample)
                self._addElementsFromSample(sample)
                break

    def _addElementsFromSample(self, sample: 'SampleView') -> None:
        self.scene().addItem(sample)
        for item in sample.getGraphOverlayObj().getGraphicItems():
            self.scene().addItem(item)

    def _removeElementsFromSample(self, sample: 'SampleView') -> None:
        """
        Removes the graphics items from the given sample from the graphics scene.
        """
        self.scene().removeItem(sample)
        for item in sample.getGraphOverlayObj().getGraphicItems():
            self.scene().removeItem(item)

    @QtCore.pyqtSlot(str)
    def _closeSample(self, samplename: str) -> None:
        for sampleview in self._sampleviews:
            if sampleview.getName() == samplename:
                self._removeElementsFromSample(sampleview)
                self._sampleviews.remove(sampleview)
                self._logger.info(f"Closed Sample {samplename}")
                break

    def _saveSampleView(self, sampleview: 'SampleView') -> None:
        """
        Saves the given sampleview.
        """
        directory: str = self.getSampleSaveDirectory()
        savePath: str = os.path.join(directory, sampleview.getSaveFileName())
        sampleview.saveCoordinatesToSampleData()
        sampleData: 'Sample' = sampleview.getSampleDataToSave()
        with open(savePath, "wb") as fp:
            pickle.dump(sampleData, fp)

        self._logger.info(f"Saved sampleview {sampleview.getName()} at {savePath}")

    @QtCore.pyqtSlot(str)
    def _viewActivated(self, samplename: str) -> None:
        """
        Handles activation of a new sampleview, i.e., deactivates the previously active one.
        :param samplename: The name of the sample
        :return:
        """
        for view in self._sampleviews:
            if view.getName() != samplename and view.isActive():
                view.deactivate()
        self.scene().update()

    @staticmethod
    def getSampleSaveDirectory() -> str:
        """
        Returns the path of a directory used for storing individual sample views.
        """
        path: str = os.path.join(getAppFolder(), "Samples")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def getViewSaveDirectory() -> str:
        """
        Returns the path of a directoy used for storing the entirety of the current selection.
        """
        path: str = os.path.join(getAppFolder(), "Views")
        os.makedirs(path, exist_ok=True)
        return path

    def exportSpectra(self) -> None:
        """
        Exports the labelled spectra for use in other software.
        """
        specColl: SpectraCollection = self.getLabelledSpectraFromAllViews()
        specArr, assignments = specColl.getXY()

        folder: str = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory where to save to.")
        numSpecs, ok = QtWidgets.QInputDialog.getInt(self, "Max. Number of Spectra to Export?",
                                                     "Enter the max. number of spectra to export.",
                                                          2000, 100, 10000, 500)
        if folder and ok:
            if len(assignments) > numSpecs:
                randInd: np.ndarray = np.array(random.sample(list(np.arange(len(assignments))), numSpecs))
                specArr = specArr[randInd, :]
                assignments = assignments[randInd]

            uniqueAssignments: List[str] = list(np.unique(assignments))
            assignmentDict: Dict[str, int] = {cls: uniqueAssignments.index(cls)+1 for cls in assignments}  # class 0 get's ignored by PLS Toolbox, hence we have the +1 here.
            numberAssignments: np.ndarray = np.array([assignmentDict[cls] for cls in assignments])

            specPath: str = os.path.join(folder, f"Exported Spectra from {len(self._sampleviews)} samples.txt")
            np.savetxt(specPath, specArr)
            assignPath: str = os.path.join(folder, f"Exported Assignments from {len(self._sampleviews)} samples.txt")
            np.savetxt(assignPath, numberAssignments)

            codePath: str = os.path.join(folder, f"Exported Spectra Encoding from {len(self._sampleviews)} samples.txt")
            with open(codePath, "w") as fp:
                json.dump(assignmentDict, fp)

            QtWidgets.QMessageBox.about(self, "Info", f"Spectra and Assignments saved to\n{folder}\n\n"
                                                      f"Class encoding:\n"
                                                      f"{assignmentDict}")


