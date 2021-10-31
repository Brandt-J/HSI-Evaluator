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
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Union, TYPE_CHECKING
import numpy as np
from PyQt5 import QtWidgets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from logger import getLogger
from projectPaths import getAppFolder
from classification.neuralNet import NeuralNetClf, loadModelFromFile

if TYPE_CHECKING:
    from logging import Logger


@dataclass
class SavedClassifier:
    clf: 'BaseClassifier'
    validReport: dict
    preproMethod: Union[None, str] = None


class BatchClassificationResult:
    """
    Container for storing results from a classification of a batch of spectra.
    """
    def __init__(self, probabilityMatrix: np.ndarray, labelEncoder: 'LabelEncoder') -> None:
        """
        :param probabilityMatrix: (NxM) Matrix giving the probabilities for N spectra to belong to M classes
        :param labelEncoder: Fitted label encoder for converting indices to string class names
        """
        self._probabilityMatrix: np.ndarray = probabilityMatrix
        self._labelEncoder: 'LabelEncoder' = labelEncoder

    def getResults(self, cutoff: float = 0.0) -> np.ndarray:
        """
        Returns the classification results as an array. Results with a max. probability of less than the cutoff are
        set to "unknown".
        """
        maxIndices: np.ndarray = np.argmax(self._probabilityMatrix, axis=1)
        results: np.ndarray = self._labelEncoder.inverse_transform(maxIndices).astype('U128')  # max length of 128 chars per class name
        maxProbs: np.ndarray = np.max(self._probabilityMatrix, axis=1)
        results[maxProbs < cutoff] = "unknown"
        return results


class BaseClassifier:
    """
    Base class for a classifier
    """
    title: str = ''

    def __init__(self):
        self._logger: Logger = getLogger(f"classifier {self.title}")
        self._labelEncoder: Union[None, LabelEncoder] = None

    def getControls(self) -> QtWidgets.QGroupBox:
        """
        Return the groupbox for classifier controls in the ui.
        """
        return QtWidgets.QGroupBox(self.title)

    def _fitLabelEncoder(self, ytest: np.ndarray, ytrain: np.ndarray) -> None:
        """
        Sets the unique labels for the current test/train set.
        """
        allLabels: np.ndarray = np.hstack((ytest, ytrain))
        self._labelEncoder = LabelEncoder().fit(allLabels)

    def train(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
        """
        Train the classifier with the given data.
        """
        raise NotImplementedError

    def predict(self, spectra: np.ndarray) -> BatchClassificationResult:
        """
        Predict labels for the given spectra
        :param spectra: (NxM) array of N spectra with M wavelengths.
        :return Batch Classification Result allowing to determine final class according to a confidence threshold-
        """
        raise NotImplementedError


class KNN(BaseClassifier):
    title = "k-Nearest Neighbors"

    def __init__(self):
        super(KNN, self).__init__()
        self._clf: Union[None, KNeighborsClassifier] = None
        self._k: int = 5
        self._kSpinBox: Union[None, QtWidgets.QSpinBox] = None
        self._recreateSpinBox()

    def getControls(self) -> QtWidgets.QGroupBox:
        self._recreateSpinBox()
        optnGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox("KNN Options")
        optnGroup.setLayout(QtWidgets.QFormLayout())
        optnGroup.layout().addRow("Num. Neighbors:", self._kSpinBox)
        return optnGroup

    def _recreateSpinBox(self) -> None:
        self._kSpinBox = QtWidgets.QSpinBox()
        self._kSpinBox.setMinimum(2)
        self._kSpinBox.setMaximum(20)
        self._kSpinBox.setValue(self._k)
        self._kSpinBox.valueChanged.connect(self._update_k)

    def train(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
        self._clf = KNeighborsClassifier(n_neighbors=self._k)
        self._fitLabelEncoder(y_test, y_train)
        self._clf.fit(x_train, self._labelEncoder.transform(y_train))

    def predict(self, spectra: np.ndarray) -> BatchClassificationResult:
        assert self._clf is not None, "Classifier was not yet created!!"
        probMat: np.ndarray = self._clf.predict_proba(spectra)
        return BatchClassificationResult(probMat, self._labelEncoder)

    def _update_k(self) -> None:
        self._k = self._kSpinBox.value()


class SVM(BaseClassifier):
    title = "Support Vector Machine"

    def __init__(self):
        super(SVM, self).__init__()
        self._clf: Union[None, svm.SVC] = None
        self._kernel: str = "rbf"
        self._kernelSelector: Union[None, QtWidgets.QComboBox] = None
        self._recreateComboBox()

    def getControls(self) -> QtWidgets.QGroupBox:
        self._recreateComboBox()
        group: QtWidgets.QGroupBox = QtWidgets.QGroupBox("SVM Options")
        layout: QtWidgets.QFormLayout = QtWidgets.QFormLayout()
        layout.addRow("Kernel", self._kernelSelector)
        group.setLayout(layout)
        return group

    def train(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
        self._clf = svm.SVC(kernel=self._kernel, probability=True)
        self._fitLabelEncoder(y_test, y_train)
        self._clf.fit(x_train, self._labelEncoder.transform(y_train))

    def predict(self, spectra: np.ndarray) -> BatchClassificationResult:
        assert self._clf is not None, "Classifier was not yet created!!"
        probMat: np.ndarray = self._clf.predict_proba(spectra)
        return BatchClassificationResult(probMat, self._labelEncoder)

    def _recreateComboBox(self) -> None:
        self._kernelSelector = QtWidgets.QComboBox()
        self._kernelSelector.addItems(["linear", "poly", "rbf", "sigmoid", "precomputed"])
        self._kernelSelector.setCurrentText(self._kernel)
        self._kernelSelector.currentTextChanged.connect(self._updateClassifier)

    def _updateClassifier(self) -> None:
        self._kernel = self._kernelSelector.currentText()


class NeuralNet(BaseClassifier):
    title = "Neural Net"

    def __init__(self):
        super(NeuralNet, self).__init__()
        self._clf: Union[None, NeuralNetClf] = None
        self._spinEpochs: Union[None, QtWidgets.QSpinBox] = QtWidgets.QSpinBox()
        self._numEpochs: int = 20
        self._modelSavePath: Union[None, str] = None
        self._recreateEpochsSpinbox()

    def getControls(self) -> QtWidgets.QGroupBox:
        self._recreateEpochsSpinbox()
        optnGroup: QtWidgets.QGroupBox = QtWidgets.QGroupBox("Neural Net")
        optnGroup.setLayout(QtWidgets.QFormLayout())
        optnGroup.layout().addRow("Num. Epochs Training:", self._spinEpochs)
        return optnGroup

    def train(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
        self._fitLabelEncoder(y_test, y_train)
        self._clf = NeuralNetClf(x_train.shape[1], len(self._labelEncoder.classes_))
        y_train = to_categorical(self._labelEncoder.transform(y_train))
        y_test = to_categorical(self._labelEncoder.transform(y_test))
        self._clf.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=self._numEpochs)

    def predict(self, spectra: np.ndarray) -> BatchClassificationResult:
        if self._clf is None:
            if self._modelSavePath is not None:
                self._clf = loadModelFromFile(self._modelSavePath)
            else:
                raise ClassificationError("Neural Net Classifier does not exist and is not saved")

        probMat: np.ndarray = self._clf.predict(spectra)
        return BatchClassificationResult(probMat, self._labelEncoder)

    def _recreateEpochsSpinbox(self) -> None:
        self._spinEpochs = QtWidgets.QSpinBox()
        self._spinEpochs.setMinimum(1)
        self._spinEpochs.setMaximum(1000)
        self._spinEpochs.setValue(self._numEpochs)
        self._spinEpochs.valueChanged.connect(self._numEpochsChanged)

    def _numEpochsChanged(self) -> None:
        """
        Updates the numepochs value according to the spinbox value. Triggered by using the spinbox.
        """
        self._numEpochs = self._spinEpochs.value()


class ClassificationError(Exception):
    """
    Custom Error class for errors occuring during classification.
    """
    def __init__(self, errorText):
        self.errorText = errorText

    def __str__(self):
        return repr(self.errorText)
