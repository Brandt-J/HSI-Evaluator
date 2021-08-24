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
from PyQt5 import QtWidgets
from typing import *
from sklearn import svm


def getClassifiers() -> List['BaseClassifier']:
    # return [NeuralNet(), SVM(), RDF()]
    return [SVM()]


class ClassificationError(Exception):
    """
    Custom Error class for errors occuring during classification.
    """
    def __init__(self, errorText):
        self.errorText = errorText

    def __str__(self):
        return repr(self.errorText)


class BaseClassifier:
    """
    Base class for a classifier
    """
    title: str = ''

    def __init__(self):
        self._uniqueLabels: Dict[str, int] = {}  # Dictionary mapping class names to their unique indices

    def getControls(self) -> QtWidgets.QGroupBox:
        """
        Return the groupbox for classifier controls in the ui.
        """
        return QtWidgets.QGroupBox(self.title)

    def makePickleable(self) -> None:
        """
        Can be overloaded if the classifier cannot be pickled, because it stores QWidgets, for instance.
        We are using processes, so the classifier needs to be pickleable.
        """
        pass

    def restoreNotPickleable(self) -> None:
        """Restores the originale version, see comment to makePickleable method"""

    def setWavenumbers(self, wavenums: np.ndarray) -> None:
        """
        Overload, if the classifier needs to be configured to the wavenumbers.
        :param wavenums: 1d array of wavenumbers
        """
        pass

    def _setUniqueLabels(self, ytest: np.ndarray, ytrain: np.ndarray) -> None:
        """
        Sets the unique labels for the current test/train set.
        """
        self._uniqueLabels = {}
        allLabels: np.ndarray = np.hstack((ytest, ytrain))
        for i, label in enumerate(np.unique(allLabels)):
            self._uniqueLabels[label] = i

    def _convertLabelsToNumbers(self, textlabels: np.ndarray) -> np.ndarray:
        """
        Takes an array of text labels and returns the corresponding array of indices, according to the unique labels.
        """
        return np.array([self._uniqueLabels[lbl] for lbl in textlabels])

    def _convertNumbersToLabels(self, numberLabels: np.ndarray) -> np.ndarray:
        """
        Takes an array of text number and returns the corresponding array of text labels, according to the unique labels.
        """
        key_list = list(self._uniqueLabels.keys())
        val_list = list(self._uniqueLabels.values())

        textLabels: List[str] = []
        for num in numberLabels:
            position: int = val_list.index(num)
            textLabels.append(key_list[position])

        return np.array(textLabels)

    def train(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
        """
        Train the classifier with the given data.
        """
        raise NotImplementedError

    def predict(self, spectra: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given spectra
        :param spectra: (NxM) array of N spectra with M wavenumbers.
        :return
        """
        raise NotImplementedError


# class NeuralNet(BaseClassifier):
#     title = "Neural Net"
#
#     def train(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
#         pass
#
#     def predict(self, spectra: np.ndarray) -> np.ndarray:
#         return np.zeros(spectra.shape[0])
#
#
# class RDF(BaseClassifier):
#     title = "Random Decision Forest"
#
#     def train(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
#         pass
#
#     def predict(self, spectra: np.ndarray) -> np.ndarray:
#         return np.zeros(spectra.shape[0])


class SVM(BaseClassifier):
    title = "Support Vector Machine"

    def __init__(self):
        super(SVM, self).__init__()
        self._clf: Union[None, svm.SVC] = None
        self._kernel: str = "linear"
        self._kernelSelector: Union[None, QtWidgets.QComboBox] = None
        self._recreateComboBox()

    def _recreateComboBox(self) -> None:
        self._kernelSelector = QtWidgets.QComboBox()
        self._kernelSelector.addItems(["linear", "poly", "rbf", "sigmoid", "precomputed"])
        self._kernelSelector.setCurrentText(self._kernel)
        self._kernelSelector.currentTextChanged.connect(self._updateClassifier)

    def _updateClassifier(self) -> None:
        self._kernel = self._kernelSelector.currentText()

    def getControls(self) -> QtWidgets.QGroupBox:
        group: QtWidgets.QGroupBox = QtWidgets.QGroupBox("SVM Options")
        layout: QtWidgets.QFormLayout = QtWidgets.QFormLayout()
        layout.addRow("Kernel", self._kernelSelector)
        group.setLayout(layout)
        return group

    def makePickleable(self) -> None:
        self._kernelSelector = None

    def restoreNotPickleable(self) -> None:
        self._recreateComboBox()

    def train(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
        self._clf = svm.SVC(kernel=self._kernel)
        self._setUniqueLabels(y_test, y_train)
        self._clf.fit(x_train, self._convertLabelsToNumbers(y_train))

    def predict(self, spectra: np.ndarray) -> np.ndarray:
        assert self._clf is not None, "Classifier was not yet created!!"
        labels: np.ndarray = self._clf.predict(spectra)
        return self._convertNumbersToLabels(labels)
