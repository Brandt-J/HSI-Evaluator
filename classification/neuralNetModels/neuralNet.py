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
from typing import List

from tensorflow.keras.layers import Dense, Dropout, InputLayer, Conv1D, MaxPool1D, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential, load_model

from classification.neuralNetModels.globalMetrics import GlobalRecall, GlobalPrecision


def loadModelFromFile(fname: str) -> 'NeuralNetClf':
    """
    Loads a saved neural net model from file.
    """
    return load_model(fname)


class NeuralNetClf(Sequential):
    def __init__(self, numFeatures: int, numClasses: int, numNeuronsPerHiddenLayer: List[int] = [200, 100, 50],
                 dropout: float = 0.1):
        super(NeuralNetClf, self).__init__()
        self.add(InputLayer(input_shape=(numFeatures)))
        for numNeurons in numNeuronsPerHiddenLayer:
            self.add(Dense(numNeurons, activation="relu"))
            self.add(Dropout(dropout))
        self.add(Dense(numClasses, activation="softmax"))
        self.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[GlobalPrecision, GlobalRecall])
        # self.summary()


class ConvNeuralNetCLF(Sequential):
    def __init__(self, numFeatures: int, numClasses: int):
        super(ConvNeuralNetCLF, self).__init__()
        self.add(InputLayer(input_shape=(numFeatures, 1)))
        self.add(Conv1D(16, 3, padding="same", activation="relu"))
        # self.add(Dropout(0.1))
        self.add(MaxPool1D(2, padding="same"))
        self.add(BatchNormalization())
        self.add(Conv1D(32, 3, padding="same", activation="relu"))
        # self.add(Dropout(0.1))
        self.add(MaxPool1D(2, padding="same"))
        self.add(BatchNormalization())
        self.add(Conv1D(64, 3, padding="same", activation="relu"))
        self.add(Dropout(0.1))
        self.add(MaxPool1D(2, padding="same"))
        self.add(Conv1D(128, 3, padding="same", activation="relu"))
        self.add(Dropout(0.1))
        self.add(Flatten())
        self.add(Dense(100, activation="relu"))
        self.add(Dropout(0.3))
        self.add(Dense(50, activation="relu"))
        self.add(Dense(numClasses, activation="softmax"))
        self.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[Precision, Recall])
        self.summary()
