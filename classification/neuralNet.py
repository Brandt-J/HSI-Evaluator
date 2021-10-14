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
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.metrics import Precision, Recall


def loadModelFromFile(fname: str) -> 'NeuralNetClf':
    """
    Loads a saved neural net model from file.
    """
    return load_model(fname)


class NeuralNetClf(Sequential):
    def __init__(self, numFeatures: int, numClasses: int):
        super(NeuralNetClf, self).__init__()
        self.add(Dense(100, input_dim=numFeatures, activation="relu"))
        self.add(BatchNormalization())
        self.add(Dense(50, activation="relu"))
        self.add(BatchNormalization())
        self.add(Dense(50, activation="relu"))
        self.add(Dense(20, activation="relu"))
        self.add(Dense(numClasses, activation="softmax"))
        self.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[Precision(), Recall()])
        # self.summary()


