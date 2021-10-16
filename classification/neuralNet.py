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

from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import TensorBoard


def loadModelFromFile(fname: str) -> 'NeuralNetClf':
    """
    Loads a saved neural net model from file.
    """
    return load_model(fname)


class NeuralNetClf(Sequential):
    def __init__(self, numFeatures: int, numClasses: int, numNeuronsPerHiddenLayer: List[int] = [100, 50, 50], dropout: float = 0.1):
        super(NeuralNetClf, self).__init__()
        self.add(InputLayer(input_shape=(numFeatures)))
        for numNeurons in numNeuronsPerHiddenLayer:
            self.add(Dense(numNeurons, input_dim=numFeatures, activation="relu"))
            self.add(Dropout(dropout))
        self.add(Dense(numClasses, activation="softmax"))
        self.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[Precision(), Recall()])
        self.summary()


if __name__ == '__main__':
    import numpy as np
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from collections import Counter
    import time
    import matplotlib.pyplot as plt
    import sys
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication(sys.argv)

    from preprocessing.routines import deriv_smooth, normalizeIntensities, NormMode

    specs: np.ndarray = np.load("spectra.npy")
    specs = deriv_smooth(specs, windowSize=9, polydegree=2, derivative=1)
    specs = normalizeIntensities(specs, NormMode.Length)
    # specs = (specs - specs.min()) / (specs.max() - specs.min())

    assignments: np.ndarray = np.load("labels.npy")
    assignments[assignments == "sediment A"] = "Sediment"
    assignments[assignments == "sediment B"] = "Sediment"
    assignments[assignments == "sediment C"] = "Sediment"
    assignments[assignments == "Sediment R"] = "Sediment"
    assignments[assignments == "eppi blue"] = "Eppi"
    assignments[assignments == "eppi green"] = "Eppi"
    print(Counter(assignments))

    encoder: LabelEncoder = LabelEncoder().fit(assignments)
    y: np.ndarray = encoder.transform(assignments)

    X_train, X_test, y_train, y_test = train_test_split(specs, y, test_size=0.2)
    numNeuronsLists: List[List[int]] = [[100, 50, 50],
                                        [20, 20, 10],
                                        [200, 100, 50],
                                        [100, 100, 50, 20, 10]
                                        ]
    dropout = 0.1
    numNeurons = [200, 100, 50]
    # for numNeurons in numNeuronsLists:
    modelName: str = f"Dense {'_'.join([str(i) for i in numNeurons])}_dropout_{dropout}_500epochs"
    print(modelName)
    nn: NeuralNetClf = NeuralNetClf(specs.shape[1], len(np.unique(assignments)),
                                    numNeuronsPerHiddenLayer=numNeurons, dropout=dropout)
    tensboard: TensorBoard = TensorBoard(log_dir=f"logs/{modelName}")
    t0 = time.time()
    history = nn.fit(X_train, to_categorical(y_train),
                     validation_data=(X_test, to_categorical(y_test)),
                     batch_size=64,
                     epochs=500, verbose=1,
                     callbacks=[tensboard])
    print(f"training NN took {time.time()-t0} seconds")
    t0 = time.time()
    pred: np.ndarray = nn.predict(X_test)
    pred = np.array([pred[i, :].argmax() for i in range(pred.shape[0])])
    print(f"Pred NN: {time.time()-t0} seconds\n", classification_report(encoder.inverse_transform(y_test),
                                                                        encoder.inverse_transform(pred),
                                                                        zero_division=0))
