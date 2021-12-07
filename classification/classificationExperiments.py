import random
from typing import Union

import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import imblearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from collections import Counter
import time
import matplotlib.pyplot as plt
import sys
from PyQt5 import QtWidgets
app = QtWidgets.QApplication(sys.argv)

from preprocessing.routines import deriv_smooth, normalizeIntensities
from preprocessing.preprocessors import NormMode
from classification.neuralNetModels.neuralNet import NeuralNetClf
from classification.neuralNetModels.resNet import ResNet1D

specs: np.ndarray = np.load("spectra.npy")
assignments: np.ndarray = np.load("labels.npy")
assert len(assignments) == specs.shape[0]
assert specs.shape[1] == 102

maxNumSpecs = 1e6
if len(assignments) > maxNumSpecs:
    ind: np.ndarray = np.array(random.sample(range(len(assignments)), int(maxNumSpecs)))
    specs = specs[ind, :]
    assignments = assignments[ind]


for i in range(len(assignments)):
    assignments[i] = assignments[i].replace("weathered", "")
    assignments[i] = assignments[i].replace("pristine", "")
    assignments[i] = assignments[i].replace("blue", "")
    assignments[i] = assignments[i].replace("white", "")
    assignments[i] = assignments[i].replace("_", "")

assignments[assignments == "sediment A"] = "Sediment"
assignments[assignments == "sediment B"] = "Sediment"
assignments[assignments == "sediment C"] = "Sediment"
# assignments[assignments != "Sediment"] = "Polymer"
indNoAmber: np.ndarray = np.where(assignments != "Amber")[0]
specs = specs[indNoAmber, :]
assignments = assignments[indNoAmber]
print("raw", Counter(assignments))

# specs = deriv_smooth(specs, windowSize=13, polydegree=3, derivative=1)
specs = normalizeIntensities(specs, NormMode.Length)

encoder: LabelEncoder = LabelEncoder().fit(assignments)
y: np.ndarray = encoder.transform(assignments)

X_train, X_test, y_train, y_test = train_test_split(specs, y, test_size=0.1)
X_train, y_train = imblearn.under_sampling.RandomUnderSampler().fit_resample(X_train, y_train)
X_test, y_test = imblearn.under_sampling.RandomUnderSampler().fit_resample(X_test, y_test)

print("final train:", Counter(y_train))
print("final test:", Counter(y_test))


def to_categorical_uniqueLabels(y_arr: np.ndarray, uniqueLabels: list) -> np.ndarray:
    y_cat: np.ndarray = np.zeros((len(y_arr), len(uniqueLabels)))
    for i, cur_y in enumerate(y_arr):
        index = uniqueLabels.index(cur_y)
        y_cat[i, index] = 1
    return y_cat


def evaluateTraining(nnModel: Union[NeuralNetClf, ResNet1D], startTime: float, modelName: str, histDict: dict) -> None:
    print(f"training {modelName} took {time.time() - startTime} seconds")
    t0 = time.time()
    pred: np.ndarray = nnModel.predict(X_test)
    pred = np.array([pred[i, :].argmax() for i in range(pred.shape[0])])
    print(f"Pred NN: {time.time() - t0} seconds\n", classification_report(encoder.inverse_transform(y_test),
                                                                          encoder.inverse_transform(pred),
                                                                          zero_division=0))
    createHistPlot(histDict, modelName)
    confMatFig: plt.Figure = ConfusionMatrixDisplay.from_predictions(encoder.inverse_transform(y_test),
                                                                     encoder.inverse_transform(pred)).figure_
    confMatFig.suptitle(modelName)


def createHistPlot(histDict: dict, title: str) -> None:
    histplot: plt.Figure = plt.figure()
    histax: plt.Axes = histplot.add_subplot()
    histax.plot(histDict["precision"], label="Precision")
    histax.plot(histDict["val_precision"], label="Val_Precision")
    histax.plot(histDict["recall"], label="Recall")
    histax.plot(histDict["val_recall"], label="Val_Recall")
    histax.legend()
    histax.set_title(title)


numFeatures, numClasses = specs.shape[1], len(np.unique(assignments))

# modelName: str = "DenseNet_200_100_50_OnlyNorm_Batch32"
# print(modelName)
# nn: NeuralNetClf = NeuralNetClf(numFeatures, numClasses, [200, 100, 50])
# tensboard: TensorBoard = TensorBoard(log_dir=f"logs/{modelName}")
# t0 = time.time()
# history = nn.fit(X_train, to_categorical(y_train),
#                  validation_data=(X_test, to_categorical(y_test)),
#                  batch_size=32,
#                  epochs=100, verbose=2,
#                  callbacks=[tensboard])
#
# evaluateTraining(nn, t0, modelName, history.history)


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# for numBlocks in [3, 4]:
#     for numLayers in [3, 4]:
# for dropout in [0.1, 0.3]:
numBlocks, numLayers = 3, 3
dropout = 0.3
modelName = f"ResNet1D_{numBlocks}x{numLayers}_onlyNorm_batch128_dropout_{dropout}_allSpecs"
print(modelName)
resnet: ResNet1D = ResNet1D(numFeatures, numClasses, n_blocks=numBlocks, n_layers=numLayers, kSize=3, dropout=dropout)
tensboard: TensorBoard = TensorBoard(log_dir=f"logs/{modelName}")
# modelCheckpoint: ModelCheckpoint = ModelCheckpoint("ResNet_2x2_moreFilters_ModelSavesOnlyNorm64", monitor="val_accuracy")
# resnet.summary()
t0 = time.time()
history2 = resnet.fit(X_train, to_categorical(y_train),
                      validation_data=(X_test, to_categorical(y_test)),
                      batch_size=128,
                      epochs=50, verbose=1,
                      callbacks=[tensboard])

evaluateTraining(resnet, t0, modelName, history2.history)
