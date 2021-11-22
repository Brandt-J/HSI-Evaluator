import random
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import imblearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

for i in range(len(assignments)):
    assignments[i] = assignments[i].replace("weathered", "")
    assignments[i] = assignments[i].replace("pristine", "")
    assignments[i] = assignments[i].replace("blue", "")
    assignments[i] = assignments[i].replace("white", "")
    assignments[i] = assignments[i].replace("_", "")

assignments[assignments == "sediment A"] = "Sediment"
assignments[assignments == "sediment B"] = "Sediment"
assignments[assignments == "sediment C"] = "Sediment"
indNoAmber: np.ndarray = np.where(assignments != "Amber")[0]
specs = specs[indNoAmber, :]
assignments = assignments[indNoAmber]
print("raw", Counter(assignments))


specs = deriv_smooth(specs, windowSize=13, polydegree=3, derivative=1)
specs = normalizeIntensities(specs, NormMode.Length)

encoder: LabelEncoder = LabelEncoder().fit(assignments)
y: np.ndarray = encoder.transform(assignments)

X_train, X_test, y_train, y_test = train_test_split(specs, y, test_size=0.2)
X_train, y_train = imblearn.over_sampling.SMOTE().fit_resample(X_train, y_train)
X_test, y_test = imblearn.under_sampling.RandomUnderSampler().fit_resample(X_test, y_test)
print("balanced train:", Counter(y_train))
print("balanced test:", Counter(y_test))

dropout = 0.1
numNeurons = [200, 100, 50]

modelName: str = "DenseNet_with_Preproc"
print(modelName)

numFeatures, numClasses = specs.shape[1], len(np.unique(assignments))
nn: NeuralNetClf = NeuralNetClf(numFeatures, numClasses,
                                numNeuronsPerHiddenLayer=numNeurons, dropout=dropout)
tensboard: TensorBoard = TensorBoard(log_dir=f"logs/{modelName}")
t0 = time.time()
history = nn.fit(X_train, to_categorical(y_train),
                 validation_data=(X_test, to_categorical(y_test)),
                 batch_size=64,
                 epochs=50, verbose=1,
                 callbacks=[tensboard])
print(f"training NN took {time.time()-t0} seconds")
t0 = time.time()
pred: np.ndarray = nn.predict(X_test)
pred = np.array([pred[i, :].argmax() for i in range(pred.shape[0])])
print(f"Pred NN: {time.time()-t0} seconds\n", classification_report(encoder.inverse_transform(y_test),
                                                                    encoder.inverse_transform(pred),
                                                                    zero_division=0))


modelName = "ResNet1D_with_Preproc"
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
resnet: ResNet1D = ResNet1D(numFeatures, numClasses, n_blocks=6, n_layers=4, kSize=3)
tensboard: TensorBoard = TensorBoard(log_dir=f"logs/{modelName}")
modelCheckpoint: ModelCheckpoint = ModelCheckpoint("ResNetModelSavesWithPreprocess", monitor="val_accuracy")
# resnet.summary()
t0 = time.time()
history = resnet.fit(X_train, to_categorical(y_train),
                     validation_data=(X_test, to_categorical(y_test)),
                     batch_size=64,
                     epochs=100, verbose=1,
                     callbacks=[tensboard, modelCheckpoint])
print(f"training Resnet took {time.time()-t0} seconds")
t0 = time.time()
pred: np.ndarray = resnet.predict(X_test)
pred = np.array([pred[i, :].argmax() for i in range(pred.shape[0])])
print(f"Pred Resnet: {time.time()-t0} seconds\n", classification_report(encoder.inverse_transform(y_test),
                                                                    encoder.inverse_transform(pred),
                                                                    zero_division=0))


# modelName: str = f"Conv Network_20epochs"
# print(modelName)
# nn: ConvNeuralNetCLF = ConvNeuralNetCLF(specs.shape[1], len(np.unique(assignments)))
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
#
# tensboard: TensorBoard = TensorBoard(log_dir=f"logs/{modelName}")
# t0 = time.time()
# history = nn.fit(X_train, to_categorical(y_train),
#                  validation_data=(X_test, to_categorical(y_test)),
#                  batch_size=64,
#                  epochs=20, verbose=1,
#                  callbacks=[tensboard])
# print(f"training NN took {time.time()-t0} seconds")
# t0 = time.time()
# pred: np.ndarray = nn.predict(X_test)
# pred = np.array([pred[i, :].argmax() for i in range(pred.shape[0])])
# print(f"Pred NN: {time.time()-t0} seconds\n", classification_report(encoder.inverse_transform(y_test),
#                                                                     encoder.inverse_transform(pred),
#                                                                     zero_division=0))
