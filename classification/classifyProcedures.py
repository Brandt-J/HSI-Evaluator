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
import time
import numpy as np
from typing import *
from enum import Enum
from threading import Event
from dataclasses import dataclass
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import imblearn

from logger import getLogger
from helperfunctions import getRandomSpectraFromArray
from classification.classifiers import ClassificationError, KNN, SVM, NeuralNet
from preprocessing.preprocessors import MSCProc

if TYPE_CHECKING:
    from classification.classifiers import BaseClassifier, BatchClassificationResult
    from gui.sampleview import SampleView
    from preprocessing.preprocessors import Preprocessor, MSCProc
    from particles import Particle
    from spectraObject import SpectraObject
    from dataObjects import Sample
    from logging import Logger


def getClassifiers() -> List['BaseClassifier']:
    """
    Returns a list with the available classifiers.
    """
    return [NeuralNet(), SVM(), KNN()]


class ClassifyMode(Enum):
    """
    Enum for defining whether to run classification on an entire image, or to only classifiy the particles.
    """
    WholeImage = 0
    Particles = 1


class BalanceMode(Enum):
    """
    Mode for choosing a balancing method of the training data.
    """
    NoBalancing = 0
    UnderRandom = 1
    UnderNearMiss = 2
    OverRandom = 3
    OverSMOTE = 4


@dataclass
class TrainingResult:
    """
    Object for transferring classification training (and validation) result
    """
    classifier: 'BaseClassifier'
    validReportString: str
    validReportDict: dict


def trainClassifier(trainSampleList: List['Sample'], classifier: 'BaseClassifier', preprocessors: List['Preprocessor'],
                    maxSpecsPerClass: int, testSize: float, balanceMode: BalanceMode, stopEvent: Event,
                    receiveTrainResultFunc: Callable[[TrainingResult], None]) -> None:
    """
    Method for training the classifier and applying it to the samples. It currently also does the preprocessing.
    The classifier will be put back in the queue after training and validation.
    :param trainSampleList: List of Sample objects used for classifier training
    :param classifier: The Classifier to use.
    :param preprocessors: The preprocessors to use.
    :param maxSpecsPerClass: The maximum number of spectra per class to use.
    :param testSize: Fraction of the data used for testing.
    :param balanceMode: Desired mode for balancing the dataset.
    :param stopEvent: Event that is set if computation should be stopped.
    :param receiveTrainResultFunc: Function to send the training result to.
    """
    if stopEvent.is_set():
        return

    xtrain, xtest, ytrain, ytest = getTestTrainSpectraFromSamples(trainSampleList, maxSpecsPerClass, testSize,
                                                                  balanceMode, preprocessors)

    logger: 'Logger' = getLogger("TrainingProcess")
    # training
    logger.debug(f"starting training on {xtrain.shape[0]} spectra")
    t0 = time.time()
    try:
        classifier.train(xtrain, xtest, ytrain, ytest)
    except Exception as e:
        raise ClassificationError(f"Error during classifier Training: {e}")
    logger.debug(f'Training {classifier.title} on {xtrain.shape[0]} spectra took {round(time.time() - t0, 2)} seconds')
    if stopEvent.is_set():
        return

    # validation
    predResult: 'BatchClassificationResult' = classifier.predict(xtest)
    ypredicted: np.ndarray = predResult.getResults(cutoff=0.0)
    reportDict: dict = classification_report(ytest, ypredicted, output_dict=True, zero_division=0)
    reportStr: str = classification_report(ytest, ypredicted, output_dict=False, zero_division=0)
    logger.info(reportStr)

    receiveTrainResultFunc(TrainingResult(classifier, reportStr, reportDict))


def classifySamples(inferenceSampleList: List['SampleView'], classifier: 'BaseClassifier', mode: 'ClassifyMode',
                    preprocessors: List['Preprocessor'], perParticleBinning: int, stopEvent: Event,
                    iterationCallback: Callable) -> None:
    """
    Method for training the classifier and applying it to the samples. It currently also does the preprocessing.
    :param inferenceSampleList: List of Samples on which we want to run classification.
    :param classifier: The Classifier to use
    :param mode: The classification mode to do.
    :param preprocessors: Preprocessors to use.
    :param perParticleBinning: Binning of spectra, if classification in particle mode.
    :param stopEvent: Event that is Set if the process should be cancelled
    :param iterationCallback: Function that is called after finishing each sample
    """
    for sample in inferenceSampleList:
        if stopEvent.is_set():
            return
        classifySample(sample.getSampleData(), classifier, mode, preprocessors, perParticleBinning)

        if mode == ClassifyMode.WholeImage:
            sample.updateClassImageInGraphView()
        else:
            sample.updateParticlesInGraphUI()

        iterationCallback()


def classifySample(sample: 'Sample', classifier: 'BaseClassifier', mode: ClassifyMode, preprocessors: List['Preprocessor'],
                   perparticleBinning: int) -> None:
    """
    Estimates the classes for each spectrum
    :param sample: The Sample to classifiy
    :param classifier: The Classifier object to use.
    :param mode: The classification mode to apply.
    :param preprocessors: The preprocessor to apply.
    :param perparticleBinning: If mode is particle mode, the number of spectra to bin per particle
    """
    logger: 'Logger' = getLogger("Classifier Inference")

    background: np.ndarray = sample.getAveragedBackgroundSpectrum()
    if mode == ClassifyMode.WholeImage:
        specObject: 'SpectraObject' = sample.specObj
        specArr: np.ndarray = preprocessSpectra(specObject.getSpecArray(), preprocessors, background)
        try:
            batchResult: 'BatchClassificationResult' = classifier.predict(specArr)
        except Exception as e:
            raise ClassificationError(f"Error during classifier inference (image mode): {e}")
        else:
            sample.setBatchResults(batchResult)

    elif mode == ClassifyMode.Particles:
        sample.resetParticleResults()
        particles: List['Particle'] = sample.getAllParticles()
        cube: np.ndarray = sample.getSpecCube()
        if len(particles) == 0:
            logger.warning(f"No particles found in sample {sample.name}, cannot classify them..")
        for particle in particles:
            specArr: np.ndarray = particle.getSpectraArray(cube, binning=perparticleBinning)
            specArr = preprocessSpectra(specArr, preprocessors, background)
            try:
                batchRes: 'BatchClassificationResult' = classifier.predict(specArr)
            except Exception as e:
                raise ClassificationError(f"Error during classifier inference (particle mode): {e}")
            else:
                particle.setBatchResult(batchRes)


def getTestTrainSpectraFromSamples(sampleList: List['Sample'], maxSpecsPerClass: int, testSize: float, balanceMode: BalanceMode,
                                   preprocessors: List['Preprocessor']) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gets all labelled spectra from the indicated sampleview. Spectra and labels are concatenated in one array, each.
    :param sampleList: List of sampleviews to use.
    :param maxSpecsPerClass: Max. number of spectra per class.
    :param testSize: Fraction of the data to use as test size.
    :param balanceMode: Dataset balancing mode to use.
    :param preprocessors: The preprocessors to apply.
    :return: Tuple[Xtrain, Xtest, ytrain, ytest]
    """
    logger: 'Logger' = getLogger("PrepareSpecsForTraining")
    labels: List[str] = []
    spectra: Union[None, np.ndarray] = None
    for sample in sampleList:
        spectraDict: Dict[str, np.ndarray] = sample.getLabelledSpectra()
        background: np.ndarray = sample.getAveragedBackgroundSpectrum()
        for name, specArr in spectraDict.items():
            numSpecs = specArr.shape[0]
            if numSpecs > maxSpecsPerClass:
                specArr = getRandomSpectraFromArray(specArr, maxSpecsPerClass)
                logger.debug(f"Reduced {numSpecs} spectra from {name} to {specArr.shape[0]} spectra")
                numSpecs = maxSpecsPerClass

            classLabels = [name]*numSpecs
            labels += classLabels

            specArr = preprocessSpectra(specArr, preprocessors, background, labels=classLabels)

            if spectra is None:
                spectra = specArr
            else:
                spectra = np.vstack((spectra, specArr))

    labels: np.ndarray = np.array(labels)

    if balanceMode == BalanceMode.UnderRandom:
        spectra, labels = imblearn.under_sampling.RandomUnderSampler().fit_resample(spectra, labels)
    elif balanceMode == BalanceMode.UnderNearMiss:
        spectra, labels = imblearn.under_sampling.NearMiss().fit_resample(spectra, labels)
    elif balanceMode == BalanceMode.OverRandom:
        spectra, labels = imblearn.over_sampling.RandomOverSampler().fit_resample(spectra, labels)
    elif balanceMode == BalanceMode.OverSMOTE:
        spectra, labels = imblearn.over_sampling.SMOTE().fit_resample(spectra, labels)

    return train_test_split(spectra, labels, test_size=testSize, random_state=42)


def createClassImg(cubeShape: tuple, assignments: np.ndarray, colorCodes: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    """
    Creates an overlay image of the current classification
    :param cubeShape: Shape of the cube array
    :param assignments: Array of class names for each pixel
    :param colorCodes: Dictionary mapping class names to rgb values
    :return: np.ndarray of RGBA image as classification overlay
    """
    clfImg: np.ndarray = np.zeros((cubeShape[1], cubeShape[2], 4), dtype=np.uint8)
    i: int = 0
    for y in range(cubeShape[1]):
        for x in range(cubeShape[2]):
            clfImg[y, x, :3] = colorCodes[assignments[i]]
            clfImg[y, x, 3] = 255
            i += 1

    return clfImg


def preprocessSpectra(specArr: np.ndarray, preprocessors: List['Preprocessor'], background: np.ndarray,
                      labels: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Applies the specified preprocessing to the spectra array.
    :param specArr: (MxN) shape array of M spectra with N wavelenghts.
    :param preprocessors: List of preprocessors to apply
    :param background: Averaged background spectrum.
    :param labels: Optional: Array of M labels
    :return: preprocessed spectra array
    """
    specArr = specArr.copy()  # We don't want to override any original data...
    for preprocessor in preprocessors:
        if type(preprocessor) == MSCProc:
            preprocessor: MSCProc = cast(MSCProc, preprocessor)
            specArr = preprocessor.applyToSpectra(specArr, labels)
        else:
            specArr = preprocessor.applyToSpectra(specArr)

    return specArr


def splitUpArray(specArr: np.ndarray, numParts: int = 8) -> List[np.ndarray]:
    """
    Splits up the given array into a list of arrays.
    :param specArr: (NxM) shape array of N spectra with w wavelenghts.
    :param numParts: number of parts
    :param: List with numParts arrays.
    """
    arrList: List[np.ndarray] = []
    stepSize: int = specArr.shape[0] // numParts + 1
    for i in range(numParts):
        start = i*stepSize
        end = min([(i+1)*stepSize, specArr.shape[0]])
        arrList.append(specArr[start:end, :])
    return arrList
