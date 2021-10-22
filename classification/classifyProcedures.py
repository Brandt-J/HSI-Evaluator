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
from multiprocessing import Queue, Event
from dataclasses import dataclass
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from logger import getLogger
from helperfunctions import getRandomSpectraFromArray
from classification.classifiers import ClassificationError, KNN, SVM, NeuralNet

if TYPE_CHECKING:
    from classification.classifiers import BaseClassifier, BatchClassificationResult
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


@dataclass
class TrainingResult:
    """
    Object for transferring classification training (and validation) result
    """
    classifier: 'BaseClassifier'
    validReportString: str
    validReportDict: dict


def trainClassifier(trainSampleList: List['Sample'], classifier: 'BaseClassifier', maxSpecsPerClass: int,
                    testSize: float, queue: Queue, stopEvent: Event) -> None:
    """
    Method for training the classifier and applying it to the samples. It currently also does the preprocessing.
    The classifier will be put back in the queue after training and validation.
    :param trainSampleList: List of Sample objects used for classifier training
    :param classifier: The Classifier to use
    :param maxSpecsPerClass: The maximum number of spectra per class to use
    :param testSize: Fraction of the data used for testing
    :param queue: Dataqueue for communication between processes.
    :param stopEvent: Event that is set if computation should be stopped.
    """
    if stopEvent.is_set():
        return

    logger: 'Logger' = getLogger("TrainingProcess")
    # training
    xtrain, xtest, ytrain, ytest = getTestTrainSpectraFromSamples(trainSampleList, maxSpecsPerClass, testSize)
    logger.debug(f"starting training on {xtrain.shape[0]} spectra")
    t0 = time.time()
    try:
        classifier.train(xtrain, xtest, ytrain, ytest)
    except Exception as e:
        queue.put(ClassificationError(f"Error during classifier Training: {e}"))
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

    classifier.makePickleable()
    queue.put(TrainingResult(classifier, reportStr, reportDict))


def classifySamples(inferenceSampleList: List['Sample'], classifier: 'BaseClassifier', mode: 'ClassifyMode',
                    queue: Queue, stopEvent: Event) -> None:
    """
    Method for training the classifier and applying it to the samples. It currently also does the preprocessing.
    :param inferenceSampleList: List of Samples on which we want to run classification.
    :param classifier: The Classifier to use
    :param mode: The classification mode to do.
    :param queue: Dataqueue for communication between processes.
    :param stopEvent: Event that is Set if the process should be cancelled
    """
    finishedSamples: List['Sample'] = []
    for sample in inferenceSampleList:
        if stopEvent.is_set():
            return
        completed: 'Sample' = classifySample(sample, classifier, mode, queue)
        finishedSamples.append(completed)
        queue.put("finished Sample")
    queue.put(finishedSamples)


def classifySample(sample: 'Sample', classifier: 'BaseClassifier', mode: ClassifyMode, queue: 'Queue') -> 'Sample':
    """
    Estimates the classes for each spectrum
    :param sample: The Sample to classifiy
    :param classifier: The Classifier object to use
    :param mode: The classification mode to apply
    :param queue: The dataqueue to push errors to
    """
    logger: 'Logger' = getLogger("Classifier Application")

    if mode == ClassifyMode.WholeImage:
        specObject: 'SpectraObject' = sample.specObj
        specArr = specObject.getPreprocessedSpecArr()
        try:
            batchResult: 'BatchClassificationResult' = classifier.predict(specArr)
        except Exception as e:
            error: ClassificationError = ClassificationError(f"Error during classifier inference (image mode): {e}")
            queue.put(error)

        else:
            sample.setBatchResults(batchResult)

    elif mode == ClassifyMode.Particles:
        sample.resetParticleResults()
        particles: List['Particle'] = sample.getAllParticles()
        cube: np.ndarray = sample.getPreprocessedSpecCube()
        if len(particles) == 0:
            logger.warning(f"No particles found in sample {sample.name}, cannot classify them..")
        for particle in particles:
            specArr: np.ndarray = particle.getSpectra(cube)
            try:
                batchRes: 'BatchClassificationResult' = classifier.predict(specArr)
            except Exception as e:
                error: ClassificationError = ClassificationError(f"Error during classifier inference (particle mode): {e}")
                raise error
            else:
                particle.setBatchResult(batchRes)

    return sample


def getTestTrainSpectraFromSamples(sampleList: List['Sample'], maxSpecsPerClass: int, testSize: float,
                                   ignoreBackground: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gets all labelled spectra from the indicated sampleview. Spectra and labels are concatenated in one array, each.
    :param sampleList: List of sampleviews to use
    :param maxSpecsPerClass: Max. number of spectra per class
    :param testSize: Fraction of the data to use as test size
    :param ignoreBackground: Whether or not to skip background pixels
    :return: Tuple[Xtrain, Xtest, ytrain, ytest]
    """
    logger: 'Logger' = getLogger("PrepareSpecsForTraining")
    labels: List[str] = []
    spectra: Union[None, np.ndarray] = None
    for sample in sampleList:
        spectraDict: Dict[str, np.ndarray] = sample.getLabelledPreprocessedSpectra()
        for name, specs in spectraDict.items():
            if ignoreBackground and name.lower() == "background":
                continue

            numSpecs = specs.shape[0]
            if numSpecs > maxSpecsPerClass:
                specs = getRandomSpectraFromArray(specs, maxSpecsPerClass)
                logger.debug(f"Reduced {numSpecs} spectra from {name} to {specs.shape[0]} spectra")
                numSpecs = maxSpecsPerClass

            labels += [name]*numSpecs
            if spectra is None:
                spectra = specs
            else:
                spectra = np.vstack((spectra, specs))

    labels: np.ndarray = np.array(labels)
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
    i: int = 0  # counter for cube
    t0 = time.time()
    for y in range(cubeShape[1]):
        for x in range(cubeShape[2]):
            clfImg[y, x, :3] = colorCodes[assignments[i]]
            clfImg[y, x, 3] = 255
            i += 1

    print('generating class image', round(time.time()-t0, 2))
    return clfImg
