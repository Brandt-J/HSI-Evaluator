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
from multiprocessing import Queue, Event
import numpy as np
from typing import *
from dataclasses import dataclass

from classification.classifiers import BaseClassifier, ClassificationError, KNN, SVM
from logger import getLogger
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from spectraObject import SpectraObject
    from dataObjects import Sample
    from logging import Logger


def getClassifiers() -> List['BaseClassifier']:
    # return [NeuralNet(), SVM(), RDF()]
    return [SVM(), KNN()]


@dataclass
class TrainingResult:
    """
    Object for transferring classification training (and validation) result
    """
    classifier: 'BaseClassifier'
    validReportString: str
    validReportDict: dict


def trainClassifier(trainSampleList: List['Sample'], classifier: 'BaseClassifier', testSize: float, queue: Queue,
                    stopEvent: Event) -> None:
    """
    Method for training the classifier and applying it to the samples. It currently also does the preprocessing.
    The classifier will be put back in the queue after training and validation.
    :param trainSampleList: List of Sample objects used for classifier training
    :param classifier: The Classifier to use
    :param testSize: Fraction of the data used for testing
    :param queue: Dataqueue for communication between processes.
    :param stopEvent: Event that is set if computation should be stopped.
    """
    trainingSpectra: Dict[str, np.ndarray] = {}
    for sample in trainSampleList:
        for cls, specs in sample.getLabelledSpectra().items():
            if cls not in trainingSpectra:
                trainingSpectra[cls] = specs
            else:
                trainingSpectra[cls] = np.vstack((trainingSpectra[cls], specs))

    if stopEvent.is_set():
        return

    logger: 'Logger' = getLogger("TrainingProcess")
    # training
    xtrain, xtest, ytrain, ytest = getTestTrainSpectraFromSamples(trainSampleList, testSize)
    t0 = time.time()
    try:
        classifier.train(xtrain, xtest, ytrain, ytest)
    except Exception as e:
        queue.put(ClassificationError(f"Error during classifier Trining: {e}"))
        raise ClassificationError(f"Error during classifier Trining: {e}")
    logger.debug(f'Training {classifier.title} on {xtrain.shape[0]} spectra took {round(time.time() - t0, 2)} seconds')
    if stopEvent.is_set():
        return

    # validation
    ypredicted = classifier.predict(xtest)
    reportDict: dict = classification_report(ytest, ypredicted, output_dict=True)
    reportStr: str = classification_report(ytest, ypredicted, output_dict=False)
    logger.info(reportStr)
    queue.put(TrainingResult(classifier, reportStr, reportDict))


def classifySamples(inferenceSampleList: List['Sample'], classifier: 'BaseClassifier', colorDict: Dict[str, Tuple[int, int, int]],
                    queue: Queue, stopEvent: Event) -> None:
    """
    Method for training the classifier and applying it to the samples. It currently also does the preprocessing.
    :param inferenceSampleList: List of Samples on which we want to run classification.
    :param classifier: The Classifier to use
    :param colorDict: Dictionary mapping all classes to RGB values, used for image generation
    :param queue: Dataqueue for communication between processes.
    :param stopEvent: Event that is Set if the process should be cancelled
    """
    logger: 'Logger' = getLogger("Classifier Application")
    finishedSamples: List['Sample'] = []
    for i, sample in enumerate(inferenceSampleList):
        t0 = time.time()
        logger.debug(f"Starting classifcation on {sample.name}")
        specObj = sample.specObj
        if stopEvent.is_set():
            return

        try:
            assignments: List[str] = getClassesForPixels(specObj, classifier, ignoreBackground=False)
        except Exception as e:
            queue.put(ClassificationError(e))
            raise ClassificationError(e)

        if stopEvent.is_set():
            return

        cubeShape = specObj.getCube().shape
        clfImg: np.ndarray = createClassImg(cubeShape, assignments, colorDict)
        sample.setClassOverlay(clfImg)
        logger.debug(f'Finished classification on sample {sample.name} in {round(time.time()-t0, 2)} seconds'
                     f' ({i+1} of {len(inferenceSampleList)} samples done)')
        finishedSamples.append(sample)
        queue.put("finished sample")

    queue.put(finishedSamples)


def getClassesForPixels(specObject: 'SpectraObject', classifier: 'BaseClassifier', ignoreBackground: bool) -> List[str]:
    """
    Estimates the classes for each pixel
    :param specObject: The spectraObject to use
    :param classifier: The classifier to use
    :param ignoreBackground: Whether or not to ignore background pixels
    :return: List of class names per spectrum
    """
    specList: List[np.ndarray] = []
    cube: np.ndarray = specObject.getCube()
    backgroundIndices: Set[int] = specObject.getBackgroundIndices()
    i: int = 0
    for y in range(cube.shape[1]):
        for x in range(cube.shape[2]):
            if not ignoreBackground or (ignoreBackground and i not in backgroundIndices):
                specList.append(cube[:, y, x])
            i += 1

    specArr: np.ndarray = np.array(specList)
    try:
        result: np.ndarray = classifier.predict(specArr)
    except Exception as e:
        raise ClassificationError("Error during classifier inference: {e}")
    return list(result)


def getTestTrainSpectraFromSamples(sampleList: List['Sample'], testSize: float,
                                   ignoreBackground: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gets all labelled spectra from the indicated sampleview. Spectra and labels are concatenated in one array, each.
    :param sampleList: List of sampleviews to use
    :param testSize: Fraction of the data to use as test size
    :param ignoreBackground: Whether or not to skip background pixels
    :return: Tuple[Xtrain, Xtest, ytrain, ytest]
    """
    labels: List[str] = []
    spectra: Union[None, np.ndarray] = None
    for sample in sampleList:
        spectraDict: Dict[str, np.ndarray] = sample.getLabelledSpectra()
        for name, specs in spectraDict.items():
            if ignoreBackground and name.lower() == "background":
                continue

            numSpecs = specs.shape[0]
            labels += [name]*numSpecs
            if spectra is None:
                spectra = specs
            else:
                spectra = np.vstack((spectra, specs))

    labels: np.ndarray = np.array(labels)
    return train_test_split(spectra, labels, test_size=testSize, random_state=42)


def createClassImg(cubeShape: tuple, assignments: List[str], colorCodes: Dict[str, Tuple[int, int, int]],
                   ignoreIndices: Set[int] = set()) -> np.ndarray:
    """
    Creates an overlay image of the current classification
    :param cubeShape: Shape of the cube array
    :param assignments: List of class names for each pixel
    :param colorCodes: Dictionary mapping class names to rgb values
    :param ignoreIndices: Set of pixel indices to ignore (i.e., background pixels)
    :return: np.ndarray of RGBA image as classification overlay
    """
    clfImg: np.ndarray = np.zeros((cubeShape[1], cubeShape[2], 4), dtype=np.uint8)
    i: int = 0  # counter for cube
    j: int = 0  # counter for assignment List
    t0 = time.time()
    for y in range(cubeShape[1]):
        for x in range(cubeShape[2]):
            if i not in ignoreIndices:
                clfImg[y, x, :3] = colorCodes[assignments[j]]
                clfImg[y, x, 3] = 255
                j += 1
            i += 1

    print('generating class image', round(time.time()-t0, 2))
    return clfImg
