from unittest import TestCase
import numpy as np
import sys
from PyQt5 import QtWidgets

from gui.spectraPlots import SpectraPreviewWidget


class TestSpectraPreview(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)

    def test_limitSpecNumber(self):
        specPreview: SpectraPreviewWidget = SpectraPreviewWidget()
        maxSpecs: int = 10  # not more than 10 spectra can be taken
        specPreview._numSpecSpinner.setValue(maxSpecs)

        fewSpecs: np.ndarray = np.random.rand(5, 10)
        fewSpecsClipped: np.ndarray = specPreview._limitSpecNumber(fewSpecs)
        self.assertTrue(np.array_equal(fewSpecs, fewSpecsClipped))

        moreSpecs: np.ndarray = np.random.rand(30, 10)
        moreSpecsClipped: np.ndarray = specPreview._limitSpecNumber(moreSpecs)
        self.assertTrue(moreSpecsClipped.shape[0] == maxSpecs)
        self.assertTrue(moreSpecsClipped.shape[1] == moreSpecs.shape[1])
