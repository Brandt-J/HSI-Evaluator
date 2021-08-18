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
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2

from SpectraProcessing.classification import NeuralNetClassifier
from SpectraProcessing.Preprocessing.balancing import BalanceMode, balanceDataset


spectra: np.ndarray = np.load("gui/Spectra of Particles4Types_corrected.npy")
assignments: np.ndarray = np.load("gui/Assignments of Particles4Types_corrected.npy")


indBackground = np.where(assignments == 0)[0]
avg = np.mean(spectra[indBackground, :], axis=0)
avg = (avg - avg.min()) / (avg.max() - avg.min())

for i in range(spectra.shape[0]):
    curSpec = spectra[i, :].copy()
    curSpec = (curSpec - curSpec.min()) / (curSpec.max() - curSpec.min())
    curSpec -= avg
    spectra[i, :] = curSpec


for k in [100, 50, 20, 10, 5]:
    print(f'reducing to {k} features')
    condensed = spectra.copy()
    minVal = condensed.min()
    condensed -= minVal
    condensed = SelectKBest(chi2, k=k).fit_transform(condensed, assignments)
    condensed += minVal
    specOrig, assignOrig = condensed.copy(), assignments.copy()

    X, y = balanceDataset(condensed, assignments, BalanceMode.OVER_SMOTE)

    clf: NeuralNetClassifier = NeuralNetClassifier(condensed.shape[1], np.max(assignments)+1)
    clf.trainWithSpectra(X, y)
    results = clf.evaluateSpectra(specOrig)
    print(classification_report(assignOrig, np.array(results)))
