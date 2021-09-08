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

from typing import TYPE_CHECKING, List

from spectraObject import SpectraObject
from logger import getLogger

if TYPE_CHECKING:
    from logging import Logger
    from dataObjects import Sample, View

currentSampleVersion = 1
currentViewVersion = 1

logger: 'Logger' = getLogger("LegacyConvert")


def assertUpToDateSample(sampleData: 'Sample') -> 'Sample':
    """Convenience method for converting from older versions"""

    if not hasattr(sampleData, "specObj"):
        sampleData = _updateSampleToVersion0(sampleData)

    if not hasattr(sampleData, "version"):
        sampleData = _updateSampleToVersion1(sampleData)

    return sampleData


def assertUpToDateView(view: 'View') -> 'View':
    """Convenience method for converting from older versions"""

    if not hasattr(view, "version"):
        view = _updateViewToVersion1(view)

    return view


def _updateSampleToVersion0(sampledata: 'Sample') -> 'Sample':
    logger.info(f"Converting sample {sampledata.name} to Version 0")
    sampledata.specObj = SpectraObject()
    sampledata.specObj._wavenumbers = None
    return sampledata


def _updateSampleToVersion1(sampleData: 'Sample') -> 'Sample':
    logger.info(f"Converting sample {sampleData.name} to Version 1")
    sampleData.specObj._wavelengths = sampleData.specObj._wavenumbers
    del sampleData.specObj._wavenumbers
    sampleData.version = 1
    return sampleData


def _updateViewToVersion1(view: 'View') -> 'View':
    logger.info(f"Converting view {view.title} to Version 1")
    from dataObjects import Sample  # import statement down here to avoid cyclic reference...
    samples: List['Sample'] = []
    for sample in view.samples:
        newSample: Sample = Sample()
        sample = assertUpToDateSample(sample)
        newSample.__dict__.update(sample.__dict__)
        samples.append(newSample)
    view.samples = samples
    return view
