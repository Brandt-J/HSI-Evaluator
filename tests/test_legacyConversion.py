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


from unittest import TestCase

from legacyConvert import _preprocNames2NodeClasses
from gui.nodegraph.nodes import *


class TestLegacyConvert(TestCase):
    def test_convertProcessStackToNodeGraph(self) -> None:
        preprocNames: List[str] = ['Derivative1', 'Detrend', 'SNV', 'Normalize']
        correpondingTypes: list = _preprocNames2NodeClasses(preprocNames)
        self.assertEqual(len(correpondingTypes), len(preprocNames))
        self.assertEqual(correpondingTypes[0], NodeSmoothDeriv)
        self.assertEqual(correpondingTypes[1], NodeDetrend)
        self.assertEqual(correpondingTypes[2], NodeSNV)
        self.assertEqual(correpondingTypes[3], NodeNormalize)
