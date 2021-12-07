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

"""
The purpose of these global metrics here is to reuse the built-in keras metrics, but with not-changing names.
If instancing Precision class multiple times, each instance gets a unique name (precision_1, precision_2, ...),
which is unpractical for easy plotting and value comparison across different models.
Reusing the same istance circumvents that issue.
"""

from tensorflow.keras.metrics import Precision, Recall

GlobalPrecision: Precision = Precision()
GlobalRecall: Recall = Recall()
