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
Adapted from: https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py#L37
"""

import time
from typing import List

from tensorflow.keras.layers import Activation, add, Conv1D, GlobalAveragePooling1D, Dense, BatchNormalization, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint


class ResNet1D(Model):
    def __init__(self, numFeatures: int, numClasses: int, n_blocks: int, n_layers: int, kSize: int):
        n_filters = 64

        input_layer = Input((numFeatures, 1))
        for i in range(n_blocks):
            if i == 0:
                output_block = self._addResBlock(32, n_layers, kSize, input_layer)
            else:
                output_block = self._addResBlock(n_filters, n_layers, kSize, output_block)

        # FINAL
        gap_layer = GlobalAveragePooling1D()(output_block)
        output_layer = Dense(numClasses, activation='softmax')(gap_layer)

        super(ResNet1D, self).__init__(inputs=input_layer, outputs=output_layer)

        self.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=[Precision(), Recall()])

    def _addResBlock(self, n_filters: int, kSize: int, n_layers: int, input_layer):
        for i in range(n_layers):
            if i == 0:
                conv = Conv1D(filters=n_filters, kernel_size=kSize, padding='same')(input_layer)
            else:
                conv = Conv1D(filters=n_filters, kernel_size=kSize, padding='same')(conv)
            conv = BatchNormalization()(conv)
            conv = Activation('relu')(conv)

        # expand channels for the sum
        if input_layer.shape[2] != n_filters:
            shortcut = Conv1D(filters=n_filters, kernel_size=1, padding='same')(input_layer)
            shortcut = BatchNormalization()(shortcut)
        else:
            shortcut = BatchNormalization()(input_layer)

        outputLayer = add([shortcut, conv])
        outputLayer = Activation('relu')(outputLayer)
        return outputLayer
