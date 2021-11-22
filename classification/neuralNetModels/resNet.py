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


class ResNet1D(Model):
    def __init__(self, numFeatures: int, numClasses: int):
        n_feature_maps = 64

        input_layer = Input((numFeatures, 1))

        # BLOCK 1
        conv_x = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_1 = add([shortcut_y, conv_z])
        output_block_1 = Activation('relu')(output_block_1)

        # BLOCK 2
        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_2 = add([shortcut_y, conv_z])
        output_block_2 = Activation('relu')(output_block_2)

        # BLOCK 3
        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = BatchNormalization()(output_block_2)

        output_block_3 = add([shortcut_y, conv_z])
        output_block_3 = Activation('relu')(output_block_3)

        # FINAL
        gap_layer = GlobalAveragePooling1D()(output_block_3)
        output_layer = Dense(numClasses, activation='softmax')(gap_layer)

        super(ResNet1D, self).__init__(inputs=input_layer, outputs=output_layer)

        self.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=[Precision(), Recall()])
        #
        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        #
        # file_path = self.output_directory + 'best_model.hdf5'
        #
        # model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
        #                                                    save_best_only=True)
        #
        # self.callbacks = [reduce_lr, model_checkpoint]

        # return model