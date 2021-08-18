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

import logging
import os
from typing import List
from projectPaths import getAppFolder


createdLoggers: List[str] = []


def getLogger(name: str = 'DefaultLogger') -> logging.Logger:
    logger: logging.Logger = logging.getLogger(name)
    if name not in createdLoggers:
        _configureLogger(logger)
        createdLoggers.append(name)
    return logger


def _configureLogger(logger: logging.Logger) -> None:
    logger.setLevel(logging.DEBUG)

    _logPath: str = os.path.join(getAppFolder(), "Logging")
    os.makedirs(_logPath, exist_ok=True)
    _fileHandler = logging.FileHandler(os.path.join(_logPath, "log.txt"))
    _consoleHandler = logging.StreamHandler()
    _fileHandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    _consoleHandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    logger.addHandler(_fileHandler)
    logger.addHandler(_consoleHandler)

