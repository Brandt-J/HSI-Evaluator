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

import configparser
import os

__all__ = ["sampleDirectory", "snapScanFolder", "sqlLogin"]

from typing import Dict

from logger import getLogger

logger = getLogger("ConfigReader")

defaultFile: str = "config_default.cfg"
customFile: str = "config.cfg"

config = configparser.ConfigParser()
configLoaded, maxAttempts = False, 5
counter: int = 0
folder: str = os.getcwd()
while not configLoaded and counter < maxAttempts:
    defaultPath: str = os.path.join(folder, defaultFile)
    customPath: str = os.path.join(folder, customFile)

    if os.path.exists(customPath):
        config.read(customPath)
        configLoaded = True
        logger.info(f"Reading config from custom file.")
    elif os.path.exists(defaultPath):
        config.read(defaultPath)
        configLoaded = True
        logger.info(f"Reading config from default file.")

    if not configLoaded:
        folder = os.path.dirname(folder)
        logger.info(f"Could not find config files in directory {folder}, retrying in parent dir.")
    counter += 1

if not configLoaded:
    logger.critical(f"Did not find custom or default logfile. These should be in the HSI-Evaluator main diretory.\n"
                    f"Please do not move them, especially not the default file. They should be named:\n"
                    f"'{defaultFile}' for the default config and '{customFile}' for the custom config.")
    raise FileNotFoundError("Config Files could not be laoded. See readConfig.py for details.")

sampleDirectory: str = config["PATHS"]["SampleDirectory"]
snapScanFolder: str = config["PATHS"]["SnapscanFolder"]


sqlLogin: Dict[str, str] = {"user": config["SQL Credentials"]["Username"],
                            "password": config["SQL Credentials"]["Password"],
                            "host": config["SQL Credentials"]["Host"],
                            "database": config["SQL Credentials"]["Database"]}

