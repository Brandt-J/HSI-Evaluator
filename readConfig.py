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

defaultPath: str = "config_default.cfg"
customPath: str = "config.cfg"

config = configparser.ConfigParser()
if os.path.exists(customPath):
    config.read(customPath)
else:
    config.read(defaultPath)

sampleDirectory: str = config["PATHS"]["SampleDirectory"]
snapScanFolder: str = config["PATHS"]["SnapscanFolder"]


sqlLogin: Dict[str, str] = {"user": config["SQL Credentials"]["Username"],
                            "password": config["SQL Credentials"]["Password"],
                            "host": config["SQL Credentials"]["Host"],
                            "database": config["SQL Credentials"]["Database"]}

