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
import json
import os
import mysql.connector
from typing import Union, TYPE_CHECKING, List

from projectPaths import getAppFolder
from logger import getLogger

if TYPE_CHECKING:
    from logging import Logger


class DBConnection:
    def __init__(self):
        self._connection: Union[None, mysql.connector.connection.MySQLConnection] = None
        self._logger: 'Logger' = getLogger("SQL Connection")

    def connect(self) -> None:
        """
        Establishes a connection to the database.
        """
        try:
            config: dict = self._getConfigDict()
        except FileNotFoundError as e:
            self._logger.critical(f"Configuration file for SQL Connection not found: {e}")
            raise ConnectionError(e)

        try:
            self._connection = mysql.connector.connect(**config)
            self._logger.info("Successfully connected to database.")
        except mysql.connector.errors.ProgrammingError as e:
            self._logger.critical(f"Connection error: {e}")
            raise ConnectionError(e)

    def disconnect(self) -> None:
        """
        Closes the connection to the SQL Database.
        """
        if self._connection is not None:
            self._connection.disconnect()
        self._connection = None
        self._logger.info("Disconnected from database.")

    def _assertConnection(self) -> None:
        """
        Used for asserting that a valid connection is present.
        """
        if self._connection is None:
            self.connect()  # Will raise exception if not possible

    def getSamples(self) -> List[str]:
        """
        Gets List of SampleNames
        """
        

    def _getConfigDict(self) -> dict:
        path: str = os.path.join(getAppFolder(), "dbconfig.txt")
        config: dict = {}
        if os.path.exists(path):
            with open(path, "r") as fp:
                config = json.load(fp)
        else:
            raise FileNotFoundError(f"No database config found at {path}")
        return config