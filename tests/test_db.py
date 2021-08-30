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
import mysql.connector
from database.database import DBConnection


def testRaiseFunc():
    raise FileNotFoundError("Config not found")


class TestDatabase(TestCase):

    def testConnection(self):
        conn: DBConnection = DBConnection()
        self.assertTrue(conn._connection is None)
        conn.connect()
        self.assertEqual(type(conn._connection), mysql.connector.connection.MySQLConnection)
        conn.disconnect()
        self.assertTrue(conn._connection is None)

        conn._getConfigDict = testRaiseFunc
        self.assertRaises(ConnectionError, conn.connect)
