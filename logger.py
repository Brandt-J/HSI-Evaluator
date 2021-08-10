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

