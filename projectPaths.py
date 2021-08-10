from PyQt5 import QtCore, QtWidgets
import os


def getAppFolder() -> str:
    """
    Returns a writable locatione, specific for the Imec Evaluator App.
    """
    app = QtWidgets.QApplication.instance()
    if app is None:
        # if it does not exist then a QApplication is created
        app = QtWidgets.QApplication([])

    app.setApplicationName("IMECEvaluator")
    appFolder: str = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.AppLocalDataLocation)
    os.makedirs(appFolder, exist_ok=True)
    return appFolder