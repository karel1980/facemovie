from PyQt5.QtWidgets import QApplication
import sys
from gui.main import MainWindow

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
