# import sys
# from PyQt5 import QtCore, QtGui, QtWidgets, uic
# from PyQt5.QtCore import Qt
#
#
# class MainWindow(QtWidgets.QMainWindow):
#
#     def __init__(self):
#         super().__init__()
#
#         self.label = QtWidgets.QLabel()
#         canvas = QtGui.QPixmap(400, 300)
#         canvas.fill(Qt.white)
#         self.label.setPixmap(canvas)
#         self.setCentralWidget(self.label)
#
#         self.last_x, self.last_y = None, None
#
#     def mouseMoveEvent(self, e):
#         if self.last_x is None: # First event.
#             self.last_x = e.x()
#             self.last_y = e.y()
#             return # Ignore the first time.
#
#         painter = QtGui.QPainter(self.label.pixmap())
#         painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
#         painter.end()
#         self.update()
#
#         # Update the origin for next time.
#         self.last_x = e.x()
#         self.last_y = e.y()
#
#     def mouseReleaseEvent(self, e):
#         self.last_x = None
#         self.last_y = None
#
#
# app = QtWidgets.QApplication(sys.argv)
# window = MainWindow()
# window.show()
# app.exec_()


import sys
import qpageview
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    v = qpageview.View()
    v.loadPdf("sample.pdf")
    v.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
