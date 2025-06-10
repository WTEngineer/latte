#!/usr/bin/env python
# -*- coding: latin1 -*-

import sys
import os
import fitz
import time
from PIL import Image, ImageDraw
from datetime import datetime, date
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import pyqtSignal, Qt
import cv2
import configparser
from threading import Thread
from face_det import FaceDetector
from face_recog import FaceRecognizer
from mrz_reader import MRZReader
import numpy as np
import math
import sqlite3
import ctypes
from gov_request import connect_gov

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# Light stylesheet
L_BG = "background-color: rgb(245,245,245);\n" \
           "color: rgb(80, 80, 80);\n" \
           "font: 14pt \"Arial\";" \


T_BG = "background-color: rgba(0,0,0,220);\n" \
           "color: rgb(80, 80, 80);\n" \
           "font: 14pt \"Arial\";" \


L_BG_T = "background-color: rgb(225,225,225);\n" \
           "color: rgb(80, 80, 80);\n" \
           "font: 22pt \"Arial\";" \


L_BG_L = "background-color: rgb(225,225,225);\n" \
           "color: rgb(80, 80, 80);\n" \
           "font: 12pt \"Arial\";" \


L_BG_W_1 = "background-color: rgb(225,225,225);\n" \
           "color: rgb(255, 80, 80);\n" \
           "font: 22pt \"Arial\";" \
           "font: Bold;" \

L_BG_W_2 = "background-color: rgb(225,225,225);\n" \
           "color: rgb(250,250,250);\n" \
           "font: 22pt \"Arial\";" \
           "font: Bold;" \


L_BG_R = "background-color: rgb(240,235,235);\n" \
           "color: rgb(80, 80, 80);\n" \
           "font: 22pt \"Arial\";" \

L_BG_F = "background-color: rgb(255,255,255);\n" \
           "color: rgb(80, 80, 80);\n" \
           "font: 8pt \"Arial\";" \
            "border-color: rgb(255, 255, 255);" \
            "border-style: solid;" \
            "border-width: 2px;" \

L_BG_PERSON = "background-color: rgb(255,255,255);\n" \
           "color: rgb(80, 80, 80);\n" \
           "font: 8pt \"Arial\";" \
            "border-color: rgb(220, 220, 220);" \
            "border-style: solid;" \
            "border-width: 4px;" \

L_DETAIL_TEXT = "background-color: rgb(255,255,255);\n" \
           "color: rgb(100, 100, 100);\n" \
           "font: 10pt \"Arial\";" \

L_DETAIL_VALUE = "background-color: rgb(255,255,255);\n" \
           "color: rgb(80, 80, 80);\n" \
           "font: 10pt \"Arial\";" \


L_CAM = "background-color: rgb(180,180,180);\n" \
        "color: rgb(80, 80, 80);\n" \
        "font: 10pt \"Arial\";" \
        "border-color: rgb(255, 255, 255);" \
        "border-style: solid;" \
        "border-width: 1px;" \

L_CAM_ICON = "background-color: rgb(180,180,180);" \
            "border-color: rgb(210, 210, 210);" \
            "border-style: solid;" \
            "border-radius: 34;" \
            "border-width: 0px;" \

L_BT_L = "QLabel { background-color: rgb(200, 200, 200);" \
        "color: rgb(80, 80, 80);" \
        "font: 12pt \"Arial\";" \
        "border-color: rgb(255, 255, 255);" \
        "border-style: solid;" \
        "border-width: 2px;}" \
        "QLabel::hover { background-color : rgb(220, 220, 220);}"\
        "QLabel::pressed {background-color: rgb(240, 240, 240); \
        border-color: rgb(200, 200, 200); \
        border-style: solid;}"

L_BT_N = "QPushButton { background-color: rgb(200, 200, 200);" \
        "color: rgb(80, 80, 80);" \
        "font: 12pt \"Arial\";" \
        "border-color: rgb(255, 255, 255);" \
        "border-style: solid;" \
        "border-radius: 10;" \
        "border-width: 2px;}" \
        "QPushButton::hover { background-color : rgb(220, 220, 220);}"\
        "QPushButton::pressed {background-color: rgb(240, 240, 240); \
        border-color: rgb(200, 200, 200); \
        border-style: solid;}"


L_BT_P = "background-color: rgb(255, 255, 255);" \
            "color: rgb(50, 50, 255);" \
            "font: 12pt \"Arial\";" \
            "border-color: rgb(255, 255, 255);" \
            "border-style: solid;" \
            "border-radius: 10;" \
            "border-width: 2px;" \


L_BT_TAB1 = "background-color: rgb(225, 225, 225);" \
            "color: rgb(80, 80, 80);" \
            "font: 12pt \"Arial\";" \
            "border-color: rgb(225, 225, 225);" \
            "border-style: solid;" \
            "border-radius: 10;" \
            "border-width: 2px;" \

L_BT_TAB2 = "background-color: rgb(245, 245, 245);" \
            "color: rgb(80, 80, 80);" \
            "font: 12pt \"Arial\";" \
            "border-color: rgb(225, 225, 225);" \
            "border-style: solid;" \
            "border-radius: 10;" \
            "border-width: 2px;" \


L_ST_TEXT = "background-color: rgb(225, 225, 225);" \
            "color: rgb(80, 80, 80);" \
            "font: 12pt \"Arial\";" \

L_ST_VAL = "background-color: rgb(255, 255, 255);" \
            "color: rgb(80, 80, 80);" \
            "font: 12pt \"Arial\";" \


spcial_char_map = {ord('ä'): 'ae', ord('ü'): 'ue', ord('ö'): 'oe', ord('ß'): 'ss', ord('Ä'): 'Ae', ord('Ü'): 'Ue',
                   ord('Ö'): 'Oe'}


class FaceRecogSys(QtWidgets.QMainWindow):
    cam1_capture_signal = pyqtSignal()
    cam2_capture_signal = pyqtSignal()
    cam3_capture_signal = pyqtSignal()
    cam4_capture_signal = pyqtSignal()

    register_capture_signal = pyqtSignal()

    update_detected_persons_signal = pyqtSignal()
    add_person_history_signal = pyqtSignal()

    def __init__(self):
        super(FaceRecogSys, self).__init__()

        ###############################################
        self.setWindowTitle("EOAS - Echtzeit Oasis Abfrage System")

        self.setGeometry(int(0 * X_SCALE), int(0 * Y_SCALE), int(800 * X_SCALE), int(600 * Y_SCALE))
        self.setStyleSheet(L_BG)

        self.top_label = QtWidgets.QLabel(self)
        self.top_label.setText("EOAS - Echtzeit Oasis Abfrage System")
        self.top_label.setAlignment(QtCore.Qt.AlignCenter)
        self.top_label.setGeometry(int(0 * X_SCALE), int(0 * Y_SCALE), int(800 * X_SCALE), int(40 * Y_SCALE))
        self.top_label.setStyleSheet(L_BG_T)

        self.warning_unregister_message = QtWidgets.QLabel(self)
        self.warning_unregister_message.setAlignment(QtCore.Qt.AlignCenter)
        self.warning_unregister_message.setText("Person nicht registriert !")
        self.warning_unregister_message.setGeometry(int(600 * X_SCALE), int(0 * Y_SCALE), int(200 * X_SCALE), int(40 * Y_SCALE))
        self.warning_unregister_message.setStyleSheet(L_BG_W_1)
        self.warning_unregister_message.hide()

        self.warning_danger_message = QtWidgets.QLabel(self)
        self.warning_danger_message.setAlignment(QtCore.Qt.AlignCenter)
        self.warning_danger_message.setText("Person gefährlich !")
        self.warning_danger_message.setGeometry(int(600 * X_SCALE), int(0 * Y_SCALE), int(200 * X_SCALE), int(40 * Y_SCALE))
        self.warning_danger_message.setStyleSheet(L_BG_W_1)
        self.warning_danger_message.hide()

        self.warning_blinking_flag = True

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.blinking_warning_message)
        self.timer.start(300)

        self.right_label = QtWidgets.QLabel(self)
        self.right_label.setGeometry(int(690 * X_SCALE), int(40 * Y_SCALE), int(110 * X_SCALE), int(560 * Y_SCALE))
        self.right_label.setStyleSheet(L_BG_R)

        #################### Monitor Page ###############
        # ----------  face item -------
        self.mo_real_face_1 = QtWidgets.QLabel(self)
        self.mo_real_face_1.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.mo_real_face_1.setPixmap(pix)
        self.mo_real_face_1.setGeometry(int(5 * X_SCALE), int(40 * Y_SCALE), int(50 * X_SCALE), int(70 * Y_SCALE))
        self.mo_real_face_1.setStyleSheet(L_BG_F)

        self.mo_saved_face_1 = QtWidgets.QLabel(self)
        self.mo_saved_face_1.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.mo_saved_face_1.setPixmap(pix)
        self.mo_saved_face_1.setGeometry(int(55 * X_SCALE), int(40 * Y_SCALE), int(50 * X_SCALE),int(70 * Y_SCALE))
        self.mo_saved_face_1.setStyleSheet(L_BG_F)

        self.mo_detail_1 = QtWidgets.QLabel(self)
        self.mo_detail_1.setText("Ismail Tuna\n\nAge : 43\nGender : Male\nAccessed : 0\n2024-02-22 12:22:34")
        self.mo_detail_1.setGeometry(int(105 * X_SCALE), int(40 * Y_SCALE), int(80 * X_SCALE), int(70 * Y_SCALE))
        self.mo_detail_1.setStyleSheet(L_BG_F)

        self.mo_icon_1 = QtWidgets.QLabel(self)
        self.mo_icon_1.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/danger.png")
        self.mo_icon_1.setPixmap(pix)
        self.mo_icon_1.setGeometry(int(170 * X_SCALE), int(40 * Y_SCALE), int(15 * X_SCALE), int(20 * Y_SCALE))
        self.mo_icon_1.setStyleSheet(L_BG_F)

        self.mo_real_face_2 = QtWidgets.QLabel(self)
        self.mo_real_face_2.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.mo_real_face_2.setPixmap(pix)
        self.mo_real_face_2.setGeometry(int(5 * X_SCALE), int(115 * Y_SCALE), int(50 * X_SCALE), int(70 * Y_SCALE))
        self.mo_real_face_2.setStyleSheet(L_BG_F)

        self.mo_saved_face_2 = QtWidgets.QLabel(self)
        self.mo_saved_face_2.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.mo_saved_face_2.setPixmap(pix)
        self.mo_saved_face_2.setGeometry(int(55 * X_SCALE), int(115 * Y_SCALE), int(50 * X_SCALE), int(70 * Y_SCALE))
        self.mo_saved_face_2.setStyleSheet(L_BG_F)

        self.mo_detail_2 = QtWidgets.QLabel(self)
        self.mo_detail_2.setText("Ismail Tuna\n\nAge : 43\nGender : Male\nAccessed : 0\n2024-02-22 12:22:34")
        self.mo_detail_2.setGeometry(int(105 * X_SCALE), int(115 * Y_SCALE), int(80 * X_SCALE), int(70 * Y_SCALE))
        self.mo_detail_2.setStyleSheet(L_BG_F)

        self.mo_icon_2 = QtWidgets.QLabel(self)
        self.mo_icon_2.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/danger.png")
        self.mo_icon_2.setPixmap(pix)
        self.mo_icon_2.setGeometry(int(170 * X_SCALE), int(115 * Y_SCALE), int(15 * X_SCALE), int(20 * Y_SCALE))
        self.mo_icon_2.setStyleSheet(L_BG_F)

        self.mo_real_face_3 = QtWidgets.QLabel(self)
        self.mo_real_face_3.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.mo_real_face_3.setPixmap(pix)
        self.mo_real_face_3.setGeometry(int(5 * X_SCALE), int(190 * Y_SCALE), int(50 * X_SCALE), int(70 * Y_SCALE))
        self.mo_real_face_3.setStyleSheet(L_BG_F)

        self.mo_saved_face_3 = QtWidgets.QLabel(self)
        self.mo_saved_face_3.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.mo_saved_face_3.setPixmap(pix)
        self.mo_saved_face_3.setGeometry(int(55 * X_SCALE), int(190 * Y_SCALE), int(50 * X_SCALE), int(70 * Y_SCALE))
        self.mo_saved_face_3.setStyleSheet(L_BG_F)

        self.mo_detail_3 = QtWidgets.QLabel(self)
        self.mo_detail_3.setText("Ismail Tuna\n\nAge : 43\nGender : Male\nAccessed : 0\n2024-02-22 12:22:34")
        self.mo_detail_3.setGeometry(int(105 * X_SCALE), int(190 * Y_SCALE), int(80 * X_SCALE), int(70 * Y_SCALE))
        self.mo_detail_3.setStyleSheet(L_BG_F)

        self.mo_icon_3 = QtWidgets.QLabel(self)
        self.mo_icon_3.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/danger.png")
        self.mo_icon_3.setPixmap(pix)
        self.mo_icon_3.setGeometry(int(170 * X_SCALE), int(190 * Y_SCALE), int(15 * X_SCALE), int(20 * Y_SCALE))
        self.mo_icon_3.setStyleSheet(L_BG_F)

        self.mo_real_face_4 = QtWidgets.QLabel(self)
        self.mo_real_face_4.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.mo_real_face_4.setPixmap(pix)
        self.mo_real_face_4.setGeometry(int(5 * X_SCALE), int(265 * Y_SCALE), int(50 * X_SCALE), int(70 * Y_SCALE))
        self.mo_real_face_4.setStyleSheet(L_BG_F)

        self.mo_saved_face_4 = QtWidgets.QLabel(self)
        self.mo_saved_face_4.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.mo_saved_face_4.setPixmap(pix)
        self.mo_saved_face_4.setGeometry(int(55 * X_SCALE), int(265 * Y_SCALE), int(50 * X_SCALE), int(70 * Y_SCALE))
        self.mo_saved_face_4.setStyleSheet(L_BG_F)

        self.mo_detail_4 = QtWidgets.QLabel(self)
        self.mo_detail_4.setText("Ismail Tuna\n\nAge : 43\nGender : Male\nAccessed : 0\n2024-02-22 12:22:34")
        self.mo_detail_4.setGeometry(int(105 * X_SCALE), int(265 * Y_SCALE), int(80 * X_SCALE), int(70 * Y_SCALE))
        self.mo_detail_4.setStyleSheet(L_BG_F)

        self.mo_icon_4 = QtWidgets.QLabel(self)
        self.mo_icon_4.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/danger.png")
        self.mo_icon_4.setPixmap(pix)
        self.mo_icon_4.setGeometry(int(170 * X_SCALE), int(265 * Y_SCALE), int(15 * X_SCALE), int(20 * Y_SCALE))
        self.mo_icon_4.setStyleSheet(L_BG_F)

        self.mo_real_face_5 = QtWidgets.QLabel(self)
        self.mo_real_face_5.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.mo_real_face_5.setPixmap(pix)
        self.mo_real_face_5.setGeometry(int(5 * X_SCALE), int(340 * Y_SCALE), int(50 * X_SCALE), int(70 * Y_SCALE))
        self.mo_real_face_5.setStyleSheet(L_BG_F)

        self.mo_saved_face_5 = QtWidgets.QLabel(self)
        self.mo_saved_face_5.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.mo_saved_face_5.setPixmap(pix)
        self.mo_saved_face_5.setGeometry(int(55 * X_SCALE), int(340 * Y_SCALE), int(50 * X_SCALE), int(70 * Y_SCALE))
        self.mo_saved_face_5.setStyleSheet(L_BG_F)

        self.mo_detail_5 = QtWidgets.QLabel(self)
        self.mo_detail_5.setText("Ismail Tuna\n\nAge : 43\nGender : Male\nAccessed : 0\n2024-02-22 12:22:34")
        self.mo_detail_5.setGeometry(int(105 * X_SCALE), int(340 * Y_SCALE), int(80 * X_SCALE), int(70 * Y_SCALE))
        self.mo_detail_5.setStyleSheet(L_BG_F)

        self.mo_icon_5 = QtWidgets.QLabel(self)
        self.mo_icon_5.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/danger.png")
        self.mo_icon_5.setPixmap(pix)
        self.mo_icon_5.setGeometry(int(170 * X_SCALE), int(340 * Y_SCALE), int(15 * X_SCALE), int(20 * Y_SCALE))
        self.mo_icon_5.setStyleSheet(L_BG_F)

        self.mo_real_face_6 = QtWidgets.QLabel(self)
        self.mo_real_face_6.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.mo_real_face_6.setPixmap(pix)
        self.mo_real_face_6.setGeometry(int(5 * X_SCALE), int(415 * Y_SCALE), int(50 * X_SCALE), int(70 * Y_SCALE))
        self.mo_real_face_6.setStyleSheet(L_BG_F)

        self.mo_saved_face_6 = QtWidgets.QLabel(self)
        self.mo_saved_face_6.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.mo_saved_face_6.setPixmap(pix)
        self.mo_saved_face_6.setGeometry(int(55 * X_SCALE), int(415 * Y_SCALE), int(50 * X_SCALE), int(70 * Y_SCALE))
        self.mo_saved_face_6.setStyleSheet(L_BG_F)

        self.mo_detail_6 = QtWidgets.QLabel(self)
        self.mo_detail_6.setText("Ismail Tuna\n\nAge : 43\nGender : Male\nAccessed : 0\n2024-02-22 12:22:34")
        self.mo_detail_6.setGeometry(int(105 * X_SCALE), int(415 * Y_SCALE), int(80 * X_SCALE), int(70 * Y_SCALE))
        self.mo_detail_6.setStyleSheet(L_BG_F)

        self.mo_icon_6 = QtWidgets.QLabel(self)
        self.mo_icon_6.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/danger.png")
        self.mo_icon_6.setPixmap(pix)
        self.mo_icon_6.setGeometry(int(170 * X_SCALE), int(415 * Y_SCALE), int(15 * X_SCALE), int(20 * Y_SCALE))
        self.mo_icon_6.setStyleSheet(L_BG_F)

        self.mo_real_face_7 = QtWidgets.QLabel(self)
        self.mo_real_face_7.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.mo_real_face_7.setPixmap(pix)
        self.mo_real_face_7.setGeometry(int(5 * X_SCALE), int(490 * Y_SCALE), int(50 * X_SCALE), int(70 * Y_SCALE))
        self.mo_real_face_7.setStyleSheet(L_BG_F)

        self.mo_saved_face_7 = QtWidgets.QLabel(self)
        self.mo_saved_face_7.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.mo_saved_face_7.setPixmap(pix)
        self.mo_saved_face_7.setGeometry(int(55 * X_SCALE), int(490 * Y_SCALE), int(50 * X_SCALE), int(70 * Y_SCALE))
        self.mo_saved_face_7.setStyleSheet(L_BG_F)

        self.mo_detail_7 = QtWidgets.QLabel(self)
        self.mo_detail_7.setText("Ismail Tuna\n\nAge : 43\nGender : Male\nAccessed : 0\n2024-02-22 12:22:34")
        self.mo_detail_7.setGeometry(int(105 * X_SCALE), int(490 * Y_SCALE), int(80 * X_SCALE), int(70 * Y_SCALE))
        self.mo_detail_7.setStyleSheet(L_BG_F)

        self.mo_icon_7 = QtWidgets.QLabel(self)
        self.mo_icon_7.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/danger.png")
        self.mo_icon_7.setPixmap(pix)
        self.mo_icon_7.setGeometry(int(170 * X_SCALE), int(490 * Y_SCALE), int(15 * X_SCALE), int(20 * Y_SCALE))
        self.mo_icon_7.setStyleSheet(L_BG_F)

        self.mo_prev_e_btn = QtWidgets.QLabel(self)
        self.mo_prev_e_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/prev_e.png")
        self.mo_prev_e_btn.setPixmap(pix)
        self.mo_prev_e_btn.setGeometry(int(20 * X_SCALE), int(575 * Y_SCALE), int(10 * X_SCALE), int(15 * Y_SCALE))
        self.mo_prev_e_btn.setStyleSheet(L_BG)

        self.mo_prev_btn = QtWidgets.QLabel(self)
        self.mo_prev_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/prev.png")
        self.mo_prev_btn.setPixmap(pix)
        self.mo_prev_btn.setGeometry(int(35 * X_SCALE), int(575 * Y_SCALE), int(15 * X_SCALE), int(15 * Y_SCALE))
        self.mo_prev_btn.setStyleSheet(L_BG)

        self.mo_next_e_btn = QtWidgets.QLabel(self)
        self.mo_next_e_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/next_e.png")
        self.mo_next_e_btn.setPixmap(pix)
        self.mo_next_e_btn.setGeometry(int(115 * X_SCALE), int(575 * Y_SCALE), int(10 * X_SCALE), int(15 * Y_SCALE))
        self.mo_next_e_btn.setStyleSheet(L_BG)

        self.mo_next_btn = QtWidgets.QLabel(self)
        self.mo_next_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/next.png")
        self.mo_next_btn.setPixmap(pix)
        self.mo_next_btn.setGeometry(int(95 * X_SCALE), int(575 * Y_SCALE), int(15 * X_SCALE), int(15 * Y_SCALE))
        self.mo_next_btn.setStyleSheet(L_BG)

        self.mo_page_num = QtWidgets.QLabel(self)
        self.mo_page_num.setText("1/1")
        self.mo_page_num.setAlignment(QtCore.Qt.AlignCenter)
        self.mo_page_num.setGeometry(int(55 * X_SCALE), int(575 * Y_SCALE), int(35 * X_SCALE), int(15 * Y_SCALE))
        self.mo_page_num.setStyleSheet(L_BG_F)

        self.mo_clearall_btn = QtWidgets.QPushButton(self)
        self.mo_clearall_btn.setGeometry(int(135 * X_SCALE), int(570 * Y_SCALE), int(50 * X_SCALE), int(25 * Y_SCALE))
        self.mo_clearall_btn.setText("Clear All")
        self.mo_clearall_btn.setStyleSheet(L_BT_N)

        # ------------- cameras -----------

        self.mo_camera1_view = QtWidgets.QLabel(self)
        self.mo_camera1_view.setGeometry(int(190 * X_SCALE), int(40 * Y_SCALE), int(250 * X_SCALE), int(280 * Y_SCALE))
        self.mo_camera1_view.setStyleSheet(L_CAM)

        self.mo_camera1_icon = QtWidgets.QLabel(self)
        self.mo_camera1_icon.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/camera.png")
        self.mo_camera1_icon.setPixmap(pix)
        self.mo_camera1_icon.setGeometry(int(295 * X_SCALE), int(150 * Y_SCALE), int(40 * X_SCALE), int(60 * Y_SCALE))
        self.mo_camera1_icon.setStyleSheet(L_CAM_ICON)

        self.mo_camera2_view = QtWidgets.QLabel(self)
        self.mo_camera2_view.setGeometry(int(440 * X_SCALE), int(40 * Y_SCALE), int(250 * X_SCALE), int(280 * Y_SCALE))
        self.mo_camera2_view.setStyleSheet(L_CAM)

        self.mo_camera2_icon = QtWidgets.QLabel(self)
        self.mo_camera2_icon.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/camera.png")
        self.mo_camera2_icon.setPixmap(pix)
        self.mo_camera2_icon.setGeometry(int(545 * X_SCALE), int(150 * Y_SCALE), int(40 * X_SCALE), int(60 * Y_SCALE))
        self.mo_camera2_icon.setStyleSheet(L_CAM_ICON)

        self.mo_camera3_view = QtWidgets.QLabel(self)
        self.mo_camera3_view.setGeometry(int(190 * X_SCALE), int(320 * Y_SCALE), int(250 * X_SCALE), int(280 * Y_SCALE))
        self.mo_camera3_view.setStyleSheet(L_CAM)

        self.mo_camera3_icon = QtWidgets.QLabel(self)
        self.mo_camera3_icon.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/camera.png")
        self.mo_camera3_icon.setPixmap(pix)
        self.mo_camera3_icon.setGeometry(int(295 * X_SCALE), int(430 * Y_SCALE), int(40 * X_SCALE), int(60 * Y_SCALE))
        self.mo_camera3_icon.setStyleSheet(L_CAM_ICON)

        self.mo_camera4_view = QtWidgets.QLabel(self)
        self.mo_camera4_view.setGeometry(int(440 * X_SCALE), int(320 * Y_SCALE), int(250 * X_SCALE), int(280 * Y_SCALE))
        self.mo_camera4_view.setStyleSheet(L_CAM)

        self.mo_camera4_icon = QtWidgets.QLabel(self)
        self.mo_camera4_icon.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/camera.png")
        self.mo_camera4_icon.setPixmap(pix)
        self.mo_camera4_icon.setGeometry(int(545 * X_SCALE), int(430 * Y_SCALE), int(40 * X_SCALE), int(60 * Y_SCALE))
        self.mo_camera4_icon.setStyleSheet(L_CAM_ICON)

        ####################### Person Manage Page ################
        self.pm_person_1 = QtWidgets.QLabel(self)
        self.pm_person_1.setGeometry(int(10 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.pm_person_1.setStyleSheet(L_BG_PERSON)

        self.pm_person_1_detail_text = QtWidgets.QLabel(self)
        self.pm_person_1_detail_text.setGeometry(int(20 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE), int(125 * Y_SCALE))
        self.pm_person_1_detail_text.setText("Name\nGeburtstag\nAlter\nStatus")
        self.pm_person_1_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.pm_person_1_detail_value = QtWidgets.QLabel(self)
        self.pm_person_1_detail_value.setGeometry(int(70 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE), int(125 * Y_SCALE))
        self.pm_person_1_detail_value.setText("Ismail\nTuna\nNone\n10.01.1970\nMale\nOK")
        self.pm_person_1_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.pm_person_1_face = QtWidgets.QLabel(self)
        self.pm_person_1_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.pm_person_1_face.setPixmap(pix)
        self.pm_person_1_face.setGeometry(int(45 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.pm_person_1_face.setStyleSheet(L_BG_F)

        self.pm_person_1_edit_btn = QtWidgets.QPushButton(self)
        self.pm_person_1_edit_btn.setGeometry(int(15 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE), int(25 * Y_SCALE))
        self.pm_person_1_edit_btn.setText("Bearbeiten")
        self.pm_person_1_edit_btn.setStyleSheet(L_BT_N)

        self.pm_person_1_delete_btn = QtWidgets.QPushButton(self)
        self.pm_person_1_delete_btn.setGeometry(int(75 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE), int(25 * Y_SCALE))
        self.pm_person_1_delete_btn.setText("Löschen")
        self.pm_person_1_delete_btn.setStyleSheet(L_BT_N)

        self.pm_person_2 = QtWidgets.QLabel(self)
        self.pm_person_2.setGeometry(int(145 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.pm_person_2.setStyleSheet(L_BG_PERSON)

        self.pm_person_2_detail_text = QtWidgets.QLabel(self)
        self.pm_person_2_detail_text.setGeometry(int(155 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                              int(125 * Y_SCALE))
        self.pm_person_2_detail_text.setText("Name\nGeburtstag\nAlter\nStatus")
        self.pm_person_2_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.pm_person_2_detail_value = QtWidgets.QLabel(self)
        self.pm_person_2_detail_value.setGeometry(int(205 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                               int(125 * Y_SCALE))
        self.pm_person_2_detail_value.setText("Ismail\nTuna\nNone\n10.01.1970\nMale\nOK")
        self.pm_person_2_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.pm_person_2_face = QtWidgets.QLabel(self)
        self.pm_person_2_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.pm_person_2_face.setPixmap(pix)
        self.pm_person_2_face.setGeometry(int(180 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.pm_person_2_face.setStyleSheet(L_BG_F)

        self.pm_person_2_edit_btn = QtWidgets.QPushButton(self)
        self.pm_person_2_edit_btn.setGeometry(int(150 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE), int(25 * Y_SCALE))
        self.pm_person_2_edit_btn.setText("Bearbeiten")
        self.pm_person_2_edit_btn.setStyleSheet(L_BT_N)

        self.pm_person_2_delete_btn = QtWidgets.QPushButton(self)
        self.pm_person_2_delete_btn.setGeometry(int(210 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE),
                                             int(25 * Y_SCALE))
        self.pm_person_2_delete_btn.setText("Löschen")
        self.pm_person_2_delete_btn.setStyleSheet(L_BT_N)

        self.pm_person_3 = QtWidgets.QLabel(self)
        self.pm_person_3.setGeometry(int(280 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.pm_person_3.setStyleSheet(L_BG_PERSON)

        self.pm_person_3_detail_text = QtWidgets.QLabel(self)
        self.pm_person_3_detail_text.setGeometry(int(290 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                              int(125 * Y_SCALE))
        self.pm_person_3_detail_text.setText("Name\nGeburtstag\nAlter\nStatus")
        self.pm_person_3_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.pm_person_3_detail_value = QtWidgets.QLabel(self)
        self.pm_person_3_detail_value.setGeometry(int(340 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                               int(125 * Y_SCALE))
        self.pm_person_3_detail_value.setText("Ismail\nTuna\nNone\n10.01.1970\nMale\nOK")
        self.pm_person_3_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.pm_person_3_face = QtWidgets.QLabel(self)
        self.pm_person_3_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.pm_person_3_face.setPixmap(pix)
        self.pm_person_3_face.setGeometry(int(315 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.pm_person_3_face.setStyleSheet(L_BG_F)

        self.pm_person_3_edit_btn = QtWidgets.QPushButton(self)
        self.pm_person_3_edit_btn.setGeometry(int(285 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE), int(25 * Y_SCALE))
        self.pm_person_3_edit_btn.setText("Bearbeiten")
        self.pm_person_3_edit_btn.setStyleSheet(L_BT_N)

        self.pm_person_3_delete_btn = QtWidgets.QPushButton(self)
        self.pm_person_3_delete_btn.setGeometry(int(345 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE),
                                             int(25 * Y_SCALE))
        self.pm_person_3_delete_btn.setText("Löschen")
        self.pm_person_3_delete_btn.setStyleSheet(L_BT_N)

        self.pm_person_4 = QtWidgets.QLabel(self)
        self.pm_person_4.setGeometry(int(415 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.pm_person_4.setStyleSheet(L_BG_PERSON)

        self.pm_person_4_detail_text = QtWidgets.QLabel(self)
        self.pm_person_4_detail_text.setGeometry(int(425 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                              int(125 * Y_SCALE))
        self.pm_person_4_detail_text.setText("Name\nGeburtstag\nAlter\nStatus")
        self.pm_person_4_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.pm_person_4_detail_value = QtWidgets.QLabel(self)
        self.pm_person_4_detail_value.setGeometry(int(475 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                               int(125 * Y_SCALE))
        self.pm_person_4_detail_value.setText("Ismail\nTuna\nNone\n10.01.1970\nMale\nOK")
        self.pm_person_4_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.pm_person_4_face = QtWidgets.QLabel(self)
        self.pm_person_4_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.pm_person_4_face.setPixmap(pix)
        self.pm_person_4_face.setGeometry(int(450 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.pm_person_4_face.setStyleSheet(L_BG_F)

        self.pm_person_4_edit_btn = QtWidgets.QPushButton(self)
        self.pm_person_4_edit_btn.setGeometry(int(420 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE), int(25 * Y_SCALE))
        self.pm_person_4_edit_btn.setText("Bearbeiten")
        self.pm_person_4_edit_btn.setStyleSheet(L_BT_N)

        self.pm_person_4_delete_btn = QtWidgets.QPushButton(self)
        self.pm_person_4_delete_btn.setGeometry(int(480 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE),
                                             int(25 * Y_SCALE))
        self.pm_person_4_delete_btn.setText("Löschen")
        self.pm_person_4_delete_btn.setStyleSheet(L_BT_N)

        self.pm_person_5 = QtWidgets.QLabel(self)
        self.pm_person_5.setGeometry(int(550 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.pm_person_5.setStyleSheet(L_BG_PERSON)

        self.pm_person_5_detail_text = QtWidgets.QLabel(self)
        self.pm_person_5_detail_text.setGeometry(int(560 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                              int(125 * Y_SCALE))
        self.pm_person_5_detail_text.setText("Name\nGeburtstag\nAlter\nStatus")
        self.pm_person_5_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.pm_person_5_detail_value = QtWidgets.QLabel(self)
        self.pm_person_5_detail_value.setGeometry(int(610 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                               int(125 * Y_SCALE))
        self.pm_person_5_detail_value.setText("Ismail\nTuna\nNone\n10.01.1970\nMale\nOK")
        self.pm_person_5_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.pm_person_5_face = QtWidgets.QLabel(self)
        self.pm_person_5_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.pm_person_5_face.setPixmap(pix)
        self.pm_person_5_face.setGeometry(int(585 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.pm_person_5_face.setStyleSheet(L_BG_F)

        self.pm_person_5_edit_btn = QtWidgets.QPushButton(self)
        self.pm_person_5_edit_btn.setGeometry(int(555 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE), int(25 * Y_SCALE))
        self.pm_person_5_edit_btn.setText("Bearbeiten")
        self.pm_person_5_edit_btn.setStyleSheet(L_BT_N)

        self.pm_person_5_delete_btn = QtWidgets.QPushButton(self)
        self.pm_person_5_delete_btn.setGeometry(int(615 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE),
                                             int(25 * Y_SCALE))
        self.pm_person_5_delete_btn.setText("Löschen")
        self.pm_person_5_delete_btn.setStyleSheet(L_BT_N)

        self.pm_person_6 = QtWidgets.QLabel(self)
        self.pm_person_6.setGeometry(int(10 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.pm_person_6.setStyleSheet(L_BG_PERSON)

        self.pm_person_6_detail_text = QtWidgets.QLabel(self)
        self.pm_person_6_detail_text.setGeometry(int(20 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                              int(125 * Y_SCALE))
        self.pm_person_6_detail_text.setText("Name\nGeburtstag\nAlter\nStatus")
        self.pm_person_6_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.pm_person_6_detail_value = QtWidgets.QLabel(self)
        self.pm_person_6_detail_value.setGeometry(int(70 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                               int(125 * Y_SCALE))
        self.pm_person_6_detail_value.setText("Ismail\nTuna\nNone\n10.01.1970\nMale\nOK")
        self.pm_person_6_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.pm_person_6_face = QtWidgets.QLabel(self)
        self.pm_person_6_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.pm_person_6_face.setPixmap(pix)
        self.pm_person_6_face.setGeometry(int(45 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.pm_person_6_face.setStyleSheet(L_BG_F)

        self.pm_person_6_edit_btn = QtWidgets.QPushButton(self)
        self.pm_person_6_edit_btn.setGeometry(int(15 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE), int(25 * Y_SCALE))
        self.pm_person_6_edit_btn.setText("Bearbeiten")
        self.pm_person_6_edit_btn.setStyleSheet(L_BT_N)

        self.pm_person_6_delete_btn = QtWidgets.QPushButton(self)
        self.pm_person_6_delete_btn.setGeometry(int(75 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE),
                                             int(25 * Y_SCALE))
        self.pm_person_6_delete_btn.setText("Löschen")
        self.pm_person_6_delete_btn.setStyleSheet(L_BT_N)

        self.pm_person_7 = QtWidgets.QLabel(self)
        self.pm_person_7.setGeometry(int(145 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.pm_person_7.setStyleSheet(L_BG_PERSON)

        self.pm_person_7_detail_text = QtWidgets.QLabel(self)
        self.pm_person_7_detail_text.setGeometry(int(155 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                              int(125 * Y_SCALE))
        self.pm_person_7_detail_text.setText("Name\nGeburtstag\nAlter\nStatus")
        self.pm_person_7_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.pm_person_7_detail_value = QtWidgets.QLabel(self)
        self.pm_person_7_detail_value.setGeometry(int(205 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                               int(125 * Y_SCALE))
        self.pm_person_7_detail_value.setText("Ismail\nTuna\nNone\n10.01.1970\nMale\nOK")
        self.pm_person_7_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.pm_person_7_face = QtWidgets.QLabel(self)
        self.pm_person_7_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.pm_person_7_face.setPixmap(pix)
        self.pm_person_7_face.setGeometry(int(180 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.pm_person_7_face.setStyleSheet(L_BG_F)

        self.pm_person_7_edit_btn = QtWidgets.QPushButton(self)
        self.pm_person_7_edit_btn.setGeometry(int(150 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE), int(25 * Y_SCALE))
        self.pm_person_7_edit_btn.setText("Bearbeiten")
        self.pm_person_7_edit_btn.setStyleSheet(L_BT_N)

        self.pm_person_7_delete_btn = QtWidgets.QPushButton(self)
        self.pm_person_7_delete_btn.setGeometry(int(210 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE),
                                             int(25 * Y_SCALE))
        self.pm_person_7_delete_btn.setText("Löschen")
        self.pm_person_7_delete_btn.setStyleSheet(L_BT_N)

        self.pm_person_8 = QtWidgets.QLabel(self)
        self.pm_person_8.setGeometry(int(280 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.pm_person_8.setStyleSheet(L_BG_PERSON)

        self.pm_person_8_detail_text = QtWidgets.QLabel(self)
        self.pm_person_8_detail_text.setGeometry(int(290 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                              int(125 * Y_SCALE))
        self.pm_person_8_detail_text.setText("Name\nGeburtstag\nAlter\nStatus")
        self.pm_person_8_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.pm_person_8_detail_value = QtWidgets.QLabel(self)
        self.pm_person_8_detail_value.setGeometry(int(340 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                               int(125 * Y_SCALE))
        self.pm_person_8_detail_value.setText("Ismail\nTuna\nNone\n10.01.1970\nMale\nOK")
        self.pm_person_8_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.pm_person_8_face = QtWidgets.QLabel(self)
        self.pm_person_8_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.pm_person_8_face.setPixmap(pix)
        self.pm_person_8_face.setGeometry(int(315 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.pm_person_8_face.setStyleSheet(L_BG_F)

        self.pm_person_8_edit_btn = QtWidgets.QPushButton(self)
        self.pm_person_8_edit_btn.setGeometry(int(285 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE), int(25 * Y_SCALE))
        self.pm_person_8_edit_btn.setText("Bearbeiten")
        self.pm_person_8_edit_btn.setStyleSheet(L_BT_N)

        self.pm_person_8_delete_btn = QtWidgets.QPushButton(self)
        self.pm_person_8_delete_btn.setGeometry(int(345 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE),
                                             int(25 * Y_SCALE))
        self.pm_person_8_delete_btn.setText("Löschen")
        self.pm_person_8_delete_btn.setStyleSheet(L_BT_N)

        self.pm_person_9 = QtWidgets.QLabel(self)
        self.pm_person_9.setGeometry(int(415 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.pm_person_9.setStyleSheet(L_BG_PERSON)

        self.pm_person_9_detail_text = QtWidgets.QLabel(self)
        self.pm_person_9_detail_text.setGeometry(int(425 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                              int(125 * Y_SCALE))
        self.pm_person_9_detail_text.setText("Name\nGeburtstag\nAlter\nStatus")
        self.pm_person_9_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.pm_person_9_detail_value = QtWidgets.QLabel(self)
        self.pm_person_9_detail_value.setGeometry(int(475 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                               int(125 * Y_SCALE))
        self.pm_person_9_detail_value.setText("Ismail\nTuna\nNone\n10.01.1970\nMale\nOK")
        self.pm_person_9_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.pm_person_9_face = QtWidgets.QLabel(self)
        self.pm_person_9_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.pm_person_9_face.setPixmap(pix)
        self.pm_person_9_face.setGeometry(int(450 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.pm_person_9_face.setStyleSheet(L_BG_F)

        self.pm_person_9_edit_btn = QtWidgets.QPushButton(self)
        self.pm_person_9_edit_btn.setGeometry(int(420 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE), int(25 * Y_SCALE))
        self.pm_person_9_edit_btn.setText("Bearbeiten")
        self.pm_person_9_edit_btn.setStyleSheet(L_BT_N)

        self.pm_person_9_delete_btn = QtWidgets.QPushButton(self)
        self.pm_person_9_delete_btn.setGeometry(int(480 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE),
                                             int(25 * Y_SCALE))
        self.pm_person_9_delete_btn.setText("Löschen")
        self.pm_person_9_delete_btn.setStyleSheet(L_BT_N)

        self.pm_person_10 = QtWidgets.QLabel(self)
        self.pm_person_10.setGeometry(int(550 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.pm_person_10.setStyleSheet(L_BG_PERSON)

        self.pm_person_10_detail_text = QtWidgets.QLabel(self)
        self.pm_person_10_detail_text.setGeometry(int(560 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                              int(125 * Y_SCALE))
        self.pm_person_10_detail_text.setText("Name\nGeburtstag\nAlter\nStatus")
        self.pm_person_10_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.pm_person_10_detail_value = QtWidgets.QLabel(self)
        self.pm_person_10_detail_value.setGeometry(int(610 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                               int(125 * Y_SCALE))
        self.pm_person_10_detail_value.setText("Ismail\nTuna\nNone\n10.01.1970\nMale\nOK")
        self.pm_person_10_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.pm_person_10_face = QtWidgets.QLabel(self)
        self.pm_person_10_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.pm_person_10_face.setPixmap(pix)
        self.pm_person_10_face.setGeometry(int(585 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.pm_person_10_face.setStyleSheet(L_BG_F)

        self.pm_person_10_edit_btn = QtWidgets.QPushButton(self)
        self.pm_person_10_edit_btn.setGeometry(int(555 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE), int(25 * Y_SCALE))
        self.pm_person_10_edit_btn.setText("Bearbeiten")
        self.pm_person_10_edit_btn.setStyleSheet(L_BT_N)

        self.pm_person_10_delete_btn = QtWidgets.QPushButton(self)
        self.pm_person_10_delete_btn.setGeometry(int(615 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE),
                                             int(25 * Y_SCALE))
        self.pm_person_10_delete_btn.setText("Löschen")
        self.pm_person_10_delete_btn.setStyleSheet(L_BT_N)

        self.pm_add_btn = QtWidgets.QPushButton(self)
        self.pm_add_btn.setGeometry(int(395 * X_SCALE), int(555 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.pm_add_btn.setText("Hinzufügen")
        self.pm_add_btn.setStyleSheet(L_BT_N)

        self.pm_import_btn = QtWidgets.QPushButton(self)
        self.pm_import_btn.setGeometry(int(495 * X_SCALE), int(555 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.pm_import_btn.setText("Import")
        self.pm_import_btn.setStyleSheet(L_BT_N)

        self.pm_export_btn = QtWidgets.QPushButton(self)
        self.pm_export_btn.setGeometry(int(595 * X_SCALE), int(555 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.pm_export_btn.setText("Export")
        self.pm_export_btn.setStyleSheet(L_BT_N)

        self.pm_prev_e_btn = QtWidgets.QLabel(self)
        self.pm_prev_e_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/prev_e.png")
        self.pm_prev_e_btn.setPixmap(pix)
        self.pm_prev_e_btn.setGeometry(int(225 * X_SCALE), int(560 * Y_SCALE), int(10 * X_SCALE), int(15 * Y_SCALE))
        self.pm_prev_e_btn.setStyleSheet(L_BG)

        self.pm_prev_btn = QtWidgets.QLabel(self)
        self.pm_prev_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/prev.png")
        self.pm_prev_btn.setPixmap(pix)
        self.pm_prev_btn.setGeometry(int(245 * X_SCALE), int(560 * Y_SCALE), int(15 * X_SCALE), int(15 * Y_SCALE))
        self.pm_prev_btn.setStyleSheet(L_BG)

        self.pm_next_e_btn = QtWidgets.QLabel(self)
        self.pm_next_e_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/next_e.png")
        self.pm_next_e_btn.setPixmap(pix)
        self.pm_next_e_btn.setGeometry(int(340 * X_SCALE), int(560 * Y_SCALE), int(10 * X_SCALE), int(15 * Y_SCALE))
        self.pm_next_e_btn.setStyleSheet(L_BG)

        self.pm_next_btn = QtWidgets.QLabel(self)
        self.pm_next_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/next.png")
        self.pm_next_btn.setPixmap(pix)
        self.pm_next_btn.setGeometry(int(315 * X_SCALE), int(560 * Y_SCALE), int(15 * X_SCALE), int(15 * Y_SCALE))
        self.pm_next_btn.setStyleSheet(L_BG)

        self.pm_page_num = QtWidgets.QLabel(self)
        self.pm_page_num.setText("1/1")
        self.pm_page_num.setAlignment(QtCore.Qt.AlignCenter)
        self.pm_page_num.setGeometry(int(270 * X_SCALE), int(560 * Y_SCALE), int(35 * X_SCALE), int(15 * Y_SCALE))
        self.pm_page_num.setStyleSheet(L_BG_F)

        ######################  Person History Page  #############
        self.ph_person_1 = QtWidgets.QLabel(self)
        self.ph_person_1.setGeometry(int(10 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.ph_person_1.setStyleSheet(L_BG_PERSON)

        self.ph_person_1_detail_text = QtWidgets.QLabel(self)
        self.ph_person_1_detail_text.setGeometry(int(20 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.ph_person_1_detail_text.setText("Name\nGeschlecht\nAlter\nZeit\nOrt\nView\nAction")
        self.ph_person_1_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.ph_person_1_detail_value = QtWidgets.QLabel(self)
        self.ph_person_1_detail_value.setGeometry(int(55 * X_SCALE), int(135 * Y_SCALE), int(75 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.ph_person_1_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.ph_person_1_face = QtWidgets.QLabel(self)
        self.ph_person_1_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.ph_person_1_face.setPixmap(pix)
        self.ph_person_1_face.setGeometry(int(45 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.ph_person_1_face.setStyleSheet(L_BG_F)

        self.ph_person_2 = QtWidgets.QLabel(self)
        self.ph_person_2.setGeometry(int(145 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.ph_person_2.setStyleSheet(L_BG_PERSON)

        self.ph_person_2_detail_text = QtWidgets.QLabel(self)
        self.ph_person_2_detail_text.setGeometry(int(155 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.ph_person_2_detail_text.setText("Name\nGeschlecht\nAlter\nZeit\nOrt\nView\nAction")
        self.ph_person_2_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.ph_person_2_detail_value = QtWidgets.QLabel(self)
        self.ph_person_2_detail_value.setGeometry(int(190 * X_SCALE), int(135 * Y_SCALE), int(75 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.ph_person_2_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.ph_person_2_face = QtWidgets.QLabel(self)
        self.ph_person_2_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.ph_person_2_face.setPixmap(pix)
        self.ph_person_2_face.setGeometry(int(180 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.ph_person_2_face.setStyleSheet(L_BG_F)

        self.ph_person_3 = QtWidgets.QLabel(self)
        self.ph_person_3.setGeometry(int(280 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.ph_person_3.setStyleSheet(L_BG_PERSON)

        self.ph_person_3_detail_text = QtWidgets.QLabel(self)
        self.ph_person_3_detail_text.setGeometry(int(290 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.ph_person_3_detail_text.setText("Name\nGeschlecht\nAlter\nZeit\nOrt\nView\nAction")
        self.ph_person_3_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.ph_person_3_detail_value = QtWidgets.QLabel(self)
        self.ph_person_3_detail_value.setGeometry(int(325 * X_SCALE), int(135 * Y_SCALE), int(75 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.ph_person_3_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.ph_person_3_face = QtWidgets.QLabel(self)
        self.ph_person_3_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.ph_person_3_face.setPixmap(pix)
        self.ph_person_3_face.setGeometry(int(315 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.ph_person_3_face.setStyleSheet(L_BG_F)

        self.ph_person_4 = QtWidgets.QLabel(self)
        self.ph_person_4.setGeometry(int(415 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.ph_person_4.setStyleSheet(L_BG_PERSON)

        self.ph_person_4_detail_text = QtWidgets.QLabel(self)
        self.ph_person_4_detail_text.setGeometry(int(425 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.ph_person_4_detail_text.setText("Name\nGeschlecht\nAlter\nZeit\nOrt\nView\nAction")
        self.ph_person_4_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.ph_person_4_detail_value = QtWidgets.QLabel(self)
        self.ph_person_4_detail_value.setGeometry(int(460 * X_SCALE), int(135 * Y_SCALE), int(75 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.ph_person_4_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.ph_person_4_face = QtWidgets.QLabel(self)
        self.ph_person_4_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.ph_person_4_face.setPixmap(pix)
        self.ph_person_4_face.setGeometry(int(450 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.ph_person_4_face.setStyleSheet(L_BG_F)

        self.ph_person_5 = QtWidgets.QLabel(self)
        self.ph_person_5.setGeometry(int(550 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.ph_person_5.setStyleSheet(L_BG_PERSON)

        self.ph_person_5_detail_text = QtWidgets.QLabel(self)
        self.ph_person_5_detail_text.setGeometry(int(560 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.ph_person_5_detail_text.setText("Name\nGeschlecht\nAlter\nZeit\nOrt\nView\nAction")
        self.ph_person_5_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.ph_person_5_detail_value = QtWidgets.QLabel(self)
        self.ph_person_5_detail_value.setGeometry(int(595 * X_SCALE), int(135 * Y_SCALE), int(75 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.ph_person_5_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.ph_person_5_face = QtWidgets.QLabel(self)
        self.ph_person_5_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.ph_person_5_face.setPixmap(pix)
        self.ph_person_5_face.setGeometry(int(585 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.ph_person_5_face.setStyleSheet(L_BG_F)

        self.ph_person_6 = QtWidgets.QLabel(self)
        self.ph_person_6.setGeometry(int(10 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.ph_person_6.setStyleSheet(L_BG_PERSON)

        self.ph_person_6_detail_text = QtWidgets.QLabel(self)
        self.ph_person_6_detail_text.setGeometry(int(20 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.ph_person_6_detail_text.setText("Name\nGeschlecht\nAlter\nZeit\nOrt\nView\nAction")
        self.ph_person_6_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.ph_person_6_detail_value = QtWidgets.QLabel(self)
        self.ph_person_6_detail_value.setGeometry(int(55 * X_SCALE), int(380 * Y_SCALE), int(75 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.ph_person_6_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.ph_person_6_face = QtWidgets.QLabel(self)
        self.ph_person_6_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.ph_person_6_face.setPixmap(pix)
        self.ph_person_6_face.setGeometry(int(45 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.ph_person_6_face.setStyleSheet(L_BG_F)

        self.ph_person_7 = QtWidgets.QLabel(self)
        self.ph_person_7.setGeometry(int(145 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.ph_person_7.setStyleSheet(L_BG_PERSON)

        self.ph_person_7_detail_text = QtWidgets.QLabel(self)
        self.ph_person_7_detail_text.setGeometry(int(155 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.ph_person_7_detail_text.setText("Name\nGeschlecht\nAlter\nZeit\nOrt\nView\nAction")
        self.ph_person_7_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.ph_person_7_detail_value = QtWidgets.QLabel(self)
        self.ph_person_7_detail_value.setGeometry(int(190 * X_SCALE), int(380 * Y_SCALE), int(75 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.ph_person_7_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.ph_person_7_face = QtWidgets.QLabel(self)
        self.ph_person_7_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.ph_person_7_face.setPixmap(pix)
        self.ph_person_7_face.setGeometry(int(180 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.ph_person_7_face.setStyleSheet(L_BG_F)

        self.ph_person_8 = QtWidgets.QLabel(self)
        self.ph_person_8.setGeometry(int(280 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.ph_person_8.setStyleSheet(L_BG_PERSON)

        self.ph_person_8_detail_text = QtWidgets.QLabel(self)
        self.ph_person_8_detail_text.setGeometry(int(290 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.ph_person_8_detail_text.setText("Name\nGeschlecht\nAlter\nZeit\nOrt\nView\nAction")
        self.ph_person_8_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.ph_person_8_detail_value = QtWidgets.QLabel(self)
        self.ph_person_8_detail_value.setGeometry(int(325 * X_SCALE), int(380 * Y_SCALE), int(75 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.ph_person_8_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.ph_person_8_face = QtWidgets.QLabel(self)
        self.ph_person_8_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.ph_person_8_face.setPixmap(pix)
        self.ph_person_8_face.setGeometry(int(315 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.ph_person_8_face.setStyleSheet(L_BG_F)

        self.ph_person_9 = QtWidgets.QLabel(self)
        self.ph_person_9.setGeometry(int(415 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.ph_person_9.setStyleSheet(L_BG_PERSON)

        self.ph_person_9_detail_text = QtWidgets.QLabel(self)
        self.ph_person_9_detail_text.setGeometry(int(425 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.ph_person_9_detail_text.setText("Name\nGeschlecht\nAlter\nZeit\nOrt\nView\nAction")
        self.ph_person_9_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.ph_person_9_detail_value = QtWidgets.QLabel(self)
        self.ph_person_9_detail_value.setGeometry(int(460 * X_SCALE), int(380 * Y_SCALE), int(75 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.ph_person_9_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.ph_person_9_face = QtWidgets.QLabel(self)
        self.ph_person_9_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.ph_person_9_face.setPixmap(pix)
        self.ph_person_9_face.setGeometry(int(450 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.ph_person_9_face.setStyleSheet(L_BG_F)

        self.ph_person_10 = QtWidgets.QLabel(self)
        self.ph_person_10.setGeometry(int(550 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.ph_person_10.setStyleSheet(L_BG_PERSON)

        self.ph_person_10_detail_text = QtWidgets.QLabel(self)
        self.ph_person_10_detail_text.setGeometry(int(560 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.ph_person_10_detail_text.setText("Name\nGeschlecht\nAlter\nZeit\nOrt\nView\nAction")
        self.ph_person_10_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.ph_person_10_detail_value = QtWidgets.QLabel(self)
        self.ph_person_10_detail_value.setGeometry(int(595 * X_SCALE), int(380 * Y_SCALE), int(75 * X_SCALE),
                                                   int(125 * Y_SCALE))
        self.ph_person_10_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.ph_person_10_face = QtWidgets.QLabel(self)
        self.ph_person_10_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face.jpg")
        self.ph_person_10_face.setPixmap(pix)
        self.ph_person_10_face.setGeometry(int(585 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE),
                                           int(100 * Y_SCALE))
        self.ph_person_10_face.setStyleSheet(L_BG_F)

        self.ph_prev_e_btn = QtWidgets.QLabel(self)
        self.ph_prev_e_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/prev_e.png")
        self.ph_prev_e_btn.setPixmap(pix)
        self.ph_prev_e_btn.setGeometry(int(225 * X_SCALE), int(560 * Y_SCALE), int(10 * X_SCALE), int(15 * Y_SCALE))
        self.ph_prev_e_btn.setStyleSheet(L_BG)

        self.ph_prev_btn = QtWidgets.QLabel(self)
        self.ph_prev_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/prev.png")
        self.ph_prev_btn.setPixmap(pix)
        self.ph_prev_btn.setGeometry(int(245 * X_SCALE), int(560 * Y_SCALE), int(15 * X_SCALE), int(15 * Y_SCALE))
        self.ph_prev_btn.setStyleSheet(L_BG)

        self.ph_next_e_btn = QtWidgets.QLabel(self)
        self.ph_next_e_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/next_e.png")
        self.ph_next_e_btn.setPixmap(pix)
        self.ph_next_e_btn.setGeometry(int(340 * X_SCALE), int(560 * Y_SCALE), int(10 * X_SCALE), int(15 * Y_SCALE))
        self.ph_next_e_btn.setStyleSheet(L_BG)

        self.ph_next_btn = QtWidgets.QLabel(self)
        self.ph_next_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/next.png")
        self.ph_next_btn.setPixmap(pix)
        self.ph_next_btn.setGeometry(int(315 * X_SCALE), int(560 * Y_SCALE), int(15 * X_SCALE), int(15 * Y_SCALE))
        self.ph_next_btn.setStyleSheet(L_BG)

        self.ph_page_num = QtWidgets.QLabel(self)
        self.ph_page_num.setText("1/1")
        self.ph_page_num.setAlignment(QtCore.Qt.AlignCenter)
        self.ph_page_num.setGeometry(int(270 * X_SCALE), int(560 * Y_SCALE), int(35 * X_SCALE), int(15 * Y_SCALE))
        self.ph_page_num.setStyleSheet(L_BG_F)

        self.ph_clearall_btn = QtWidgets.QPushButton(self)
        self.ph_clearall_btn.setGeometry(int(595 * X_SCALE), int(555 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.ph_clearall_btn.setText("Clear All")
        self.ph_clearall_btn.setStyleSheet(L_BT_N)

        ######################  Person Edit Page  ################
        self.pe_person_face = QtWidgets.QLabel(self)
        self.pe_person_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.pe_person_face.setPixmap(pix)
        self.pe_person_face.setGeometry(int(45 * X_SCALE), int(50 * Y_SCALE), int(130 * X_SCALE), int(200 * Y_SCALE))
        self.pe_person_face.setStyleSheet(L_BG_F)

        self.pe_from_camera_btn = QtWidgets.QPushButton(self)
        self.pe_from_camera_btn.setGeometry(int(200 * X_SCALE), int(50 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.pe_from_camera_btn.setText("Kamera")
        self.pe_from_camera_btn.setStyleSheet(L_BT_N)

        self.pe_from_image_btn = QtWidgets.QPushButton(self)
        self.pe_from_image_btn.setGeometry(int(200 * X_SCALE), int(90 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.pe_from_image_btn.setText("Bild")
        self.pe_from_image_btn.setStyleSheet(L_BT_N)

        self.pe_read_doc_btn = QtWidgets.QPushButton(self)
        self.pe_read_doc_btn.setGeometry(int(200 * X_SCALE), int(130 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.pe_read_doc_btn.setText("Ausweis")
        self.pe_read_doc_btn.setStyleSheet(L_BT_N)

        self.pe_sign_btn = QtWidgets.QPushButton(self)
        self.pe_sign_btn.setGeometry(int(200 * X_SCALE), int(170 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.pe_sign_btn.setText("Unterschrift")
        self.pe_sign_btn.setStyleSheet(L_BT_N)

        self.pe_edit_btn = QtWidgets.QPushButton(self)
        self.pe_edit_btn.setGeometry(int(200 * X_SCALE), int(210 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.pe_edit_btn.setText("Bearbeiten")
        self.pe_edit_btn.setStyleSheet(L_BT_N)

        self.pe_text1 = QtWidgets.QLabel(self)
        self.pe_text1.setText("Vorname Nachnname:")
        self.pe_text1.setGeometry(int(45 * X_SCALE), int(300 * Y_SCALE), int(150 * X_SCALE), int(20 * Y_SCALE))
        self.pe_text1.setStyleSheet(L_BG)

        self.pe_name_value = QtWidgets.QLineEdit(self)
        self.pe_name_value.setGeometry(int(45 * X_SCALE), int(330 * Y_SCALE), int(235 * X_SCALE), int(30 * Y_SCALE))
        self.pe_name_value.setStyleSheet(L_BG)

        self.pe_text2 = QtWidgets.QLabel(self)
        self.pe_text2.setText("Geburstdatum:")
        self.pe_text2.setGeometry(int(45 * X_SCALE), int(380 * Y_SCALE), int(90 * X_SCALE), int(20 * Y_SCALE))
        self.pe_text2.setStyleSheet(L_BG)

        self.pe_birthday_value = QtWidgets.QLineEdit(self)
        self.pe_birthday_value.setGeometry(int(45 * X_SCALE), int(410 * Y_SCALE), int(90 * X_SCALE), int(30 * Y_SCALE))
        self.pe_birthday_value.setStyleSheet(L_BG)

        self.pe_text3 = QtWidgets.QLabel(self)
        self.pe_text3.setText("Alter:")
        self.pe_text3.setGeometry(int(150 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE), int(20 * Y_SCALE))
        self.pe_text3.setStyleSheet(L_BG)

        self.pe_age_value = QtWidgets.QLineEdit(self)
        self.pe_age_value.setGeometry(int(150 * X_SCALE), int(410 * Y_SCALE), int(40 * X_SCALE), int(30 * Y_SCALE))
        self.pe_age_value.setStyleSheet(L_BG)

        self.pe_text4 = QtWidgets.QLabel(self)
        self.pe_text4.setText("Status:")
        self.pe_text4.setGeometry(int(205 * X_SCALE), int(380 * Y_SCALE), int(75 * X_SCALE), int(20 * Y_SCALE))
        self.pe_text4.setStyleSheet(L_BG)

        self.pe_status_value = QtWidgets.QLineEdit(self)
        self.pe_status_value.setDisabled(True)
        self.pe_status_value.setGeometry(int(205 * X_SCALE), int(410 * Y_SCALE), int(75 * X_SCALE), int(30 * Y_SCALE))
        self.pe_status_value.setStyleSheet(L_BG)

        self.pe_gov_text = QtWidgets.QLabel(self)
        self.pe_gov_text.setText("Oasis Abfrage:")
        self.pe_gov_text.setGeometry(int(45 * X_SCALE), int(460 * Y_SCALE), int(150 * X_SCALE), int(20 * Y_SCALE))
        self.pe_gov_text.setStyleSheet(L_BG)

        self.pe_gov_message_value = QtWidgets.QTextEdit(self)
        self.pe_gov_message_value.setDisabled(True)
        self.pe_gov_message_value.setGeometry(int(45 * X_SCALE), int(490 * Y_SCALE), int(235 * X_SCALE), int(60 * Y_SCALE))
        self.pe_gov_message_value.setStyleSheet(L_BG)

        # ------------ Tab  ----------
        self.pe_setting_btn = QtWidgets.QPushButton(self)
        self.pe_setting_btn.setText("Eigenschaften")
        self.pe_setting_btn.setGeometry(int(330 * X_SCALE), int(50 * Y_SCALE), int(100 * X_SCALE), int(30 * Y_SCALE))
        self.pe_setting_btn.setStyleSheet(L_BT_TAB1)

        self.pe_agreement_btn = QtWidgets.QPushButton(self)
        self.pe_agreement_btn.setText("Vereinbarung")
        self.pe_agreement_btn.setGeometry(int(430 * X_SCALE), int(50 * Y_SCALE), int(100 * X_SCALE), int(30 * Y_SCALE))
        self.pe_agreement_btn.setStyleSheet(L_BT_TAB2)

        self.pe_label = QtWidgets.QLabel(self)
        self.pe_label.setGeometry(int(330 * X_SCALE), int(75 * Y_SCALE), int(350 * X_SCALE), int(470 * Y_SCALE))
        self.pe_label.setStyleSheet(L_ST_TEXT)

        # ------------ Agreement ------------------------
        pdffile = "database/agreements/sample.pdf"
        doc = fitz.open(pdffile)
        page = doc.load_page(0)  # number of page
        pix = page.get_pixmap()
        output = "database/agreements/sample.jpg"
        pix.save(output)
        doc.close()

        self.pe_agree_text = QtWidgets.QLabel(self)
        self.pe_agree_text.setScaledContents(True)
        pix = QtGui.QPixmap('database/agreements/sample.jpg')
        self.pe_agree_text.setPixmap(pix)
        self.pe_agree_text.setGeometry(int(400 * X_SCALE), int(85 * Y_SCALE), int(210 * X_SCALE), int(400 * Y_SCALE))
        self.pe_agree_text.setStyleSheet(L_ST_TEXT)

        self.pe_agree_print_btn = QtWidgets.QPushButton(self)
        self.pe_agree_print_btn.setGeometry(int(580 * X_SCALE), int(495 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.pe_agree_print_btn.setText("Print")
        self.pe_agree_print_btn.setStyleSheet(L_BT_N)

        # ------------ characteristic settings ----------
        self.pe_text5 = QtWidgets.QLabel(self)
        self.pe_text5.setText("Geschlect:")
        self.pe_text5.setAlignment(QtCore.Qt.AlignRight)
        self.pe_text5.setGeometry(int(370 * X_SCALE), int(110 * Y_SCALE), int(75 * X_SCALE), int(20 * Y_SCALE))
        self.pe_text5.setStyleSheet(L_ST_TEXT)

        self.pe_gender = QtWidgets.QComboBox(self)
        self.pe_gender.addItem("Mann")
        self.pe_gender.addItem("Frau")
        self.pe_gender.setGeometry(int(450 * X_SCALE), int(110 * Y_SCALE), int(170 * X_SCALE), int(20 * Y_SCALE))
        self.pe_gender.setStyleSheet(L_ST_VAL)

        self.pe_text6 = QtWidgets.QLabel(self)
        self.pe_text6.setText("Gasttyp:")
        self.pe_text6.setAlignment(QtCore.Qt.AlignRight)
        self.pe_text6.setGeometry(int(370 * X_SCALE), int(140 * Y_SCALE), int(75 * X_SCALE), int(20 * Y_SCALE))
        self.pe_text6.setStyleSheet(L_ST_TEXT)

        self.pe_guest_type = QtWidgets.QComboBox(self)
        self.pe_guest_type.addItem("Nicht zugeordnet")
        self.pe_guest_type.addItem("Sehr guter Spieler")
        self.pe_guest_type.addItem("Guter Spieler")
        self.pe_guest_type.addItem("Kaffeetrinker")
        self.pe_guest_type.setGeometry(int(450 * X_SCALE), int(140 * Y_SCALE), int(170 * X_SCALE), int(20 * Y_SCALE))
        self.pe_guest_type.setStyleSheet(L_ST_VAL)

        self.pe_text7 = QtWidgets.QLabel(self)
        self.pe_text7.setText("Sicherheitstyp:")
        self.pe_text7.setAlignment(QtCore.Qt.AlignRight)
        self.pe_text7.setGeometry(int(370 * X_SCALE), int(170 * Y_SCALE), int(75 * X_SCALE), int(20 * Y_SCALE))
        self.pe_text7.setStyleSheet(L_ST_TEXT)

        self.pe_safety_type = QtWidgets.QComboBox(self)
        self.pe_safety_type.addItem("Nicht zugeordnet")
        self.pe_safety_type.addItem("Unauffällig")
        self.pe_safety_type.addItem("Manipulationsverdacht")
        self.pe_safety_type.addItem("Manipulator")
        self.pe_safety_type.setGeometry(int(450 * X_SCALE), int(170 * Y_SCALE), int(170 * X_SCALE), int(20 * Y_SCALE))
        self.pe_safety_type.setStyleSheet(L_ST_VAL)

        self.pe_text8 = QtWidgets.QLabel(self)
        self.pe_text8.setText("Information:")
        self.pe_text8.setAlignment(QtCore.Qt.AlignRight)
        self.pe_text8.setGeometry(int(370 * X_SCALE), int(200 * Y_SCALE), int(75 * X_SCALE), int(20 * Y_SCALE))
        self.pe_text8.setStyleSheet(L_ST_TEXT)

        self.pe_info_text = QtWidgets.QTextEdit(self)
        self.pe_info_text.setGeometry(int(450 * X_SCALE), int(200 * Y_SCALE), int(170 * X_SCALE), int(120 * Y_SCALE))
        self.pe_info_text.setStyleSheet(L_ST_VAL)

        self.pe_text9 = QtWidgets.QLabel(self)
        self.pe_text9.setText("Gesperrt:")
        self.pe_text9.setAlignment(QtCore.Qt.AlignRight)
        self.pe_text9.setGeometry(int(370 * X_SCALE), int(330 * Y_SCALE), int(75 * X_SCALE), int(20 * Y_SCALE))
        self.pe_text9.setStyleSheet(L_ST_TEXT)

        self.pe_blocked = QtWidgets.QComboBox(self)
        self.pe_blocked.addItem("Nein")
        self.pe_blocked.addItem("Ja")
        self.pe_blocked.setGeometry(int(450 * X_SCALE), int(330 * Y_SCALE), int(170 * X_SCALE), int(20 * Y_SCALE))
        self.pe_blocked.setStyleSheet(L_ST_VAL)

        self.pe_text10 = QtWidgets.QLabel(self)
        self.pe_text10.setText("Von:")
        self.pe_text10.setAlignment(QtCore.Qt.AlignRight)
        self.pe_text10.setGeometry(int(370 * X_SCALE), int(360 * Y_SCALE), int(110 * X_SCALE), int(20 * Y_SCALE))
        self.pe_text10.setStyleSheet(L_ST_TEXT)

        self.pe_when_from = QtWidgets.QLineEdit(self)
        self.pe_when_from.setGeometry(int(485 * X_SCALE), int(360 * Y_SCALE), int(110 * X_SCALE), int(20 * Y_SCALE))
        self.pe_when_from.setStyleSheet(L_BG)

        self.pe_text11 = QtWidgets.QLabel(self)
        self.pe_text11.setText("Bis:")
        self.pe_text11.setAlignment(QtCore.Qt.AlignRight)
        self.pe_text11.setGeometry(int(370 * X_SCALE), int(390 * Y_SCALE), int(110 * X_SCALE), int(20 * Y_SCALE))
        self.pe_text11.setStyleSheet(L_ST_TEXT)

        self.pe_when_to = QtWidgets.QLineEdit(self)
        self.pe_when_to.setGeometry(int(485 * X_SCALE), int(390 * Y_SCALE), int(110 * X_SCALE), int(20 * Y_SCALE))
        self.pe_when_to.setStyleSheet(L_BG)

        self.pe_text12 = QtWidgets.QLabel(self)
        self.pe_text12.setText("Ort:")
        self.pe_text12.setAlignment(QtCore.Qt.AlignRight)
        self.pe_text12.setGeometry(int(370 * X_SCALE), int(420 * Y_SCALE), int(75 * X_SCALE), int(20 * Y_SCALE))
        self.pe_text12.setStyleSheet(L_ST_TEXT)

        self.pe_where = QtWidgets.QLineEdit(self)
        self.pe_where.setGeometry(int(450 * X_SCALE), int(420 * Y_SCALE), int(170 * X_SCALE), int(20 * Y_SCALE))
        self.pe_where.setStyleSheet(L_BG)

        self.pe_text13 = QtWidgets.QLabel(self)
        self.pe_text13.setText("Grund:")
        self.pe_text13.setAlignment(QtCore.Qt.AlignRight)
        self.pe_text13.setGeometry(int(370 * X_SCALE), int(450 * Y_SCALE), int(75 * X_SCALE), int(20 * Y_SCALE))
        self.pe_text13.setStyleSheet(L_ST_TEXT)

        self.pe_reason = QtWidgets.QLineEdit(self)
        self.pe_reason.setGeometry(int(450 * X_SCALE), int(450 * Y_SCALE), int(170 * X_SCALE), int(20 * Y_SCALE))
        self.pe_reason.setStyleSheet(L_BG)

        self.pe_text14 = QtWidgets.QLabel(self)
        self.pe_text14.setText("Typ:")
        self.pe_text14.setAlignment(QtCore.Qt.AlignRight)
        self.pe_text14.setGeometry(int(370 * X_SCALE), int(480 * Y_SCALE), int(75 * X_SCALE), int(20 * Y_SCALE))
        self.pe_text14.setStyleSheet(L_ST_TEXT)

        self.pe_local_type = QtWidgets.QComboBox(self)
        self.pe_local_type.addItem("Lokal")
        self.pe_local_type.addItem("Global")
        self.pe_local_type.setGeometry(int(450 * X_SCALE), int(480 * Y_SCALE), int(170 * X_SCALE), int(20 * Y_SCALE))
        self.pe_local_type.setStyleSheet(L_ST_VAL)

        self.pe_oasis_btn = QtWidgets.QPushButton(self)
        self.pe_oasis_btn.setGeometry(int(45 * X_SCALE), int(555 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.pe_oasis_btn.setText("Oasis Abfrage")
        self.pe_oasis_btn.setStyleSheet(L_BT_N)

        self.pe_save_btn = QtWidgets.QPushButton(self)
        self.pe_save_btn.setGeometry(int(480 * X_SCALE), int(555 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.pe_save_btn.setText("Speichern")
        self.pe_save_btn.setStyleSheet(L_BT_N)

        self.pe_cancel_btn = QtWidgets.QPushButton(self)
        self.pe_cancel_btn.setGeometry(int(580 * X_SCALE), int(555 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.pe_cancel_btn.setText("Abbrechen")
        self.pe_cancel_btn.setStyleSheet(L_BT_N)

        # ######################  User Manage Page  ###############
        self.um_user_1 = QtWidgets.QLabel(self)
        self.um_user_1.setGeometry(int(10 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.um_user_1.setStyleSheet(L_BG_PERSON)

        self.um_user_1_detail_text = QtWidgets.QLabel(self)
        self.um_user_1_detail_text.setText("Name\nPrivilege\nCreator\nPhone\nBlocked")
        self.um_user_1_detail_text.setGeometry(int(20 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.um_user_1_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.um_user_1_detail_value = QtWidgets.QLabel(self)
        self.um_user_1_detail_value.setGeometry(int(70 * X_SCALE), int(135 * Y_SCALE), int(65 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.um_user_1_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.um_user_1_face = QtWidgets.QLabel(self)
        self.um_user_1_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.um_user_1_face.setPixmap(pix)
        self.um_user_1_face.setGeometry(int(45 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.um_user_1_face.setStyleSheet(L_BG_F)

        self.um_user_1_edit_btn = QtWidgets.QPushButton(self)
        self.um_user_1_edit_btn.setGeometry(int(15 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE),
                                              int(25 * Y_SCALE))
        self.um_user_1_edit_btn.setText("Bearbeiten")
        self.um_user_1_edit_btn.setStyleSheet(L_BT_N)

        self.um_user_1_delete_btn = QtWidgets.QPushButton(self)
        self.um_user_1_delete_btn.setGeometry(int(75 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE),
                                                int(25 * Y_SCALE))
        self.um_user_1_delete_btn.setText("Löschen")
        self.um_user_1_delete_btn.setStyleSheet(L_BT_N)

        self.um_user_2 = QtWidgets.QLabel(self)
        self.um_user_2.setGeometry(int(145 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.um_user_2.setStyleSheet(L_BG_PERSON)

        self.um_user_2_detail_text = QtWidgets.QLabel(self)
        self.um_user_2_detail_text.setText("Name\nPrivilege\nCreator\nPhone\nBlocked")

        self.um_user_2_detail_text.setGeometry(int(155 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.um_user_2_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.um_user_2_detail_value = QtWidgets.QLabel(self)
        self.um_user_2_detail_value.setGeometry(int(205 * X_SCALE), int(135 * Y_SCALE), int(65 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.um_user_2_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.um_user_2_face = QtWidgets.QLabel(self)
        self.um_user_2_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.um_user_2_face.setPixmap(pix)
        self.um_user_2_face.setGeometry(int(180 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.um_user_2_face.setStyleSheet(L_BG_F)

        self.um_user_2_edit_btn = QtWidgets.QPushButton(self)
        self.um_user_2_edit_btn.setGeometry(int(150 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE),
                                              int(25 * Y_SCALE))
        self.um_user_2_edit_btn.setText("Bearbeiten")
        self.um_user_2_edit_btn.setStyleSheet(L_BT_N)

        self.um_user_2_delete_btn = QtWidgets.QPushButton(self)
        self.um_user_2_delete_btn.setGeometry(int(210 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE),
                                                int(25 * Y_SCALE))
        self.um_user_2_delete_btn.setText("Löschen")
        self.um_user_2_delete_btn.setStyleSheet(L_BT_N)

        self.um_user_3 = QtWidgets.QLabel(self)
        self.um_user_3.setGeometry(int(280 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.um_user_3.setStyleSheet(L_BG_PERSON)

        self.um_user_3_detail_text = QtWidgets.QLabel(self)
        self.um_user_3_detail_text.setText("Name\nPrivilege\nCreator\nPhone\nBlocked")
        self.um_user_3_detail_text.setGeometry(int(290 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.um_user_3_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.um_user_3_detail_value = QtWidgets.QLabel(self)
        self.um_user_3_detail_value.setGeometry(int(340 * X_SCALE), int(135 * Y_SCALE), int(65 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.um_user_3_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.um_user_3_face = QtWidgets.QLabel(self)
        self.um_user_3_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.um_user_3_face.setPixmap(pix)
        self.um_user_3_face.setGeometry(int(315 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.um_user_3_face.setStyleSheet(L_BG_F)

        self.um_user_3_edit_btn = QtWidgets.QPushButton(self)
        self.um_user_3_edit_btn.setGeometry(int(285 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE),
                                              int(25 * Y_SCALE))
        self.um_user_3_edit_btn.setText("Bearbeiten")
        self.um_user_3_edit_btn.setStyleSheet(L_BT_N)

        self.um_user_3_delete_btn = QtWidgets.QPushButton(self)
        self.um_user_3_delete_btn.setGeometry(int(345 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE),
                                                int(25 * Y_SCALE))
        self.um_user_3_delete_btn.setText("Löschen")
        self.um_user_3_delete_btn.setStyleSheet(L_BT_N)

        self.um_user_4 = QtWidgets.QLabel(self)
        self.um_user_4.setGeometry(int(415 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.um_user_4.setStyleSheet(L_BG_PERSON)

        self.um_user_4_detail_text = QtWidgets.QLabel(self)
        self.um_user_4_detail_text.setText("Name\nPrivilege\nCreator\nPhone\nBlocked")
        self.um_user_4_detail_text.setGeometry(int(425 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.um_user_4_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.um_user_4_detail_value = QtWidgets.QLabel(self)
        self.um_user_4_detail_value.setGeometry(int(475 * X_SCALE), int(135 * Y_SCALE), int(65 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.um_user_4_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.um_user_4_face = QtWidgets.QLabel(self)
        self.um_user_4_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.um_user_4_face.setPixmap(pix)
        self.um_user_4_face.setGeometry(int(450 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.um_user_4_face.setStyleSheet(L_BG_F)

        self.um_user_4_edit_btn = QtWidgets.QPushButton(self)
        self.um_user_4_edit_btn.setGeometry(int(420 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE),
                                              int(25 * Y_SCALE))
        self.um_user_4_edit_btn.setText("Bearbeiten")
        self.um_user_4_edit_btn.setStyleSheet(L_BT_N)

        self.um_user_4_delete_btn = QtWidgets.QPushButton(self)
        self.um_user_4_delete_btn.setGeometry(int(480 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE),
                                                int(25 * Y_SCALE))
        self.um_user_4_delete_btn.setText("Löschen")
        self.um_user_4_delete_btn.setStyleSheet(L_BT_N)

        self.um_user_5 = QtWidgets.QLabel(self)
        self.um_user_5.setGeometry(int(550 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.um_user_5.setStyleSheet(L_BG_PERSON)

        self.um_user_5_detail_text = QtWidgets.QLabel(self)
        self.um_user_5_detail_text.setText("Name\nPrivilege\nCreator\nPhone\nBlocked")
        self.um_user_5_detail_text.setGeometry(int(560 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.um_user_5_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.um_user_5_detail_value = QtWidgets.QLabel(self)
        self.um_user_5_detail_value.setGeometry(int(610 * X_SCALE), int(135 * Y_SCALE), int(65 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.um_user_5_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.um_user_5_face = QtWidgets.QLabel(self)
        self.um_user_5_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.um_user_5_face.setPixmap(pix)
        self.um_user_5_face.setGeometry(int(585 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.um_user_5_face.setStyleSheet(L_BG_F)

        self.um_user_5_edit_btn = QtWidgets.QPushButton(self)
        self.um_user_5_edit_btn.setGeometry(int(555 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE),
                                              int(25 * Y_SCALE))
        self.um_user_5_edit_btn.setText("Bearbeiten")
        self.um_user_5_edit_btn.setStyleSheet(L_BT_N)

        self.um_user_5_delete_btn = QtWidgets.QPushButton(self)
        self.um_user_5_delete_btn.setGeometry(int(615 * X_SCALE), int(250 * Y_SCALE), int(60 * X_SCALE),
                                                int(25 * Y_SCALE))
        self.um_user_5_delete_btn.setText("Löschen")
        self.um_user_5_delete_btn.setStyleSheet(L_BT_N)

        self.um_user_6 = QtWidgets.QLabel(self)
        self.um_user_6.setGeometry(int(10 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.um_user_6.setStyleSheet(L_BG_PERSON)

        self.um_user_6_detail_text = QtWidgets.QLabel(self)
        self.um_user_6_detail_text.setText("Name\nPrivilege\nCreator\nPhone\nBlocked")
        self.um_user_6_detail_text.setGeometry(int(20 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.um_user_6_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.um_user_6_detail_value = QtWidgets.QLabel(self)
        self.um_user_6_detail_value.setGeometry(int(70 * X_SCALE), int(380 * Y_SCALE), int(65 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.um_user_6_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.um_user_6_face = QtWidgets.QLabel(self)
        self.um_user_6_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.um_user_6_face.setPixmap(pix)
        self.um_user_6_face.setGeometry(int(45 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.um_user_6_face.setStyleSheet(L_BG_F)

        self.um_user_6_edit_btn = QtWidgets.QPushButton(self)
        self.um_user_6_edit_btn.setGeometry(int(15 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE),
                                              int(25 * Y_SCALE))
        self.um_user_6_edit_btn.setText("Bearbeiten")
        self.um_user_6_edit_btn.setStyleSheet(L_BT_N)

        self.um_user_6_delete_btn = QtWidgets.QPushButton(self)
        self.um_user_6_delete_btn.setGeometry(int(75 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE),
                                                int(25 * Y_SCALE))
        self.um_user_6_delete_btn.setText("Löschen")
        self.um_user_6_delete_btn.setStyleSheet(L_BT_N)

        self.um_user_7 = QtWidgets.QLabel(self)
        self.um_user_7.setGeometry(int(145 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.um_user_7.setStyleSheet(L_BG_PERSON)

        self.um_user_7_detail_text = QtWidgets.QLabel(self)
        self.um_user_7_detail_text.setText("Name\nPrivilege\nCreator\nPhone\nBlocked")
        self.um_user_7_detail_text.setGeometry(int(155 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.um_user_7_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.um_user_7_detail_value = QtWidgets.QLabel(self)
        self.um_user_7_detail_value.setGeometry(int(205 * X_SCALE), int(380 * Y_SCALE), int(65 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.um_user_7_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.um_user_7_face = QtWidgets.QLabel(self)
        self.um_user_7_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.um_user_7_face.setPixmap(pix)
        self.um_user_7_face.setGeometry(int(180 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.um_user_7_face.setStyleSheet(L_BG_F)

        self.um_user_7_edit_btn = QtWidgets.QPushButton(self)
        self.um_user_7_edit_btn.setGeometry(int(150 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE),
                                              int(25 * Y_SCALE))
        self.um_user_7_edit_btn.setText("Bearbeiten")
        self.um_user_7_edit_btn.setStyleSheet(L_BT_N)

        self.um_user_7_delete_btn = QtWidgets.QPushButton(self)
        self.um_user_7_delete_btn.setGeometry(int(210 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE),
                                                int(25 * Y_SCALE))
        self.um_user_7_delete_btn.setText("Löschen")
        self.um_user_7_delete_btn.setStyleSheet(L_BT_N)

        self.um_user_8 = QtWidgets.QLabel(self)
        self.um_user_8.setGeometry(int(280 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.um_user_8.setStyleSheet(L_BG_PERSON)

        self.um_user_8_detail_text = QtWidgets.QLabel(self)
        self.um_user_8_detail_text.setText("Name\nPrivilege\nCreator\nPhone\nBlocked")
        self.um_user_8_detail_text.setGeometry(int(290 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.um_user_8_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.um_user_8_detail_value = QtWidgets.QLabel(self)
        self.um_user_8_detail_value.setGeometry(int(340 * X_SCALE), int(380 * Y_SCALE), int(65 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.um_user_8_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.um_user_8_face = QtWidgets.QLabel(self)
        self.um_user_8_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.um_user_8_face.setPixmap(pix)
        self.um_user_8_face.setGeometry(int(315 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.um_user_8_face.setStyleSheet(L_BG_F)

        self.um_user_8_edit_btn = QtWidgets.QPushButton(self)
        self.um_user_8_edit_btn.setGeometry(int(285 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE),
                                              int(25 * Y_SCALE))
        self.um_user_8_edit_btn.setText("Bearbeiten")
        self.um_user_8_edit_btn.setStyleSheet(L_BT_N)

        self.um_user_8_delete_btn = QtWidgets.QPushButton(self)
        self.um_user_8_delete_btn.setGeometry(int(345 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE),
                                                int(25 * Y_SCALE))
        self.um_user_8_delete_btn.setText("Löschen")
        self.um_user_8_delete_btn.setStyleSheet(L_BT_N)

        self.um_user_9 = QtWidgets.QLabel(self)
        self.um_user_9.setGeometry(int(415 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.um_user_9.setStyleSheet(L_BG_PERSON)

        self.um_user_9_detail_text = QtWidgets.QLabel(self)
        self.um_user_9_detail_text.setText("Name\nPrivilege\nCreator\nPhone\nBlocked")
        self.um_user_9_detail_text.setGeometry(int(425 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.um_user_9_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.um_user_9_detail_value = QtWidgets.QLabel(self)
        self.um_user_9_detail_value.setGeometry(int(475 * X_SCALE), int(380 * Y_SCALE), int(65 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.um_user_9_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.um_user_9_face = QtWidgets.QLabel(self)
        self.um_user_9_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.um_user_9_face.setPixmap(pix)
        self.um_user_9_face.setGeometry(int(450 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.um_user_9_face.setStyleSheet(L_BG_F)

        self.um_user_9_edit_btn = QtWidgets.QPushButton(self)
        self.um_user_9_edit_btn.setGeometry(int(420 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE),
                                              int(25 * Y_SCALE))
        self.um_user_9_edit_btn.setText("Bearbeiten")
        self.um_user_9_edit_btn.setStyleSheet(L_BT_N)

        self.um_user_9_delete_btn = QtWidgets.QPushButton(self)
        self.um_user_9_delete_btn.setGeometry(int(480 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE),
                                                int(25 * Y_SCALE))
        self.um_user_9_delete_btn.setText("Löschen")
        self.um_user_9_delete_btn.setStyleSheet(L_BT_N)

        self.um_user_10 = QtWidgets.QLabel(self)
        self.um_user_10.setGeometry(int(550 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.um_user_10.setStyleSheet(L_BG_PERSON)

        self.um_user_10_detail_text = QtWidgets.QLabel(self)
        self.um_user_10_detail_text.setText("Name\nPrivilege\nCreator\nPhone\nBlocked")
        self.um_user_10_detail_text.setGeometry(int(560 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.um_user_10_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.um_user_10_detail_value = QtWidgets.QLabel(self)
        self.um_user_10_detail_value.setGeometry(int(610 * X_SCALE), int(380 * Y_SCALE), int(65 * X_SCALE),
                                                   int(125 * Y_SCALE))
        self.um_user_10_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.um_user_10_face = QtWidgets.QLabel(self)
        self.um_user_10_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.um_user_10_face.setPixmap(pix)
        self.um_user_10_face.setGeometry(int(585 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE),
                                           int(100 * Y_SCALE))
        self.um_user_10_face.setStyleSheet(L_BG_F)

        self.um_user_10_edit_btn = QtWidgets.QPushButton(self)
        self.um_user_10_edit_btn.setGeometry(int(555 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE),
                                               int(25 * Y_SCALE))
        self.um_user_10_edit_btn.setText("Bearbeiten")
        self.um_user_10_edit_btn.setStyleSheet(L_BT_N)

        self.um_user_10_delete_btn = QtWidgets.QPushButton(self)
        self.um_user_10_delete_btn.setGeometry(int(615 * X_SCALE), int(495 * Y_SCALE), int(60 * X_SCALE),
                                                 int(25 * Y_SCALE))
        self.um_user_10_delete_btn.setText("Löschen")
        self.um_user_10_delete_btn.setStyleSheet(L_BT_N)

        self.um_add_btn = QtWidgets.QPushButton(self)
        self.um_add_btn.setGeometry(int(595 * X_SCALE), int(555 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.um_add_btn.setText("Hinzufügen")
        self.um_add_btn.setStyleSheet(L_BT_N)

        self.um_prev_e_btn = QtWidgets.QLabel(self)
        self.um_prev_e_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/prev_e.png")
        self.um_prev_e_btn.setPixmap(pix)
        self.um_prev_e_btn.setGeometry(int(225 * X_SCALE), int(560 * Y_SCALE), int(10 * X_SCALE), int(15 * Y_SCALE))
        self.um_prev_e_btn.setStyleSheet(L_BG)

        self.um_prev_btn = QtWidgets.QLabel(self)
        self.um_prev_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/prev.png")
        self.um_prev_btn.setPixmap(pix)
        self.um_prev_btn.setGeometry(int(245 * X_SCALE), int(560 * Y_SCALE), int(15 * X_SCALE), int(15 * Y_SCALE))
        self.um_prev_btn.setStyleSheet(L_BG)

        self.um_next_e_btn = QtWidgets.QLabel(self)
        self.um_next_e_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/next_e.png")
        self.um_next_e_btn.setPixmap(pix)
        self.um_next_e_btn.setGeometry(int(340 * X_SCALE), int(560 * Y_SCALE), int(10 * X_SCALE), int(15 * Y_SCALE))
        self.um_next_e_btn.setStyleSheet(L_BG)

        self.um_next_btn = QtWidgets.QLabel(self)
        self.um_next_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/next.png")
        self.um_next_btn.setPixmap(pix)
        self.um_next_btn.setGeometry(int(315 * X_SCALE), int(560 * Y_SCALE), int(15 * X_SCALE), int(15 * Y_SCALE))
        self.um_next_btn.setStyleSheet(L_BG)

        self.um_page_num = QtWidgets.QLabel(self)
        self.um_page_num.setText("1/1")
        self.um_page_num.setAlignment(QtCore.Qt.AlignCenter)
        self.um_page_num.setGeometry(int(270 * X_SCALE), int(560 * Y_SCALE), int(35 * X_SCALE), int(15 * Y_SCALE))
        self.um_page_num.setStyleSheet(L_BG_F)

        #######################  User Edit Page  ##################
        self.ue_text1 = QtWidgets.QLabel(self)
        self.ue_text1.setText("Benutzer Information")
        self.ue_text1.setGeometry(int(150 * X_SCALE), int(100 * Y_SCALE), int(100 * X_SCALE), int(30 * Y_SCALE))
        self.ue_text1.setStyleSheet(L_BG)

        self.ue_label1 = QtWidgets.QLabel(self)
        self.ue_label1.setGeometry(int(150 * X_SCALE), int(130 * Y_SCALE), int(400 * X_SCALE), int(370 * Y_SCALE))
        self.ue_label1.setStyleSheet(L_ST_TEXT)

        self.ue_text2 = QtWidgets.QLabel(self)
        self.ue_text2.setText("Benutzer Name")
        self.ue_text2.setAlignment(QtCore.Qt.AlignRight)
        self.ue_text2.setGeometry(int(150 * X_SCALE), int(180 * Y_SCALE), int(150 * X_SCALE), int(30 * Y_SCALE))
        self.ue_text2.setStyleSheet(L_ST_TEXT)

        self.ue_usrname_value = QtWidgets.QLineEdit(self)
        self.ue_usrname_value.setText("Admin")
        self.ue_usrname_value.setGeometry(int(310 * X_SCALE), int(175 * Y_SCALE), int(170 * X_SCALE), int(25 * Y_SCALE))
        self.ue_usrname_value.setStyleSheet(L_ST_VAL)

        self.ue_text3 = QtWidgets.QLabel(self)
        self.ue_text3.setText("Benutzergruppe")
        self.ue_text3.setAlignment(QtCore.Qt.AlignRight)
        self.ue_text3.setGeometry(int(150 * X_SCALE), int(220 * Y_SCALE), int(150 * X_SCALE), int(30 * Y_SCALE))
        self.ue_text3.setStyleSheet(L_ST_TEXT)

        self.ue_usrgroup_value = QtWidgets.QComboBox(self)
        self.ue_usrgroup_value.addItem("Administrator")
        self.ue_usrgroup_value.addItem("Mitarbeiter")
        self.ue_usrgroup_value.setGeometry(int(310 * X_SCALE), int(215 * Y_SCALE), int(170 * X_SCALE), int(25 * Y_SCALE))
        self.ue_usrgroup_value.setStyleSheet(L_ST_VAL)

        self.ue_text4 = QtWidgets.QLabel(self)
        self.ue_text4.setText("Passwort")
        self.ue_text4.setAlignment(QtCore.Qt.AlignRight)
        self.ue_text4.setGeometry(int(150 * X_SCALE), int(260 * Y_SCALE), int(150 * X_SCALE), int(30 * Y_SCALE))
        self.ue_text4.setStyleSheet(L_ST_TEXT)

        self.ue_pass_value = QtWidgets.QLineEdit(self)
        self.ue_pass_value.setEchoMode(QtWidgets.QLineEdit.Password)
        self.ue_pass_value.setGeometry(int(310 * X_SCALE), int(255 * Y_SCALE), int(170 * X_SCALE), int(25 * Y_SCALE))
        self.ue_pass_value.setStyleSheet(L_ST_VAL)

        self.ue_text6 = QtWidgets.QLabel(self)
        self.ue_text6.setText("Passwort bestätigen")
        self.ue_text6.setAlignment(QtCore.Qt.AlignRight)
        self.ue_text6.setGeometry(int(150 * X_SCALE), int(300 * Y_SCALE), int(150 * X_SCALE), int(30 * Y_SCALE))
        self.ue_text6.setStyleSheet(L_ST_TEXT)

        self.ue_passconfirm_value = QtWidgets.QLineEdit(self)
        self.ue_passconfirm_value.setEchoMode(QtWidgets.QLineEdit.Password)
        self.ue_passconfirm_value.setGeometry(int(310 * X_SCALE), int(295 * Y_SCALE), int(170 * X_SCALE), int(25 * Y_SCALE))
        self.ue_passconfirm_value.setStyleSheet(L_ST_VAL)

        self.ue_text7 = QtWidgets.QLabel(self)
        self.ue_text7.setText("Telefon")
        self.ue_text7.setAlignment(QtCore.Qt.AlignRight)
        self.ue_text7.setGeometry(int(150 * X_SCALE), int(340 * Y_SCALE), int(150 * X_SCALE), int(30 * Y_SCALE))
        self.ue_text7.setStyleSheet(L_ST_TEXT)

        self.ue_phone_value = QtWidgets.QLineEdit(self)
        self.ue_phone_value.setGeometry(int(310 * X_SCALE), int(335 * Y_SCALE), int(170 * X_SCALE), int(25 * Y_SCALE))
        self.ue_phone_value.setStyleSheet(L_ST_VAL)

        self.ue_text8 = QtWidgets.QLabel(self)
        self.ue_text8.setText("Erstellt")
        self.ue_text8.setAlignment(QtCore.Qt.AlignRight)
        self.ue_text8.setGeometry(int(150 * X_SCALE), int(380 * Y_SCALE), int(150 * X_SCALE), int(30 * Y_SCALE))
        self.ue_text8.setStyleSheet(L_ST_TEXT)

        self.ue_creator_value = QtWidgets.QLineEdit(self)
        self.ue_creator_value.setGeometry(int(310 * X_SCALE), int(375 * Y_SCALE), int(170 * X_SCALE), int(25 * Y_SCALE))
        self.ue_creator_value.setStyleSheet(L_ST_VAL)

        self.ue_text9 = QtWidgets.QLabel(self)
        self.ue_text9.setText("Gesperrt")
        self.ue_text9.setAlignment(QtCore.Qt.AlignRight)
        self.ue_text9.setGeometry(int(150 * X_SCALE), int(420 * Y_SCALE), int(150 * X_SCALE), int(30 * Y_SCALE))
        self.ue_text9.setStyleSheet(L_ST_TEXT)

        self.ue_blocked_value = QtWidgets.QComboBox(self)
        self.ue_blocked_value.addItem("Nein")
        self.ue_blocked_value.addItem("Ja")
        self.ue_blocked_value.setGeometry(int(310 * X_SCALE), int(415 * Y_SCALE), int(170 * X_SCALE), int(25 * Y_SCALE))
        self.ue_blocked_value.setStyleSheet(L_ST_VAL)

        self.ue_save_btn = QtWidgets.QPushButton(self)
        self.ue_save_btn.setGeometry(int(480 * X_SCALE), int(555 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.ue_save_btn.setText("Speichern")
        self.ue_save_btn.setStyleSheet(L_BT_N)

        self.ue_cancel_btn = QtWidgets.QPushButton(self)
        self.ue_cancel_btn.setGeometry(int(580 * X_SCALE), int(555 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.ue_cancel_btn.setText("Abbruch")
        self.ue_cancel_btn.setStyleSheet(L_BT_N)

        #######################  User History Page  ###############
        self.uh_user_1 = QtWidgets.QLabel(self)
        self.uh_user_1.setGeometry(int(10 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.uh_user_1.setStyleSheet(L_BG_PERSON)

        self.uh_user_1_detail_text = QtWidgets.QLabel(self)
        self.uh_user_1_detail_text.setGeometry(int(20 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.uh_user_1_detail_text.setText("Benutzer\nAktion\nZeit\nInfo")
        self.uh_user_1_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.uh_user_1_detail_value = QtWidgets.QLabel(self)
        self.uh_user_1_detail_value.setGeometry(int(55 * X_SCALE), int(135 * Y_SCALE), int(80 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.uh_user_1_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.uh_user_1_face = QtWidgets.QLabel(self)
        self.uh_user_1_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.uh_user_1_face.setPixmap(pix)
        self.uh_user_1_face.setGeometry(int(45 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.uh_user_1_face.setStyleSheet(L_BG_F)

        self.uh_user_2 = QtWidgets.QLabel(self)
        self.uh_user_2.setGeometry(int(145 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.uh_user_2.setStyleSheet(L_BG_PERSON)

        self.uh_user_2_detail_text = QtWidgets.QLabel(self)
        self.uh_user_2_detail_text.setGeometry(int(155 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.uh_user_2_detail_text.setText("Benutzer\nAktion\nZeit\nInfo")
        self.uh_user_2_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.uh_user_2_detail_value = QtWidgets.QLabel(self)
        self.uh_user_2_detail_value.setGeometry(int(190 * X_SCALE), int(135 * Y_SCALE), int(80 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.uh_user_2_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.uh_user_2_face = QtWidgets.QLabel(self)
        self.uh_user_2_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.uh_user_2_face.setPixmap(pix)
        self.uh_user_2_face.setGeometry(int(180 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.uh_user_2_face.setStyleSheet(L_BG_F)

        self.uh_user_3 = QtWidgets.QLabel(self)
        self.uh_user_3.setGeometry(int(280 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.uh_user_3.setStyleSheet(L_BG_PERSON)

        self.uh_user_3_detail_text = QtWidgets.QLabel(self)
        self.uh_user_3_detail_text.setGeometry(int(290 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.uh_user_3_detail_text.setText("Benutzer\nAktion\nZeit\nInfo")
        self.uh_user_3_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.uh_user_3_detail_value = QtWidgets.QLabel(self)
        self.uh_user_3_detail_value.setGeometry(int(325 * X_SCALE), int(135 * Y_SCALE), int(80 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.uh_user_3_detail_value.setText("admin\nlogged in\n2024-02-20 12:34:43\n")
        self.uh_user_3_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.uh_user_3_face = QtWidgets.QLabel(self)
        self.uh_user_3_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.uh_user_3_face.setPixmap(pix)
        self.uh_user_3_face.setGeometry(int(315 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.uh_user_3_face.setStyleSheet(L_BG_F)

        self.uh_user_4 = QtWidgets.QLabel(self)
        self.uh_user_4.setGeometry(int(415 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.uh_user_4.setStyleSheet(L_BG_PERSON)

        self.uh_user_4_detail_text = QtWidgets.QLabel(self)
        self.uh_user_4_detail_text.setGeometry(int(425 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.uh_user_4_detail_text.setText("Benutzer\nAktion\nZeit\nInfo")
        self.uh_user_4_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.uh_user_4_detail_value = QtWidgets.QLabel(self)
        self.uh_user_4_detail_value.setGeometry(int(460 * X_SCALE), int(135 * Y_SCALE), int(80 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.uh_user_4_detail_value.setText("admin\nlogged in\n2024-02-20 12:34:43\n")
        self.uh_user_4_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.uh_user_4_face = QtWidgets.QLabel(self)
        self.uh_user_4_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.uh_user_4_face.setPixmap(pix)
        self.uh_user_4_face.setGeometry(int(450 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.uh_user_4_face.setStyleSheet(L_BG_F)

        self.uh_user_5 = QtWidgets.QLabel(self)
        self.uh_user_5.setGeometry(int(550 * X_SCALE), int(45 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.uh_user_5.setStyleSheet(L_BG_PERSON)

        self.uh_user_5_detail_text = QtWidgets.QLabel(self)
        self.uh_user_5_detail_text.setGeometry(int(560 * X_SCALE), int(135 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.uh_user_5_detail_text.setText("Benutzer\nAktion\nZeit\nInfo")
        self.uh_user_5_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.uh_user_5_detail_value = QtWidgets.QLabel(self)
        self.uh_user_5_detail_value.setGeometry(int(595 * X_SCALE), int(135 * Y_SCALE), int(80 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.uh_user_5_detail_value.setText("admin\nlogged in\n2024-02-20 12:34:43\n")
        self.uh_user_5_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.uh_user_5_face = QtWidgets.QLabel(self)
        self.uh_user_5_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.uh_user_5_face.setPixmap(pix)
        self.uh_user_5_face.setGeometry(int(585 * X_SCALE), int(50 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.uh_user_5_face.setStyleSheet(L_BG_F)

        self.uh_user_6 = QtWidgets.QLabel(self)
        self.uh_user_6.setGeometry(int(10 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.uh_user_6.setStyleSheet(L_BG_PERSON)

        self.uh_user_6_detail_text = QtWidgets.QLabel(self)
        self.uh_user_6_detail_text.setGeometry(int(20 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.uh_user_6_detail_text.setText("Benutzer\nAktion\nZeit\nInfo")
        self.uh_user_6_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.uh_user_6_detail_value = QtWidgets.QLabel(self)
        self.uh_user_6_detail_value.setGeometry(int(55 * X_SCALE), int(380 * Y_SCALE), int(80 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.uh_user_6_detail_value.setText("admin\nlogged in\n2024-02-20 12:34:43\n")
        self.uh_user_6_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.uh_user_6_face = QtWidgets.QLabel(self)
        self.uh_user_6_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.uh_user_6_face.setPixmap(pix)
        self.uh_user_6_face.setGeometry(int(45 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.uh_user_6_face.setStyleSheet(L_BG_F)

        self.uh_user_7 = QtWidgets.QLabel(self)
        self.uh_user_7.setGeometry(int(145 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.uh_user_7.setStyleSheet(L_BG_PERSON)

        self.uh_user_7_detail_text = QtWidgets.QLabel(self)
        self.uh_user_7_detail_text.setGeometry(int(155 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.uh_user_7_detail_text.setText("Benutzer\nAktion\nZeit\nInfo")
        self.uh_user_7_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.uh_user_7_detail_value = QtWidgets.QLabel(self)
        self.uh_user_7_detail_value.setGeometry(int(190 * X_SCALE), int(380 * Y_SCALE), int(80 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.uh_user_7_detail_value.setText("admin\nlogged in\n2024-02-20 12:34:43\n")
        self.uh_user_7_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.uh_user_7_face = QtWidgets.QLabel(self)
        self.uh_user_7_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.uh_user_7_face.setPixmap(pix)
        self.uh_user_7_face.setGeometry(int(180 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.uh_user_7_face.setStyleSheet(L_BG_F)

        self.uh_user_8 = QtWidgets.QLabel(self)
        self.uh_user_8.setGeometry(int(280 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.uh_user_8.setStyleSheet(L_BG_PERSON)

        self.uh_user_8_detail_text = QtWidgets.QLabel(self)
        self.uh_user_8_detail_text.setGeometry(int(290 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.uh_user_8_detail_text.setText("Benutzer\nAktion\nZeit\nInfo")
        self.uh_user_8_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.uh_user_8_detail_value = QtWidgets.QLabel(self)
        self.uh_user_8_detail_value.setGeometry(int(325 * X_SCALE), int(380 * Y_SCALE), int(80 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.uh_user_8_detail_value.setText("admin\nlogged in\n2024-02-20 12:34:43\n")
        self.uh_user_8_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.uh_user_8_face = QtWidgets.QLabel(self)
        self.uh_user_8_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.uh_user_8_face.setPixmap(pix)
        self.uh_user_8_face.setGeometry(int(315 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.uh_user_8_face.setStyleSheet(L_BG_F)

        self.uh_user_9 = QtWidgets.QLabel(self)
        self.uh_user_9.setGeometry(int(415 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.uh_user_9.setStyleSheet(L_BG_PERSON)

        self.uh_user_9_detail_text = QtWidgets.QLabel(self)
        self.uh_user_9_detail_text.setGeometry(int(425 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                                 int(125 * Y_SCALE))
        self.uh_user_9_detail_text.setText("Benutzer\nAktion\nZeit\nInfo")
        self.uh_user_9_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.uh_user_9_detail_value = QtWidgets.QLabel(self)
        self.uh_user_9_detail_value.setGeometry(int(460 * X_SCALE), int(380 * Y_SCALE), int(80 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.uh_user_9_detail_value.setText("admin\nlogged in\n2024-02-20 12:34:43\n")
        self.uh_user_9_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.uh_user_9_face = QtWidgets.QLabel(self)
        self.uh_user_9_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.uh_user_9_face.setPixmap(pix)
        self.uh_user_9_face.setGeometry(int(450 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE), int(100 * Y_SCALE))
        self.uh_user_9_face.setStyleSheet(L_BG_F)

        self.uh_user_10 = QtWidgets.QLabel(self)
        self.uh_user_10.setGeometry(int(550 * X_SCALE), int(290 * Y_SCALE), int(130 * X_SCALE), int(240 * Y_SCALE))
        self.uh_user_10.setStyleSheet(L_BG_PERSON)

        self.uh_user_10_detail_text = QtWidgets.QLabel(self)
        self.uh_user_10_detail_text.setGeometry(int(560 * X_SCALE), int(380 * Y_SCALE), int(50 * X_SCALE),
                                                  int(125 * Y_SCALE))
        self.uh_user_10_detail_text.setText("Benutzer\nAktion\nZeit\nInfo")
        self.uh_user_10_detail_text.setStyleSheet(L_DETAIL_TEXT)

        self.uh_user_10_detail_value = QtWidgets.QLabel(self)
        self.uh_user_10_detail_value.setGeometry(int(595 * X_SCALE), int(380 * Y_SCALE), int(80 * X_SCALE),
                                                   int(125 * Y_SCALE))
        self.uh_user_10_detail_value.setText("admin\nlogged in\n2024-02-20 12:34:43\n")
        self.uh_user_10_detail_value.setStyleSheet(L_DETAIL_VALUE)

        self.uh_user_10_face = QtWidgets.QLabel(self)
        self.uh_user_10_face.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.uh_user_10_face.setPixmap(pix)
        self.uh_user_10_face.setGeometry(int(585 * X_SCALE), int(295 * Y_SCALE), int(60 * X_SCALE),
                                           int(100 * Y_SCALE))
        self.uh_user_10_face.setStyleSheet(L_BG_F)

        self.uh_prev_e_btn = QtWidgets.QLabel(self)
        self.uh_prev_e_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/prev_e.png")
        self.uh_prev_e_btn.setPixmap(pix)
        self.uh_prev_e_btn.setGeometry(int(225 * X_SCALE), int(560 * Y_SCALE), int(10 * X_SCALE), int(15 * Y_SCALE))
        self.uh_prev_e_btn.setStyleSheet(L_BG)

        self.uh_prev_btn = QtWidgets.QLabel(self)
        self.uh_prev_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/prev.png")
        self.uh_prev_btn.setPixmap(pix)
        self.uh_prev_btn.setGeometry(int(245 * X_SCALE), int(560 * Y_SCALE), int(15 * X_SCALE), int(15 * Y_SCALE))
        self.uh_prev_btn.setStyleSheet(L_BG)

        self.uh_next_e_btn = QtWidgets.QLabel(self)
        self.uh_next_e_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/next_e.png")
        self.uh_next_e_btn.setPixmap(pix)
        self.uh_next_e_btn.setGeometry(int(340 * X_SCALE), int(560 * Y_SCALE), int(10 * X_SCALE), int(15 * Y_SCALE))
        self.uh_next_e_btn.setStyleSheet(L_BG)

        self.uh_next_btn = QtWidgets.QLabel(self)
        self.uh_next_btn.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/next.png")
        self.uh_next_btn.setPixmap(pix)
        self.uh_next_btn.setGeometry(int(315 * X_SCALE), int(560 * Y_SCALE), int(15 * X_SCALE), int(15 * Y_SCALE))
        self.uh_next_btn.setStyleSheet(L_BG)

        self.uh_page_num = QtWidgets.QLabel(self)
        self.uh_page_num.setText("1/1")
        self.uh_page_num.setAlignment(QtCore.Qt.AlignCenter)
        self.uh_page_num.setGeometry(int(270 * X_SCALE), int(560 * Y_SCALE), int(35 * X_SCALE), int(15 * Y_SCALE))
        self.uh_page_num.setStyleSheet(L_BG_F)

        self.uh_clearall_btn = QtWidgets.QPushButton(self)
        self.uh_clearall_btn.setGeometry(int(595 * X_SCALE), int(555 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.uh_clearall_btn.setText("Clear All")
        self.uh_clearall_btn.setStyleSheet(L_BT_N)

        ######################  Setting Page  ####################
        self.st_text1 = QtWidgets.QLabel(self)
        self.st_text1.setText("Einstellung")
        self.st_text1.setGeometry(int(30 * X_SCALE), int(70 * Y_SCALE), int(100 * X_SCALE), int(30 * Y_SCALE))
        self.st_text1.setStyleSheet(L_BG)

        self.st_label1 = QtWidgets.QLabel(self)
        self.st_label1.setGeometry(int(30 * X_SCALE), int(100 * Y_SCALE), int(300 * X_SCALE), int(200 * Y_SCALE))
        self.st_label1.setStyleSheet(L_ST_TEXT)

        self.st_text2 = QtWidgets.QLabel(self)
        self.st_text2.setText("Sprache")
        self.st_text2.setAlignment(QtCore.Qt.AlignRight)
        self.st_text2.setGeometry(int(30 * X_SCALE), int(120 * Y_SCALE), int(100 * X_SCALE), int(30 * Y_SCALE))
        self.st_text2.setStyleSheet(L_ST_TEXT)

        self.st_lang = QtWidgets.QComboBox(self)
        self.st_lang.addItem("Deutsch")
        self.st_lang.addItem("Englisch")
        self.st_lang.setGeometry(int(140 * X_SCALE), int(115 * Y_SCALE), int(100 * X_SCALE), int(25 * Y_SCALE))
        self.st_lang.setStyleSheet(L_ST_VAL)

        self.st_text3 = QtWidgets.QLabel(self)
        self.st_text3.setText("Ort")
        self.st_text3.setAlignment(QtCore.Qt.AlignRight)
        self.st_text3.setGeometry(int(30 * X_SCALE), int(160 * Y_SCALE), int(100 * X_SCALE), int(30 * Y_SCALE))
        self.st_text3.setStyleSheet(L_ST_TEXT)

        self.st_place_value = QtWidgets.QLineEdit(self)
        self.st_place_value.setText("Nicht zugewiesen")
        self.st_place_value.setGeometry(int(140 * X_SCALE), int(155 * Y_SCALE), int(140 * X_SCALE), int(25 * Y_SCALE))
        self.st_place_value.setStyleSheet(L_ST_VAL)

        self.st_text4 = QtWidgets.QLabel(self)
        self.st_text4.setText("Synchronize Time")
        self.st_text4.setAlignment(QtCore.Qt.AlignRight)
        self.st_text4.setGeometry(int(30 * X_SCALE), int(200 * Y_SCALE), int(100 * X_SCALE), int(30 * Y_SCALE))
        self.st_text4.setStyleSheet(L_ST_TEXT)

        self.st_time_value = QtWidgets.QLineEdit(self)
        self.st_time_value.setText("5")
        self.st_time_value.setGeometry(int(140 * X_SCALE), int(195 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.st_time_value.setStyleSheet(L_ST_VAL)

        self.st_text5 = QtWidgets.QLabel(self)
        self.st_text5.setText("Minuten")
        self.st_text5.setAlignment(QtCore.Qt.AlignLeft)
        self.st_text5.setGeometry(int(230 * X_SCALE), int(200 * Y_SCALE), int(50 * X_SCALE), int(30 * Y_SCALE))
        self.st_text5.setStyleSheet(L_ST_TEXT)

        self.st_text6 = QtWidgets.QLabel(self)
        self.st_text6.setText("Kamera")
        self.st_text6.setAlignment(QtCore.Qt.AlignRight)
        self.st_text6.setGeometry(int(30 * X_SCALE), int(240 * Y_SCALE), int(100 * X_SCALE), int(30 * Y_SCALE))
        self.st_text6.setStyleSheet(L_ST_TEXT)

        self.st_mode = QtWidgets.QComboBox(self)
        self.st_mode.addItem("1*1")
        self.st_mode.addItem("2*2")
        self.st_mode.setGeometry(int(140 * X_SCALE), int(235 * Y_SCALE), int(100 * X_SCALE), int(25 * Y_SCALE))
        self.st_mode.setStyleSheet(L_ST_VAL)

        self.st_text7 = QtWidgets.QLabel(self)
        self.st_text7.setText("Datenbank")
        self.st_text7.setGeometry(int(370 * X_SCALE), int(70 * Y_SCALE), int(100 * X_SCALE), int(30 * Y_SCALE))
        self.st_text7.setStyleSheet(L_BG)

        self.st_label2 = QtWidgets.QLabel(self)
        self.st_label2.setGeometry(int(370 * X_SCALE), int(100 * Y_SCALE), int(300 * X_SCALE), int(200 * Y_SCALE))
        self.st_label2.setStyleSheet(L_ST_TEXT)

        self.st_text8 = QtWidgets.QLabel(self)
        self.st_text8.setText("Adresse")
        self.st_text8.setAlignment(QtCore.Qt.AlignRight)
        self.st_text8.setGeometry(int(370 * X_SCALE), int(140 * Y_SCALE), int(100 * X_SCALE), int(30 * Y_SCALE))
        self.st_text8.setStyleSheet(L_ST_TEXT)

        self.st_text9 = QtWidgets.QLabel(self)
        self.st_text9.setText("Benutzer")
        self.st_text9.setAlignment(QtCore.Qt.AlignRight)
        self.st_text9.setGeometry(int(370 * X_SCALE), int(180 * Y_SCALE), int(100 * X_SCALE), int(30 * Y_SCALE))
        self.st_text9.setStyleSheet(L_ST_TEXT)

        self.st_text10 = QtWidgets.QLabel(self)
        self.st_text10.setText("Passwort")
        self.st_text10.setAlignment(QtCore.Qt.AlignRight)
        self.st_text10.setGeometry(int(370 * X_SCALE), int(220 * Y_SCALE), int(100 * X_SCALE), int(30 * Y_SCALE))
        self.st_text10.setStyleSheet(L_ST_TEXT)

        self.st_address_value = QtWidgets.QLineEdit(self)
        self.st_address_value.setText("127.0.0.1")
        self.st_address_value.setGeometry(int(480 * X_SCALE), int(135 * Y_SCALE), int(140 * X_SCALE), int(25 * Y_SCALE))
        self.st_address_value.setStyleSheet(L_ST_VAL)

        self.st_usrname_value = QtWidgets.QLineEdit(self)
        self.st_usrname_value.setText("root")
        self.st_usrname_value.setGeometry(int(480 * X_SCALE), int(175 * Y_SCALE), int(140 * X_SCALE), int(25 * Y_SCALE))
        self.st_usrname_value.setStyleSheet(L_ST_VAL)

        self.st_pass = QtWidgets.QLineEdit(self)
        self.st_pass.setText("**********")
        self.st_pass.setGeometry(int(480 * X_SCALE), int(215 * Y_SCALE), int(140 * X_SCALE), int(25 * Y_SCALE))
        self.st_pass.setStyleSheet(L_ST_VAL)

        self.st_default_btn = QtWidgets.QPushButton(self)
        self.st_default_btn.setGeometry(int(500 * X_SCALE), int(555 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.st_default_btn.setText("Default")
        self.st_default_btn.setStyleSheet(L_BT_N)

        self.st_apply_btn = QtWidgets.QPushButton(self)
        self.st_apply_btn.setGeometry(int(590 * X_SCALE), int(555 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.st_apply_btn.setText("Speichern")
        self.st_apply_btn.setStyleSheet(L_BT_N)

        #######################  Main Buttons  ###################

        self.monitor_btn = QtWidgets.QPushButton(self)
        self.monitor_btn.setIcon(QtGui.QIcon("./icons/monitor_b.png"))
        self.monitor_btn.setGeometry(int(695 * X_SCALE), int(40 * Y_SCALE), int(100 * X_SCALE), int(50 * Y_SCALE))
        self.monitor_btn.setText("  Monitor")
        self.monitor_btn.setStyleSheet(L_BT_P)

        self.perman_btn = QtWidgets.QPushButton(self)
        self.perman_btn.setIcon(QtGui.QIcon("./icons/person.png"))
        self.perman_btn.setGeometry(int(695 * X_SCALE), int(90 * Y_SCALE), int(100 * X_SCALE), int(50 * Y_SCALE))
        self.perman_btn.setText("  Personen Manager")
        self.perman_btn.setStyleSheet(L_BT_N)

        self.perhis_btn = QtWidgets.QPushButton(self)
        self.perhis_btn.setIcon(QtGui.QIcon("./icons/history.png"))
        self.perhis_btn.setGeometry(int(695 * X_SCALE), int(140 * Y_SCALE), int(100 * X_SCALE), int(50 * Y_SCALE))
        self.perhis_btn.setText("  Personen Historie")
        self.perhis_btn.setStyleSheet(L_BT_N)

        self.usrman_btn = QtWidgets.QPushButton(self)
        self.usrman_btn.setIcon(QtGui.QIcon("./icons/person.png"))
        self.usrman_btn.setGeometry(int(695 * X_SCALE), int(190 * Y_SCALE), int(100 * X_SCALE), int(50 * Y_SCALE))
        self.usrman_btn.setText("  Benutzer Manager")
        self.usrman_btn.setStyleSheet(L_BT_N)

        self.usrhis_btn = QtWidgets.QPushButton(self)
        self.usrhis_btn.setIcon(QtGui.QIcon("./icons/history.png"))
        self.usrhis_btn.setGeometry(int(695 * X_SCALE), int(240 * Y_SCALE), int(100 * X_SCALE), int(50 * Y_SCALE))
        self.usrhis_btn.setText("  Benutzer Historie")
        self.usrhis_btn.setStyleSheet(L_BT_N)

        self.setting_btn = QtWidgets.QPushButton(self)
        self.setting_btn.setIcon(QtGui.QIcon("./icons/setting.png"))
        self.setting_btn.setGeometry(int(695 * X_SCALE), int(290 * Y_SCALE), int(100 * X_SCALE), int(50 * Y_SCALE))
        self.setting_btn.setText("  Einstellung")
        self.setting_btn.setStyleSheet(L_BT_N)

        self.exit_btn = QtWidgets.QPushButton(self)
        self.exit_btn.setGeometry(int(695 * X_SCALE), int(530 * Y_SCALE), int(100 * X_SCALE), int(50 * Y_SCALE))
        self.exit_btn.setText("Exit")
        self.exit_btn.setStyleSheet(L_BT_N)

        #######################  Face and ID Doc view page  ######
        self.cp_bg_label = QtWidgets.QLabel(self)
        self.cp_bg_label.setStyleSheet(T_BG)
        self.cp_bg_label.setGeometry(int(0 * X_SCALE), int(0 * Y_SCALE), int(800 * X_SCALE), int(600 * Y_SCALE))

        self.cp_frame_label = QtWidgets.QLabel(self)
        self.cp_frame_label.setStyleSheet(L_BG)
        self.cp_frame_label.setGeometry(int(150 * X_SCALE), int(100 * Y_SCALE), int(500 * X_SCALE), int(400 * Y_SCALE))

        self.cp_camera_icon = QtWidgets.QLabel(self)
        self.cp_camera_icon.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/camera.png")
        self.cp_camera_icon.setPixmap(pix)
        self.cp_camera_icon.setStyleSheet(L_BG)
        self.cp_camera_icon.setGeometry(int(320 * X_SCALE), int(200 * Y_SCALE), int(160 * X_SCALE), int(200 * Y_SCALE))

        self.cp_capture_btn = QtWidgets.QPushButton(self)
        self.cp_capture_btn.setGeometry(int(480 * X_SCALE), int(555 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.cp_capture_btn.setText("Aufnehemen")
        self.cp_capture_btn.setStyleSheet(L_BT_N)

        self.cp_cancel_btn = QtWidgets.QPushButton(self)
        self.cp_cancel_btn.setGeometry(int(580 * X_SCALE), int(555 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.cp_cancel_btn.setText("Abbruch")
        self.cp_cancel_btn.setStyleSheet(L_BT_N)

        #######################  Capture Signature page  ######
        self.sign_bg_label = QtWidgets.QLabel(self)
        self.sign_bg_label.setStyleSheet(T_BG)
        self.sign_bg_label.setGeometry(int(0 * X_SCALE), int(0 * Y_SCALE), int(800 * X_SCALE), int(600 * Y_SCALE))

        self.sign_frame_label = QtWidgets.QLabel(self)
        self.sign_frame_label.setStyleSheet(L_BG)
        self.sign_frame_label.setGeometry(int(200 * X_SCALE), int(150 * Y_SCALE), int(400 * X_SCALE), int(300 * Y_SCALE))

        self.sign_canvas = QtGui.QPixmap(int(400 * X_SCALE), int(300 * Y_SCALE))
        self.sign_canvas.fill(QtCore.Qt.white)
        self.sign_frame_label.setPixmap(self.sign_canvas)

        self.painter = QtGui.QPainter(self.sign_canvas)
        self.painter.setPen(QtGui.QPen(Qt.black, 5, Qt.SolidLine,
                            Qt.RoundCap, Qt.RoundJoin))

        self.sign_frame_label.mouseMoveEvent = self.mouse_move
        self.sign_frame_label.mouseReleaseEvent = self.mouse_release

        self.sign_clear_btn = QtWidgets.QPushButton(self)
        self.sign_clear_btn.setGeometry(int(260 * X_SCALE), int(460 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.sign_clear_btn.setText("Löschen")
        self.sign_clear_btn.setStyleSheet(L_BT_N)

        self.sign_capture_btn = QtWidgets.QPushButton(self)
        self.sign_capture_btn.setGeometry(int(360 * X_SCALE), int(460 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.sign_capture_btn.setText("Aufnehmen")
        self.sign_capture_btn.setStyleSheet(L_BT_N)

        self.sign_cancel_btn = QtWidgets.QPushButton(self)
        self.sign_cancel_btn.setGeometry(int(460 * X_SCALE), int(460 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.sign_cancel_btn.setText("Abbruch")
        self.sign_cancel_btn.setStyleSheet(L_BT_N)

        self.last_x, self.last_y = None, None

        ###########  Log in View  #####################
        self.login_bg_label = QtWidgets.QLabel(self)
        self.login_bg_label.setStyleSheet(T_BG)
        self.login_bg_label.setGeometry(int(0 * X_SCALE), int(40 * Y_SCALE), int(800 * X_SCALE), int(600 * Y_SCALE))

        self.login_frame_label = QtWidgets.QLabel(self)
        self.login_frame_label.setStyleSheet(L_BG_T)
        self.login_frame_label.setGeometry(int(200 * X_SCALE), int(150 * Y_SCALE), int(400 * X_SCALE), int(300 * Y_SCALE))

        self.login_face_icon = QtWidgets.QLabel(self)
        self.login_face_icon.setScaledContents(True)
        pix = QtGui.QPixmap("./icons/face_icon.png")
        self.login_face_icon.setPixmap(pix)
        self.login_face_icon.setGeometry(int(295 * X_SCALE), int(170 * Y_SCALE), int(40 * X_SCALE), int(60 * Y_SCALE))
        self.login_face_icon.setStyleSheet(L_BG_F)

        self.login_text1 = QtWidgets.QLabel(self)
        self.login_text1.setText("Log in")
        self.login_text1.setGeometry(int(350 * X_SCALE), int(210 * Y_SCALE), int(60 * X_SCALE), int(25 * Y_SCALE))
        self.login_text1.setStyleSheet(L_BG_T)

        self.login_text2 = QtWidgets.QLabel(self)
        self.login_text2.setText("Benutzer :")
        self.login_text2.setAlignment(QtCore.Qt.AlignRight)
        self.login_text2.setGeometry(int(240 * X_SCALE), int(290 * Y_SCALE), int(100 * X_SCALE), int(25 * Y_SCALE))
        self.login_text2.setStyleSheet(L_BG_L)

        self.login_text3 = QtWidgets.QLabel(self)
        self.login_text3.setText("Passwort :")
        self.login_text3.setAlignment(QtCore.Qt.AlignRight)
        self.login_text3.setGeometry(int(240 * X_SCALE), int(330 * Y_SCALE), int(100 * X_SCALE), int(25 * Y_SCALE))
        self.login_text3.setStyleSheet(L_BG_L)

        self.login_username_value = QtWidgets.QLineEdit(self)
        self.login_username_value.setGeometry(int(350 * X_SCALE), int(290 * Y_SCALE), int(140 * X_SCALE), int(20 * Y_SCALE))
        self.login_username_value.setStyleSheet(L_ST_VAL)
        self.login_username_value.setFocus()

        self.login_pass_value = QtWidgets.QLineEdit(self)
        self.login_pass_value.setEchoMode(QtWidgets.QLineEdit.Password)
        self.login_pass_value.setGeometry(int(350 * X_SCALE), int(330 * Y_SCALE), int(140 * X_SCALE), int(20 * Y_SCALE))
        self.login_pass_value.setStyleSheet(L_ST_VAL)

        self.login_login_btn = QtWidgets.QPushButton(self)
        self.login_login_btn.setGeometry(int(360 * X_SCALE), int(410 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.login_login_btn.setText("Login")
        self.login_login_btn.setStyleSheet(L_BT_N)

        self.login_cancel_btn = QtWidgets.QPushButton(self)
        self.login_cancel_btn.setGeometry(int(460 * X_SCALE), int(410 * Y_SCALE), int(80 * X_SCALE), int(25 * Y_SCALE))
        self.login_cancel_btn.setText("Abbruch")
        self.login_cancel_btn.setStyleSheet(L_BT_N)
        #############################################################
        self.monitor_btn.pressed.connect(self.monitor_btn_clicked)
        self.perman_btn.pressed.connect(self.perman_btn_clicked)
        self.perhis_btn.pressed.connect(self.perhis_btn_clicked)
        self.usrman_btn.pressed.connect(self.usrman_btn_clicked)
        self.usrhis_btn.pressed.connect(self.usrhis_btn_clicked)
        self.setting_btn.pressed.connect(self.setting_btn_clicked)
        self.exit_btn.pressed.connect(self.exit_btn_clicked)

        self.pm_add_btn.released.connect(self.pm_add_btn_clicked)
        self.pm_person_1_edit_btn.released.connect(self.pm_person_1_edit_btn_clicked)
        self.pm_person_2_edit_btn.released.connect(self.pm_person_2_edit_btn_clicked)
        self.pm_person_3_edit_btn.released.connect(self.pm_person_3_edit_btn_clicked)
        self.pm_person_4_edit_btn.released.connect(self.pm_person_4_edit_btn_clicked)
        self.pm_person_5_edit_btn.released.connect(self.pm_person_5_edit_btn_clicked)
        self.pm_person_6_edit_btn.released.connect(self.pm_person_6_edit_btn_clicked)
        self.pm_person_7_edit_btn.released.connect(self.pm_person_7_edit_btn_clicked)
        self.pm_person_8_edit_btn.released.connect(self.pm_person_8_edit_btn_clicked)
        self.pm_person_9_edit_btn.released.connect(self.pm_person_9_edit_btn_clicked)
        self.pm_person_10_edit_btn.released.connect(self.pm_person_10_edit_btn_clicked)

        self.pm_person_1_delete_btn.released.connect(self.pm_person_1_delete_btn_clicked)
        self.pm_person_2_delete_btn.released.connect(self.pm_person_2_delete_btn_clicked)
        self.pm_person_3_delete_btn.released.connect(self.pm_person_3_delete_btn_clicked)
        self.pm_person_4_delete_btn.released.connect(self.pm_person_4_delete_btn_clicked)
        self.pm_person_5_delete_btn.released.connect(self.pm_person_5_delete_btn_clicked)
        self.pm_person_6_delete_btn.released.connect(self.pm_person_6_delete_btn_clicked)
        self.pm_person_7_delete_btn.released.connect(self.pm_person_7_delete_btn_clicked)
        self.pm_person_8_delete_btn.released.connect(self.pm_person_8_delete_btn_clicked)
        self.pm_person_9_delete_btn.released.connect(self.pm_person_9_delete_btn_clicked)
        self.pm_person_10_delete_btn.released.connect(self.pm_person_10_delete_btn_clicked)

        self.pe_from_camera_btn.released.connect(self.pe_from_camera_btn_clicked)
        self.pe_from_image_btn.released.connect(self.pe_from_image_btn_clicked)
        self.pe_read_doc_btn.released.connect(self.pe_read_doc_btn_clicked)
        self.pe_sign_btn.released.connect(self.pe_sign_btn_clicked)
        self.pe_oasis_btn.released.connect(self.pe_oasis_btn_clicked)
        self.pe_save_btn.released.connect(self.pe_save_btn_clicked)
        self.pe_cancel_btn.released.connect(self.pe_cancel_btn_clicked)

        self.pe_setting_btn.released.connect(self.pe_setting_btn_clicked)
        self.pe_agreement_btn.released.connect(self.pe_agreement_btn_clicked)

        self.mo_prev_e_btn.mouseReleaseEvent = self.mo_prev_e_btn_clicked
        self.mo_prev_btn.mouseReleaseEvent = self.mo_prev_btn_clicked
        self.mo_next_e_btn.mouseReleaseEvent = self.mo_next_e_btn_clicked
        self.mo_next_btn.mouseReleaseEvent = self.mo_next_btn_clicked
        self.mo_clearall_btn.released.connect(self.mo_clearall_btn_clicked)

        self.pm_prev_e_btn.mouseReleaseEvent = self.pm_prev_e_btn_clicked
        self.pm_prev_btn.mouseReleaseEvent = self.pm_prev_btn_clicked
        self.pm_next_e_btn.mouseReleaseEvent = self.pm_next_e_btn_clicked
        self.pm_next_btn.mouseReleaseEvent = self.pm_next_btn_clicked

        self.ph_prev_e_btn.mouseReleaseEvent = self.ph_prev_e_btn_clicked
        self.ph_prev_btn.mouseReleaseEvent = self.ph_prev_btn_clicked
        self.ph_next_e_btn.mouseReleaseEvent = self.ph_next_e_btn_clicked
        self.ph_next_btn.mouseReleaseEvent = self.ph_next_btn_clicked
        self.ph_clearall_btn.released.connect(self.ph_clearall_btn_clicked)

        self.um_prev_e_btn.mouseReleaseEvent = self.um_prev_e_btn_clicked
        self.um_prev_btn.mouseReleaseEvent = self.um_prev_btn_clicked
        self.um_next_e_btn.mouseReleaseEvent = self.um_next_e_btn_clicked
        self.um_next_btn.mouseReleaseEvent = self.um_next_btn_clicked

        self.uh_prev_e_btn.mouseReleaseEvent = self.uh_prev_e_btn_clicked
        self.uh_prev_btn.mouseReleaseEvent = self.uh_prev_btn_clicked
        self.uh_next_e_btn.mouseReleaseEvent = self.uh_next_e_btn_clicked
        self.uh_next_btn.mouseReleaseEvent = self.uh_next_btn_clicked
        self.uh_clearall_btn.released.connect(self.uh_clearall_btn_clicked)

        self.um_add_btn.released.connect(self.um_add_btn_clicked)
        self.um_user_1_edit_btn.released.connect(self.um_user_1_edit_btn_clicked)
        self.um_user_2_edit_btn.released.connect(self.um_user_2_edit_btn_clicked)
        self.um_user_3_edit_btn.released.connect(self.um_user_3_edit_btn_clicked)
        self.um_user_4_edit_btn.released.connect(self.um_user_4_edit_btn_clicked)
        self.um_user_5_edit_btn.released.connect(self.um_user_5_edit_btn_clicked)
        self.um_user_6_edit_btn.released.connect(self.um_user_6_edit_btn_clicked)
        self.um_user_7_edit_btn.released.connect(self.um_user_7_edit_btn_clicked)
        self.um_user_8_edit_btn.released.connect(self.um_user_8_edit_btn_clicked)
        self.um_user_9_edit_btn.released.connect(self.um_user_9_edit_btn_clicked)
        self.um_user_10_edit_btn.released.connect(self.um_user_10_edit_btn_clicked)

        self.um_user_1_delete_btn.released.connect(self.um_user_1_delete_btn_clicked)
        self.um_user_2_delete_btn.released.connect(self.um_user_2_delete_btn_clicked)
        self.um_user_3_delete_btn.released.connect(self.um_user_3_delete_btn_clicked)
        self.um_user_4_delete_btn.released.connect(self.um_user_4_delete_btn_clicked)
        self.um_user_5_delete_btn.released.connect(self.um_user_5_delete_btn_clicked)
        self.um_user_6_delete_btn.released.connect(self.um_user_6_delete_btn_clicked)
        self.um_user_7_delete_btn.released.connect(self.um_user_7_delete_btn_clicked)
        self.um_user_8_delete_btn.released.connect(self.um_user_8_delete_btn_clicked)
        self.um_user_9_delete_btn.released.connect(self.um_user_9_delete_btn_clicked)
        self.um_user_10_delete_btn.released.connect(self.um_user_10_delete_btn_clicked)

        self.ue_save_btn.released.connect(self.ue_save_btn_clicked)
        self.ue_cancel_btn.released.connect(self.ue_cancel_btn_clicked)

        self.cp_capture_btn.released.connect(self.cp_capture_btn_clicked)
        self.cp_cancel_btn.released.connect(self.cp_cancel_btn_clicked)

        self.sign_clear_btn.released.connect(self.sign_clear_btn_clicked)
        self.sign_capture_btn.released.connect(self.sign_capture_btn_clicked)
        self.sign_cancel_btn.released.connect(self.sign_cancel_btn_clicked)

        self.login_login_btn.released.connect(self.login_login_btn_clicked)
        self.login_cancel_btn.released.connect(self.login_cancel_btn_clicked)
        # -------------------
        self.cam1_capture_signal.connect(self.draw_cam1_frame)
        self.cam2_capture_signal.connect(self.draw_cam2_frame)
        self.cam3_capture_signal.connect(self.draw_cam3_frame)
        self.cam4_capture_signal.connect(self.draw_cam4_frame)

        self.register_capture_signal.connect(self.draw_capture_frame)

        self.update_detected_persons_signal.connect(self.update_mo_show_persons)

        self.add_person_history_signal.connect(self.add_person_history)

        ###############################################
        self.face_detector1 = FaceDetector()
        self.face_detector2 = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.mrz_reader = MRZReader()
        self.db_conn = sqlite3.connect('database/face_db.db')
        self.db_conn.text_factory = lambda b: b.decode(errors='ignore')

        # -------  import registered person list from db --------
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT * FROM person_register_table")
        rows = cursor.fetchall()
        self.db_conn.commit()

        self.person_register_list = []
        for row in rows:
            id = row[0]
            name = row[1]
            birth = row[2]
            age = row[3]
            status = row[4]
            face_bytes = row[5]
            face_img = self.bytes_to_image(face_bytes)
            emb_bytes = row[6]
            emb = self.bytes_to_emb(emb_bytes)
            gender = row[7]
            guesttype = row[8]
            safetytype = row[9]
            blocked = row[10]
            whenfrom = row[11]
            whento = row[12]
            place = row[13]
            reason = row[14]
            type = row[15]
            info = row[16]

            temp = []
            temp.append(id)
            temp.append(name)
            temp.append(birth)
            temp.append(age)
            temp.append(status)
            temp.append(face_img)
            temp.append(emb)
            temp.append(gender)
            temp.append(guesttype)
            temp.append(safetytype)
            temp.append(blocked)
            temp.append(whenfrom)
            temp.append(whento)
            temp.append(place)
            temp.append(reason)
            temp.append(type)
            temp.append(info)

            self.person_register_list.append(temp)

        # -------  import history person list from db  --------
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT * FROM person_history_table")
        rows = cursor.fetchall()
        self.db_conn.commit()

        self.person_history_list = []
        for row in rows:
            id = row[0]
            name = row[1]
            gender = row[2]
            age = row[3]
            time = row[4]
            place = row[5]
            view = row[6]
            action = row[7]
            face_bytes = row[8]
            face_img = self.bytes_to_image(face_bytes)

            temp = []
            temp.append(id)
            temp.append(name)
            temp.append(gender)
            temp.append(age)
            temp.append(time)
            temp.append(place)
            temp.append(view)
            temp.append(action)
            temp.append(face_img)

            self.person_history_list.append(temp)

        # -------  import registered user list from db --------
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT * FROM user_register_table")
        rows = cursor.fetchall()
        self.db_conn.commit()

        self.user_register_list = []
        for row in rows:
            id = row[0]
            name = row[1]
            usergroup = row[2]
            password = row[3]
            creator = row[4]
            phone = row[5]
            blocked = row[6]

            temp = []
            temp.append(id)
            temp.append(name)
            temp.append(usergroup)
            temp.append(password)
            temp.append(creator)
            temp.append(phone)
            temp.append(blocked)

            self.user_register_list.append(temp)

        # -------  import history user list from db  --------
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT * FROM user_history_table")
        rows = cursor.fetchall()
        self.db_conn.commit()

        self.user_history_list = []
        for row in rows:
            id = row[0]
            name = row[1]
            action = row[2]
            time = row[3]
            content = row[4]

            temp = []
            temp.append(id)
            temp.append(name)
            temp.append(action)
            temp.append(time)
            temp.append(content)

            self.user_history_list.append(temp)

        # -------------------------------------------------------
        self.monitor_view_flag = True

        self.now_person_history_list = []

        self.prev_emb = None
        self.prev_status = True
        self.current_status = False

        self.process_detected_faces = False

        # -------  view pagenation setting  -------------
        self.mo_page_iter = 1
        self.mo_total_pages = 1

        self.pm_page_iter = 1
        self.pm_total_pages = 1

        self.ph_page_iter = 1
        self.ph_total_pages = 1

        self.um_page_iter = 1
        self.um_total_pages = 1

        self.uh_page_iter = 1
        self.uh_total_pages = 1

        # -------  entrance camera setting  -------------
        paramFile = "config.ini"
        config_params = configparser.ConfigParser()
        config_params.read(paramFile)

        self.capture_cam_id = config_params.getint("camera", "came_capturing")
        self.capture_cap = cv2.VideoCapture(self.capture_cam_id, cv2.CAP_DSHOW)
        self.capture_cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
        #self.capture_cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.capture_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.capture_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.cam1_id = config_params.getint("camera", "came_entrance")
        self.cam1_frame = None
        self.cam1_faces = None
        self.cam1_frame_ori = None
        self.cap1 = cv2.VideoCapture(self.cam1_id, cv2.CAP_DSHOW)
        self.cap1.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
        #self.cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.cam1_running = False

        self.cam2_id = config_params.getint("camera", "came_exit")
        self.cam2_frame = None
        self.cam2_faces = None
        self.cam2_frame_ori = None
        self.cap2 = cv2.VideoCapture(self.cam2_id, cv2.CAP_DSHOW)
        self.cap2.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
        #self.cap2.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.cam2_running = False

        self.cam3_frame = None
        self.cap3 = cv2.VideoCapture(9, cv2.CAP_DSHOW)
        self.cam3_running = False

        self.cam4_frame = None
        self.cap4 = cv2.VideoCapture(9, cv2.CAP_DSHOW)
        self.cam4_running = False

        self.detected_faces = None
        self.detected_person_action = None

        self.proc_timer = QtCore.QTimer(self)
        self.proc_timer.timeout.connect(self.create_proc_threads)
        self.proc_timer.start(1000)

        self.cam_check_thread = Thread(target=self.check_cameras)
        self.cam_check_thread.setDaemon(True)
        self.cam_check_thread.start()

        self.pe_face_image = None
        self.pe_person_id = None
        self.ue_user_id = None

        self.capture_frame = None
        self.capture_stop_flag = False

        #####################################
        self.showFullScreen()

        # -------------------
        self.show_monitor_view()
        self.hide_person_manage_view()
        self.hide_person_edit_view()
        self.hide_person_history_view()
        self.hide_user_manage_view()
        self.hide_user_edit_view()
        self.hide_user_history_view()
        self.hide_setting_view()
        self.hide_capture_view()
        self.hide_sign_view()
        # self.hide_login_view()

        ##########################

    def blinking_warning_message(self):
        if self.warning_blinking_flag:
            self.warning_unregister_message.setStyleSheet(L_BG_W_2)
            self.warning_danger_message.setStyleSheet(L_BG_W_2)
            self.warning_blinking_flag = False
        else:
            self.warning_unregister_message.setStyleSheet(L_BG_W_1)
            self.warning_danger_message.setStyleSheet(L_BG_W_1)
            self.warning_blinking_flag = True

    def show_login_view(self):
        self.login_bg_label.show()
        self.login_frame_label.show()
        self.login_face_icon.show()
        self.login_text1.show()
        self.login_text2.show()
        self.login_text3.show()
        self.login_username_value.show()
        self.login_pass_value.show()
        self.login_login_btn.show()
        self.login_cancel_btn.show()

    def hide_login_view(self):
        self.login_bg_label.hide()
        self.login_frame_label.hide()
        self.login_face_icon.hide()
        self.login_text1.hide()
        self.login_text2.hide()
        self.login_text3.hide()
        self.login_username_value.hide()
        self.login_pass_value.hide()
        self.login_login_btn.hide()
        self.login_cancel_btn.hide()

    def show_monitor_view(self):
        self.mo_real_face_1.show()
        self.mo_saved_face_1.show()
        self.mo_detail_1.show()
        self.mo_icon_1.show()
        self.mo_real_face_2.show()
        self.mo_saved_face_2.show()
        self.mo_detail_2.show()
        self.mo_icon_2.show()
        self.mo_real_face_3.show()
        self.mo_saved_face_3.show()
        self.mo_detail_3.show()
        self.mo_icon_3.show()
        self.mo_real_face_4.show()
        self.mo_saved_face_4.show()
        self.mo_detail_4.show()
        self.mo_icon_4.show()
        self.mo_real_face_5.show()
        self.mo_saved_face_5.show()
        self.mo_detail_5.show()
        self.mo_icon_5.show()
        self.mo_real_face_6.show()
        self.mo_saved_face_6.show()
        self.mo_detail_6.show()
        self.mo_icon_6.show()
        self.mo_real_face_7.show()
        self.mo_saved_face_7.show()
        self.mo_detail_7.show()
        self.mo_icon_7.show()
        self.mo_prev_e_btn.show()
        self.mo_prev_btn.show()
        self.mo_next_e_btn.show()
        self.mo_next_btn.show()
        self.mo_page_num.show()
        self.mo_clearall_btn.show()
        self.mo_camera1_view.show()
        self.mo_camera2_view.show()
        self.mo_camera3_view.show()
        self.mo_camera4_view.show()

        if not self.cam1_running:
            self.mo_camera1_icon.show()
        if not self.cam2_running:
            self.mo_camera2_icon.show()
        if not self.cam3_running:
            self.mo_camera3_icon.show()
        if not self.cam4_running:
            self.mo_camera4_icon.show()

        self.update_mo_show_persons()

    def hide_montor_view(self):
        self.mo_real_face_1.hide()
        self.mo_saved_face_1.hide()
        self.mo_detail_1.hide()
        self.mo_icon_1.hide()
        self.mo_real_face_2.hide()
        self.mo_saved_face_2.hide()
        self.mo_detail_2.hide()
        self.mo_icon_2.hide()
        self.mo_real_face_3.hide()
        self.mo_saved_face_3.hide()
        self.mo_detail_3.hide()
        self.mo_icon_3.hide()
        self.mo_real_face_4.hide()
        self.mo_saved_face_4.hide()
        self.mo_detail_4.hide()
        self.mo_icon_4.hide()
        self.mo_real_face_5.hide()
        self.mo_saved_face_5.hide()
        self.mo_detail_5.hide()
        self.mo_icon_5.hide()
        self.mo_real_face_6.hide()
        self.mo_saved_face_6.hide()
        self.mo_detail_6.hide()
        self.mo_icon_6.hide()
        self.mo_real_face_7.hide()
        self.mo_saved_face_7.hide()
        self.mo_detail_7.hide()
        self.mo_icon_7.hide()
        self.mo_prev_e_btn.hide()
        self.mo_prev_btn.hide()
        self.mo_next_e_btn.hide()
        self.mo_next_btn.hide()
        self.mo_page_num.hide()
        self.mo_clearall_btn.hide()
        self.mo_camera1_view.hide()
        self.mo_camera2_view.hide()
        self.mo_camera3_view.hide()
        self.mo_camera4_view.hide()
        self.mo_camera1_icon.hide()
        self.mo_camera2_icon.hide()
        self.mo_camera3_icon.hide()
        self.mo_camera4_icon.hide()

    def show_person_manage_view(self):
        self.pm_person_1.show()
        self.pm_person_1_detail_text.show()
        self.pm_person_1_detail_value.show()
        self.pm_person_1_face.show()
        self.pm_person_1_edit_btn.show()
        self.pm_person_1_delete_btn.show()
        self.pm_person_2.show()
        self.pm_person_2_detail_text.show()
        self.pm_person_2_detail_value.show()
        self.pm_person_2_face.show()
        self.pm_person_2_edit_btn.show()
        self.pm_person_2_delete_btn.show()
        self.pm_person_3.show()
        self.pm_person_3_detail_text.show()
        self.pm_person_3_detail_value.show()
        self.pm_person_3_face.show()
        self.pm_person_3_edit_btn.show()
        self.pm_person_3_delete_btn.show()
        self.pm_person_4.show()
        self.pm_person_4_detail_text.show()
        self.pm_person_4_detail_value.show()
        self.pm_person_4_face.show()
        self.pm_person_4_edit_btn.show()
        self.pm_person_4_delete_btn.show()
        self.pm_person_5.show()
        self.pm_person_5_detail_text.show()
        self.pm_person_5_detail_value.show()
        self.pm_person_5_face.show()
        self.pm_person_5_edit_btn.show()
        self.pm_person_5_delete_btn.show()
        self.pm_person_6.show()
        self.pm_person_6_detail_text.show()
        self.pm_person_6_detail_value.show()
        self.pm_person_6_face.show()
        self.pm_person_6_edit_btn.show()
        self.pm_person_6_delete_btn.show()
        self.pm_person_7.show()
        self.pm_person_7_detail_text.show()
        self.pm_person_7_detail_value.show()
        self.pm_person_7_face.show()
        self.pm_person_7_edit_btn.show()
        self.pm_person_7_delete_btn.show()
        self.pm_person_8.show()
        self.pm_person_8_detail_text.show()
        self.pm_person_8_detail_value.show()
        self.pm_person_8_face.show()
        self.pm_person_8_edit_btn.show()
        self.pm_person_8_delete_btn.show()
        self.pm_person_9.show()
        self.pm_person_9_detail_text.show()
        self.pm_person_9_detail_value.show()
        self.pm_person_9_face.show()
        self.pm_person_9_edit_btn.show()
        self.pm_person_9_delete_btn.show()
        self.pm_person_10.show()
        self.pm_person_10_detail_text.show()
        self.pm_person_10_detail_value.show()
        self.pm_person_10_face.show()
        self.pm_person_10_edit_btn.show()
        self.pm_person_10_delete_btn.show()
        self.pm_add_btn.show()
        self.pm_import_btn.show()
        self.pm_export_btn.show()
        self.pm_prev_e_btn.show()
        self.pm_prev_btn.show()
        self.pm_next_e_btn.show()
        self.pm_next_btn.show()
        self.pm_page_num.show()

        self.update_pm_show_persons()

    def hide_person_manage_view(self):
        self.pm_person_1.hide()
        self.pm_person_1_detail_text.hide()
        self.pm_person_1_detail_value.hide()
        self.pm_person_1_face.hide()
        self.pm_person_1_edit_btn.hide()
        self.pm_person_1_delete_btn.hide()
        self.pm_person_2.hide()
        self.pm_person_2_detail_text.hide()
        self.pm_person_2_detail_value.hide()
        self.pm_person_2_face.hide()
        self.pm_person_2_edit_btn.hide()
        self.pm_person_2_delete_btn.hide()
        self.pm_person_3.hide()
        self.pm_person_3_detail_text.hide()
        self.pm_person_3_detail_value.hide()
        self.pm_person_3_face.hide()
        self.pm_person_3_edit_btn.hide()
        self.pm_person_3_delete_btn.hide()
        self.pm_person_4.hide()
        self.pm_person_4_detail_text.hide()
        self.pm_person_4_detail_value.hide()
        self.pm_person_4_face.hide()
        self.pm_person_4_edit_btn.hide()
        self.pm_person_4_delete_btn.hide()
        self.pm_person_5.hide()
        self.pm_person_5_detail_text.hide()
        self.pm_person_5_detail_value.hide()
        self.pm_person_5_face.hide()
        self.pm_person_5_edit_btn.hide()
        self.pm_person_5_delete_btn.hide()
        self.pm_person_6.hide()
        self.pm_person_6_detail_text.hide()
        self.pm_person_6_detail_value.hide()
        self.pm_person_6_face.hide()
        self.pm_person_6_edit_btn.hide()
        self.pm_person_6_delete_btn.hide()
        self.pm_person_7.hide()
        self.pm_person_7_detail_text.hide()
        self.pm_person_7_detail_value.hide()
        self.pm_person_7_face.hide()
        self.pm_person_7_edit_btn.hide()
        self.pm_person_7_delete_btn.hide()
        self.pm_person_8.hide()
        self.pm_person_8_detail_text.hide()
        self.pm_person_8_detail_value.hide()
        self.pm_person_8_face.hide()
        self.pm_person_8_edit_btn.hide()
        self.pm_person_8_delete_btn.hide()
        self.pm_person_9.hide()
        self.pm_person_9_detail_text.hide()
        self.pm_person_9_detail_value.hide()
        self.pm_person_9_face.hide()
        self.pm_person_9_edit_btn.hide()
        self.pm_person_9_delete_btn.hide()
        self.pm_person_10.hide()
        self.pm_person_10_detail_text.hide()
        self.pm_person_10_detail_value.hide()
        self.pm_person_10_face.hide()
        self.pm_person_10_edit_btn.hide()
        self.pm_person_10_delete_btn.hide()
        self.pm_add_btn.hide()
        self.pm_import_btn.hide()
        self.pm_export_btn.hide()
        self.pm_prev_e_btn.hide()
        self.pm_prev_btn.hide()
        self.pm_next_e_btn.hide()
        self.pm_next_btn.hide()
        self.pm_page_num.hide()

    def show_person_history_view(self):
        self.ph_person_1.show()
        self.ph_person_1_detail_text.show()
        self.ph_person_1_detail_value.show()
        self.ph_person_1_face.show()
        self.ph_person_2.show()
        self.ph_person_2_detail_text.show()
        self.ph_person_2_detail_value.show()
        self.ph_person_2_face.show()
        self.ph_person_3.show()
        self.ph_person_3_detail_text.show()
        self.ph_person_3_detail_value.show()
        self.ph_person_3_face.show()
        self.ph_person_4.show()
        self.ph_person_4_detail_text.show()
        self.ph_person_4_detail_value.show()
        self.ph_person_4_face.show()
        self.ph_person_5.show()
        self.ph_person_5_detail_text.show()
        self.ph_person_5_detail_value.show()
        self.ph_person_5_face.show()
        self.ph_person_6.show()
        self.ph_person_6_detail_text.show()
        self.ph_person_6_detail_value.show()
        self.ph_person_6_face.show()
        self.ph_person_7.show()
        self.ph_person_7_detail_text.show()
        self.ph_person_7_detail_value.show()
        self.ph_person_7_face.show()
        self.ph_person_8.show()
        self.ph_person_8_detail_text.show()
        self.ph_person_8_detail_value.show()
        self.ph_person_8_face.show()
        self.ph_person_9.show()
        self.ph_person_9_detail_text.show()
        self.ph_person_9_detail_value.show()
        self.ph_person_9_face.show()
        self.ph_person_10.show()
        self.ph_person_10_detail_text.show()
        self.ph_person_10_detail_value.show()
        self.ph_person_10_face.show()
        self.ph_prev_e_btn.show()
        self.ph_prev_btn.show()
        self.ph_next_e_btn.show()
        self.ph_next_btn.show()
        self.ph_page_num.show()
        self.ph_clearall_btn.show()

        self.update_ph_show_persons()

    def hide_person_history_view(self):
        self.ph_person_1.hide()
        self.ph_person_1_detail_text.hide()
        self.ph_person_1_detail_value.hide()
        self.ph_person_1_face.hide()
        self.ph_person_2.hide()
        self.ph_person_2_detail_text.hide()
        self.ph_person_2_detail_value.hide()
        self.ph_person_2_face.hide()
        self.ph_person_3.hide()
        self.ph_person_3_detail_text.hide()
        self.ph_person_3_detail_value.hide()
        self.ph_person_3_face.hide()
        self.ph_person_4.hide()
        self.ph_person_4_detail_text.hide()
        self.ph_person_4_detail_value.hide()
        self.ph_person_4_face.hide()
        self.ph_person_5.hide()
        self.ph_person_5_detail_text.hide()
        self.ph_person_5_detail_value.hide()
        self.ph_person_5_face.hide()
        self.ph_person_6.hide()
        self.ph_person_6_detail_text.hide()
        self.ph_person_6_detail_value.hide()
        self.ph_person_6_face.hide()
        self.ph_person_7.hide()
        self.ph_person_7_detail_text.hide()
        self.ph_person_7_detail_value.hide()
        self.ph_person_7_face.hide()
        self.ph_person_8.hide()
        self.ph_person_8_detail_text.hide()
        self.ph_person_8_detail_value.hide()
        self.ph_person_8_face.hide()
        self.ph_person_9.hide()
        self.ph_person_9_detail_text.hide()
        self.ph_person_9_detail_value.hide()
        self.ph_person_9_face.hide()
        self.ph_person_10.hide()
        self.ph_person_10_detail_text.hide()
        self.ph_person_10_detail_value.hide()
        self.ph_person_10_face.hide()
        self.ph_prev_e_btn.hide()
        self.ph_prev_btn.hide()
        self.ph_next_e_btn.hide()
        self.ph_next_btn.hide()
        self.ph_page_num.hide()
        self.ph_clearall_btn.hide()

    def pe_draw_details(self, data):
        self.pe_name_value.setText(data[1])
        self.pe_birthday_value.setText(data[2])
        self.pe_age_value.setText(data[3])
        self.pe_status_value.setText(data[4])
        draw_img1 = data[5].copy()
        resized1 = cv2.resize(draw_img1, (int(130 * X_SCALE), int(200 * Y_SCALE)))
        converted1 = cv2.cvtColor(resized1, cv2.COLOR_BGR2RGB)
        im1 = QtGui.QImage(converted1.data, converted1.shape[1], converted1.shape[0], QtGui.QImage.Format_RGB888)
        pix1 = QtGui.QPixmap.fromImage(im1)
        self.pe_person_face.setPixmap(pix1)
        self.pe_face_image = data[5].copy()

        self.pe_gender.setCurrentText(data[7])
        self.pe_guest_type.setCurrentText(data[8])
        self.pe_safety_type.setCurrentText(data[9])
        self.pe_blocked.setCurrentText(data[10])
        self.pe_when_from.setText(data[11])
        self.pe_when_to.setText(data[12])
        self.pe_where.setText(data[13])
        self.pe_reason.setText(data[14])
        self.pe_local_type.setCurrentText(data[15])
        self.pe_info_text.setPlainText(data[16])

        self.show_pe_characteristic_setting_view()


        try:
            pix = QtGui.QPixmap(f'database/agreements/{data[1]}.jpg')
            self.pe_agree_text.setPixmap(pix)
        except:
            pix = QtGui.QPixmap('database/agreements/sample.jpg')
            self.pe_agree_text.setPixmap(pix)

    def show_person_edit_view(self):
        if self.pe_person_id == 1:
            data = self.person_register_list[0]
            self.pe_draw_details(data)
        elif self.pe_person_id == 2:
            data = self.person_register_list[1]
            self.pe_draw_details(data)
        elif self.pe_person_id == 3:
            data = self.person_register_list[2]
            self.pe_draw_details(data)
        elif self.pe_person_id == 4:
            data = self.person_register_list[3]
            self.pe_draw_details(data)
        elif self.pe_person_id == 5:
            data = self.person_register_list[4]
            self.pe_draw_details(data)
        elif self.pe_person_id == 6:
            data = self.person_register_list[5]
            self.pe_draw_details(data)
        elif self.pe_person_id == 7:
            data = self.person_register_list[6]
            self.pe_draw_details(data)
        elif self.pe_person_id == 8:
            data = self.person_register_list[7]
            self.pe_draw_details(data)
        elif self.pe_person_id == 9:
            data = self.person_register_list[8]
            self.pe_draw_details(data)
        elif self.pe_person_id == 10:
            data = self.person_register_list[9]
            self.pe_draw_details(data)
        else:
            self.pe_name_value.clear()
            self.pe_birthday_value.clear()
            self.pe_age_value.clear()
            self.pe_status_value.clear()
            self.pe_gov_message_value.clear()
            pix = QtGui.QPixmap("./icons/face_icon.png")
            self.pe_person_face.setPixmap(pix)

            self.show_pe_characteristic_setting_view()

            pix = QtGui.QPixmap('database/agreements/sample.jpg')
            self.pe_agree_text.setPixmap(pix)

        self.pe_person_face.show()
        self.pe_from_camera_btn.show()
        self.pe_from_image_btn.show()
        self.pe_read_doc_btn.show()
        self.pe_sign_btn.show()
        self.pe_edit_btn.show()
        self.pe_text1.show()
        self.pe_name_value.show()
        self.pe_text2.show()
        self.pe_birthday_value.show()
        self.pe_text3.show()
        self.pe_age_value.show()
        self.pe_text4.show()
        self.pe_status_value.show()
        self.pe_gov_text.show()
        self.pe_gov_message_value.show()
        self.pe_oasis_btn.show()
        self.pe_save_btn.show()
        self.pe_cancel_btn.show()

        self.pe_setting_btn.show()
        self.pe_agreement_btn.show()
        self.pe_label.show()

    def show_pe_characteristic_setting_view(self):
        self.pe_text5.show()
        self.pe_gender.show()
        self.pe_text6.show()
        self.pe_guest_type.show()
        self.pe_text7.show()
        self.pe_safety_type.show()
        self.pe_text8.show()
        self.pe_info_text.show()
        self.pe_text9.show()
        self.pe_blocked.show()
        self.pe_text10.show()
        self.pe_when_from.show()
        self.pe_text11.show()
        self.pe_when_to.show()
        self.pe_text12.show()
        self.pe_where.show()
        self.pe_text13.show()
        self.pe_reason.show()
        self.pe_text14.show()
        self.pe_local_type.show()

    def hide_pe_characteristic_setting_view(self):
        self.pe_text5.hide()
        self.pe_gender.hide()
        self.pe_text6.hide()
        self.pe_guest_type.hide()
        self.pe_text7.hide()
        self.pe_safety_type.hide()
        self.pe_text8.hide()
        self.pe_info_text.hide()
        self.pe_text9.hide()
        self.pe_blocked.hide()
        self.pe_text10.hide()
        self.pe_when_from.hide()
        self.pe_text11.hide()
        self.pe_when_to.hide()
        self.pe_text12.hide()
        self.pe_where.hide()
        self.pe_text13.hide()
        self.pe_reason.hide()
        self.pe_text14.hide()
        self.pe_local_type.hide()

    def show_pe_agreement_view(self):
        self.pe_agree_text.show()
        self.pe_agree_print_btn.show()

    def hide_pe_agreement_view(self):
        self.pe_agree_text.hide()
        self.pe_agree_print_btn.hide()

    def hide_person_edit_view(self):
        self.pe_person_face.hide()
        self.pe_from_camera_btn.hide()
        self.pe_from_image_btn.hide()
        self.pe_read_doc_btn.hide()
        self.pe_sign_btn.hide()
        self.pe_edit_btn.hide()
        self.pe_text1.hide()
        self.pe_name_value.hide()
        self.pe_text2.hide()
        self.pe_birthday_value.hide()
        self.pe_text3.hide()
        self.pe_age_value.hide()
        self.pe_text4.hide()
        self.pe_status_value.hide()
        self.pe_gov_text.hide()
        self.pe_gov_message_value.hide()
        self.pe_oasis_btn.hide()
        self.pe_save_btn.hide()
        self.pe_cancel_btn.hide()

        self.pe_setting_btn.hide()
        self.pe_agreement_btn.hide()
        self.pe_label.hide()

        self.hide_pe_characteristic_setting_view()
        self.hide_pe_agreement_view()

        self.pe_setting_btn.setStyleSheet(L_BT_TAB1)
        self.pe_agreement_btn.setStyleSheet(L_BT_TAB2)

    def show_capture_view(self):
        self.cp_frame_label.clear()
        self.cp_bg_label.show()
        self.cp_frame_label.show()
        self.cp_camera_icon.show()
        self.cp_capture_btn.show()
        self.cp_cancel_btn.show()

    def hide_capture_view(self):
        self.cp_bg_label.hide()
        self.cp_frame_label.hide()
        self.cp_camera_icon.hide()
        self.cp_capture_btn.hide()
        self.cp_cancel_btn.hide()

    def show_sign_view(self):
        self.sign_frame_label.clear()
        self.sign_canvas.fill(QtCore.Qt.white)
        self.sign_bg_label.show()
        self.sign_frame_label.show()
        self.sign_clear_btn.show()
        self.sign_capture_btn.show()
        self.sign_cancel_btn.show()

    def hide_sign_view(self):
        self.sign_bg_label.hide()
        self.sign_frame_label.hide()
        self.sign_clear_btn.hide()
        self.sign_capture_btn.hide()
        self.sign_cancel_btn.hide()

    def show_user_manage_view(self):
        self.um_user_1.show()
        self.um_user_1_detail_text.show()
        self.um_user_1_detail_value.show()
        self.um_user_1_face.show()
        self.um_user_1_edit_btn.show()
        self.um_user_1_delete_btn.show()
        self.um_user_2.show()
        self.um_user_2_detail_text.show()
        self.um_user_2_detail_value.show()
        self.um_user_2_face.show()
        self.um_user_2_edit_btn.show()
        self.um_user_2_delete_btn.show()
        self.um_user_3.show()
        self.um_user_3_detail_text.show()
        self.um_user_3_detail_value.show()
        self.um_user_3_face.show()
        self.um_user_3_edit_btn.show()
        self.um_user_3_delete_btn.show()
        self.um_user_4.show()
        self.um_user_4_detail_text.show()
        self.um_user_4_detail_value.show()
        self.um_user_4_face.show()
        self.um_user_4_edit_btn.show()
        self.um_user_4_delete_btn.show()
        self.um_user_5.show()
        self.um_user_5_detail_text.show()
        self.um_user_5_detail_value.show()
        self.um_user_5_face.show()
        self.um_user_5_edit_btn.show()
        self.um_user_5_delete_btn.show()
        self.um_user_6.show()
        self.um_user_6_detail_text.show()
        self.um_user_6_detail_value.show()
        self.um_user_6_face.show()
        self.um_user_6_edit_btn.show()
        self.um_user_6_delete_btn.show()
        self.um_user_7.show()
        self.um_user_7_detail_text.show()
        self.um_user_7_detail_value.show()
        self.um_user_7_face.show()
        self.um_user_7_edit_btn.show()
        self.um_user_7_delete_btn.show()
        self.um_user_8.show()
        self.um_user_8_detail_text.show()
        self.um_user_8_detail_value.show()
        self.um_user_8_face.show()
        self.um_user_8_edit_btn.show()
        self.um_user_8_delete_btn.show()
        self.um_user_9.show()
        self.um_user_9_detail_text.show()
        self.um_user_9_detail_value.show()
        self.um_user_9_face.show()
        self.um_user_9_edit_btn.show()
        self.um_user_9_delete_btn.show()
        self.um_user_10.show()
        self.um_user_10_detail_text.show()
        self.um_user_10_detail_value.show()
        self.um_user_10_face.show()
        self.um_user_10_edit_btn.show()
        self.um_user_10_delete_btn.show()
        self.um_add_btn.show()
        self.um_prev_e_btn.show()
        self.um_prev_btn.show()
        self.um_next_e_btn.show()
        self.um_next_btn.show()
        self.um_page_num.show()

        self.update_um_show_users()

    def hide_user_manage_view(self):
        self.um_user_1.hide()
        self.um_user_1_detail_text.hide()
        self.um_user_1_detail_value.hide()
        self.um_user_1_face.hide()
        self.um_user_1_edit_btn.hide()
        self.um_user_1_delete_btn.hide()
        self.um_user_2.hide()
        self.um_user_2_detail_text.hide()
        self.um_user_2_detail_value.hide()
        self.um_user_2_face.hide()
        self.um_user_2_edit_btn.hide()
        self.um_user_2_delete_btn.hide()
        self.um_user_3.hide()
        self.um_user_3_detail_text.hide()
        self.um_user_3_detail_value.hide()
        self.um_user_3_face.hide()
        self.um_user_3_edit_btn.hide()
        self.um_user_3_delete_btn.hide()
        self.um_user_4.hide()
        self.um_user_4_detail_text.hide()
        self.um_user_4_detail_value.hide()
        self.um_user_4_face.hide()
        self.um_user_4_edit_btn.hide()
        self.um_user_4_delete_btn.hide()
        self.um_user_5.hide()
        self.um_user_5_detail_text.hide()
        self.um_user_5_detail_value.hide()
        self.um_user_5_face.hide()
        self.um_user_5_edit_btn.hide()
        self.um_user_5_delete_btn.hide()
        self.um_user_6.hide()
        self.um_user_6_detail_text.hide()
        self.um_user_6_detail_value.hide()
        self.um_user_6_face.hide()
        self.um_user_6_edit_btn.hide()
        self.um_user_6_delete_btn.hide()
        self.um_user_7.hide()
        self.um_user_7_detail_text.hide()
        self.um_user_7_detail_value.hide()
        self.um_user_7_face.hide()
        self.um_user_7_edit_btn.hide()
        self.um_user_7_delete_btn.hide()
        self.um_user_8.hide()
        self.um_user_8_detail_text.hide()
        self.um_user_8_detail_value.hide()
        self.um_user_8_face.hide()
        self.um_user_8_edit_btn.hide()
        self.um_user_8_delete_btn.hide()
        self.um_user_9.hide()
        self.um_user_9_detail_text.hide()
        self.um_user_9_detail_value.hide()
        self.um_user_9_face.hide()
        self.um_user_9_edit_btn.hide()
        self.um_user_9_delete_btn.hide()
        self.um_user_10.hide()
        self.um_user_10_detail_text.hide()
        self.um_user_10_detail_value.hide()
        self.um_user_10_face.hide()
        self.um_user_10_edit_btn.hide()
        self.um_user_10_delete_btn.hide()
        self.um_add_btn.hide()
        self.um_prev_e_btn.hide()
        self.um_prev_btn.hide()
        self.um_next_e_btn.hide()
        self.um_next_btn.hide()
        self.um_page_num.hide()

    def ue_draw_details(self, data):
        self.ue_usrname_value.setText(data[1])
        self.ue_usrgroup_value.setCurrentText(data[2])
        self.ue_pass_value.setText(data[3])
        self.ue_passconfirm_value.setText(data[3])
        self.ue_creator_value.setText(data[4])
        self.ue_phone_value.setText(data[5])
        self.ue_blocked_value.setCurrentText(data[6])

    def show_user_edit_view(self):
        if self.ue_user_id == 1:
            data = self.user_register_list[0]
            self.ue_draw_details(data)
        elif self.ue_user_id == 2:
            data = self.user_register_list[1]
            self.ue_draw_details(data)
        elif self.ue_user_id == 3:
            data = self.user_register_list[2]
            self.ue_draw_details(data)
        elif self.ue_user_id == 4:
            data = self.user_register_list[3]
            self.ue_draw_details(data)
        elif self.ue_user_id == 5:
            data = self.user_register_list[4]
            self.ue_draw_details(data)
        elif self.ue_user_id == 6:
            data = self.user_register_list[5]
            self.ue_draw_details(data)
        elif self.ue_user_id == 7:
            data = self.user_register_list[6]
            self.ue_draw_details(data)
        elif self.ue_user_id == 8:
            data = self.user_register_list[7]
            self.ue_draw_details(data)
        elif self.ue_user_id == 9:
            data = self.user_register_list[8]
            self.ue_draw_details(data)
        elif self.ue_user_id == 10:
            data = self.user_register_list[9]
            self.ue_draw_details(data)
        else:
            self.ue_usrname_value.clear()
            self.ue_pass_value.clear()
            self.ue_passconfirm_value.clear()
            self.ue_phone_value.clear()
            self.ue_creator_value.clear()

        self.ue_text1.show()
        self.ue_label1.show()
        self.ue_text2.show()
        self.ue_usrname_value.show()
        self.ue_text3.show()
        self.ue_usrgroup_value.show()
        self.ue_text4.show()
        self.ue_pass_value.show()
        self.ue_text6.show()
        self.ue_passconfirm_value.show()
        self.ue_text7.show()
        self.ue_phone_value.show()
        self.ue_text8.show()
        self.ue_creator_value.show()
        self.ue_text9.show()
        self.ue_blocked_value.show()
        self.ue_save_btn.show()
        self.ue_cancel_btn.show()

    def hide_user_edit_view(self):
        self.ue_text1.hide()
        self.ue_label1.hide()
        self.ue_text2.hide()
        self.ue_usrname_value.hide()
        self.ue_text3.hide()
        self.ue_usrgroup_value.hide()
        self.ue_text4.hide()
        self.ue_pass_value.hide()
        self.ue_text6.hide()
        self.ue_passconfirm_value.hide()
        self.ue_text7.hide()
        self.ue_phone_value.hide()
        self.ue_text8.hide()
        self.ue_creator_value.hide()
        self.ue_text9.hide()
        self.ue_blocked_value.hide()
        self.ue_save_btn.hide()
        self.ue_cancel_btn.hide()

    def show_user_history_view(self):
        self.uh_user_1.show()
        self.uh_user_1_detail_text.show()
        self.uh_user_1_detail_value.show()
        self.uh_user_1_face.show()
        self.uh_user_2.show()
        self.uh_user_2_detail_text.show()
        self.uh_user_2_detail_value.show()
        self.uh_user_2_face.show()
        self.uh_user_3.show()
        self.uh_user_3_detail_text.show()
        self.uh_user_3_detail_value.show()
        self.uh_user_3_face.show()
        self.uh_user_4.show()
        self.uh_user_4_detail_text.show()
        self.uh_user_4_detail_value.show()
        self.uh_user_4_face.show()
        self.uh_user_5.show()
        self.uh_user_5_detail_text.show()
        self.uh_user_5_detail_value.show()
        self.uh_user_5_face.show()
        self.uh_user_6.show()
        self.uh_user_6_detail_text.show()
        self.uh_user_6_detail_value.show()
        self.uh_user_6_face.show()
        self.uh_user_7.show()
        self.uh_user_7_detail_text.show()
        self.uh_user_7_detail_value.show()
        self.uh_user_7_face.show()
        self.uh_user_8.show()
        self.uh_user_8_detail_text.show()
        self.uh_user_8_detail_value.show()
        self.uh_user_8_face.show()
        self.uh_user_9.show()
        self.uh_user_9_detail_text.show()
        self.uh_user_9_detail_value.show()
        self.uh_user_9_face.show()
        self.uh_user_10.show()
        self.uh_user_10_detail_text.show()
        self.uh_user_10_detail_value.show()
        self.uh_user_10_face.show()
        self.uh_prev_e_btn.show()
        self.uh_prev_btn.show()
        self.uh_next_e_btn.show()
        self.uh_next_btn.show()
        self.uh_page_num.show()
        self.uh_clearall_btn.show()
        
        self.update_uh_show_users()

    def hide_user_history_view(self):
        self.uh_user_1.hide()
        self.uh_user_1_detail_text.hide()
        self.uh_user_1_detail_value.hide()
        self.uh_user_1_face.hide()
        self.uh_user_2.hide()
        self.uh_user_2_detail_text.hide()
        self.uh_user_2_detail_value.hide()
        self.uh_user_2_face.hide()
        self.uh_user_3.hide()
        self.uh_user_3_detail_text.hide()
        self.uh_user_3_detail_value.hide()
        self.uh_user_3_face.hide()
        self.uh_user_4.hide()
        self.uh_user_4_detail_text.hide()
        self.uh_user_4_detail_value.hide()
        self.uh_user_4_face.hide()
        self.uh_user_5.hide()
        self.uh_user_5_detail_text.hide()
        self.uh_user_5_detail_value.hide()
        self.uh_user_5_face.hide()
        self.uh_user_6.hide()
        self.uh_user_6_detail_text.hide()
        self.uh_user_6_detail_value.hide()
        self.uh_user_6_face.hide()
        self.uh_user_7.hide()
        self.uh_user_7_detail_text.hide()
        self.uh_user_7_detail_value.hide()
        self.uh_user_7_face.hide()
        self.uh_user_8.hide()
        self.uh_user_8_detail_text.hide()
        self.uh_user_8_detail_value.hide()
        self.uh_user_8_face.hide()
        self.uh_user_9.hide()
        self.uh_user_9_detail_text.hide()
        self.uh_user_9_detail_value.hide()
        self.uh_user_9_face.hide()
        self.uh_user_10.hide()
        self.uh_user_10_detail_text.hide()
        self.uh_user_10_detail_value.hide()
        self.uh_user_10_face.hide()
        self.uh_prev_e_btn.hide()
        self.uh_prev_btn.hide()
        self.uh_next_e_btn.hide()
        self.uh_next_btn.hide()
        self.uh_page_num.hide()
        self.uh_clearall_btn.hide()

    def show_setting_view(self):
        self.st_text1.show()
        self.st_text2.show()
        self.st_text3.show()
        self.st_text4.show()
        self.st_text5.show()
        self.st_text6.show()
        self.st_text7.show()
        self.st_text8.show()
        self.st_text9.show()
        self.st_text10.show()
        self.st_label1.show()
        self.st_lang.show()
        self.st_place_value.show()
        self.st_time_value.show()
        self.st_mode.show()
        self.st_label2.show()
        self.st_address_value.show()
        self.st_usrname_value.show()
        self.st_pass.show()
        self.st_default_btn.show()
        self.st_apply_btn.show()

    def hide_setting_view(self):
        self.st_text1.hide()
        self.st_text2.hide()
        self.st_text3.hide()
        self.st_text4.hide()
        self.st_text5.hide()
        self.st_text6.hide()
        self.st_text7.hide()
        self.st_text8.hide()
        self.st_text9.hide()
        self.st_text10.hide()
        self.st_label1.hide()
        self.st_lang.hide()
        self.st_place_value.hide()
        self.st_time_value.hide()
        self.st_mode.hide()
        self.st_label2.hide()
        self.st_address_value.hide()
        self.st_usrname_value.hide()
        self.st_pass.hide()
        self.st_default_btn.hide()
        self.st_apply_btn.hide()

    def deactive_btns(self):
        self.monitor_btn.setIcon(QtGui.QIcon("./icons/monitor.png"))
        self.perman_btn.setIcon(QtGui.QIcon("./icons/person.png"))
        self.perhis_btn.setIcon(QtGui.QIcon("./icons/history.png"))
        self.usrman_btn.setIcon(QtGui.QIcon("./icons/person.png"))
        self.usrhis_btn.setIcon(QtGui.QIcon("./icons/history.png"))
        self.setting_btn.setIcon(QtGui.QIcon("./icons/setting.png"))

        self.monitor_btn.setStyleSheet(L_BT_N)
        self.perman_btn.setStyleSheet(L_BT_N)
        self.perhis_btn.setStyleSheet(L_BT_N)
        self.usrman_btn.setStyleSheet(L_BT_N)
        self.usrhis_btn.setStyleSheet(L_BT_N)
        self.setting_btn.setStyleSheet(L_BT_N)

    def hide_all_views(self):
        self.hide_montor_view()
        self.hide_person_manage_view()
        self.hide_person_edit_view()
        self.hide_person_history_view()
        self.hide_user_manage_view()
        self.hide_user_edit_view()
        self.hide_user_history_view()
        self.hide_setting_view()

    def monitor_btn_clicked(self):
        self.deactive_btns()
        self.monitor_btn.setStyleSheet(L_BT_P)
        self.monitor_btn.setIcon(QtGui.QIcon("./icons/monitor_b.png"))

        self.hide_all_views()
        self.show_monitor_view()

        self.monitor_view_flag = True

    def perman_btn_clicked(self):
        self.deactive_btns()
        self.perman_btn.setStyleSheet(L_BT_P)
        self.perman_btn.setIcon(QtGui.QIcon("./icons/person_b.png"))

        self.hide_all_views()
        self.show_person_manage_view()

        self.monitor_view_flag = False

    def perhis_btn_clicked(self):
        self.deactive_btns()
        self.perhis_btn.setStyleSheet(L_BT_P)
        self.perhis_btn.setIcon(QtGui.QIcon("./icons/history_b.png"))

        self.hide_all_views()
        self.show_person_history_view()

        self.monitor_view_flag = False

    def usrman_btn_clicked(self):
        self.deactive_btns()
        self.usrman_btn.setStyleSheet(L_BT_P)
        self.usrman_btn.setIcon(QtGui.QIcon("./icons/person_b.png"))

        self.hide_all_views()
        self.show_user_manage_view()

        self.monitor_view_flag = False

    def usrhis_btn_clicked(self):
        self.deactive_btns()
        self.usrhis_btn.setStyleSheet(L_BT_P)
        self.usrhis_btn.setIcon(QtGui.QIcon("./icons/history_b.png"))

        self.hide_all_views()
        self.show_user_history_view()

        self.monitor_view_flag = False

    def setting_btn_clicked(self):
        self.deactive_btns()
        self.setting_btn.setStyleSheet(L_BT_P)
        self.setting_btn.setIcon(QtGui.QIcon("./icons/setting_b.png"))

        self.hide_all_views()
        self.show_setting_view()

        self.monitor_view_flag = False

    def exit_btn_clicked(self):
        self.close()

    def pm_add_btn_clicked(self):
        self.hide_person_manage_view()
        self.pe_person_id = None
        self.show_person_edit_view()

    def pm_person_1_edit_btn_clicked(self):
        self.hide_person_manage_view()
        self.pe_person_id = 1
        self.show_person_edit_view()

    def pm_person_2_edit_btn_clicked(self):
        self.hide_person_manage_view()
        self.pe_person_id = 2
        self.show_person_edit_view()

    def pm_person_3_edit_btn_clicked(self):
        self.hide_person_manage_view()
        self.pe_person_id = 3
        self.show_person_edit_view()

    def pm_person_4_edit_btn_clicked(self):
        self.hide_person_manage_view()
        self.pe_person_id = 4
        self.show_person_edit_view()

    def pm_person_5_edit_btn_clicked(self):
        self.hide_person_manage_view()
        self.pe_person_id = 5
        self.show_person_edit_view()

    def pm_person_6_edit_btn_clicked(self):
        self.hide_person_manage_view()
        self.pe_person_id = 6
        self.show_person_edit_view()

    def pm_person_7_edit_btn_clicked(self):
        self.hide_person_manage_view()
        self.pe_person_id = 7
        self.show_person_edit_view()

    def pm_person_8_edit_btn_clicked(self):
        self.hide_person_manage_view()
        self.pe_person_id = 8
        self.show_person_edit_view()

    def pm_person_9_edit_btn_clicked(self):
        self.hide_person_manage_view()
        self.pe_person_id = 9
        self.show_person_edit_view()

    def pm_person_10_edit_btn_clicked(self):
        self.hide_person_manage_view()
        self.pe_person_id = 10
        self.show_person_edit_view()

    def pm_delete_registered_person(self, idx):
        # ----  save person to db --------
        try:
            cursor = self.db_conn.cursor()
            sqlite_insert_blob_query = """ DELETE from person_register_table WHERE id=?"""

            cursor.execute(sqlite_insert_blob_query, (idx, ))
            self.db_conn.commit()
            print("Delete one registered person successfully.")
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to delete one registered person.", error)

    def pm_person_1_delete_btn_clicked(self):
        self.pe_person_id = 1
        index = self.person_register_list[self.pe_person_id - 1][0]
        self.pm_delete_registered_person(index)
        self.person_register_list.pop(self.pe_person_id - 1)
        self.update_pm_show_persons()

    def pm_person_2_delete_btn_clicked(self):
        self.pe_person_id = 2
        index = self.person_register_list[self.pe_person_id - 1][0]
        self.pm_delete_registered_person(index)
        self.person_register_list.pop(self.pe_person_id - 1)
        self.update_pm_show_persons()

    def pm_person_3_delete_btn_clicked(self):
        self.pe_person_id = 3
        index = self.person_register_list[self.pe_person_id - 1][0]
        self.pm_delete_registered_person(index)
        self.person_register_list.pop(self.pe_person_id - 1)
        self.update_pm_show_persons()

    def pm_person_4_delete_btn_clicked(self):
        self.pe_person_id = 4
        index = self.person_register_list[self.pe_person_id - 1][0]
        self.pm_delete_registered_person(index)
        self.person_register_list.pop(self.pe_person_id - 1)
        self.update_pm_show_persons()

    def pm_person_5_delete_btn_clicked(self):
        self.pe_person_id = 5
        index = self.person_register_list[self.pe_person_id - 1][0]
        self.pm_delete_registered_person(index)
        self.person_register_list.pop(self.pe_person_id - 1)
        self.update_pm_show_persons()

    def pm_person_6_delete_btn_clicked(self):
        self.pe_person_id = 6
        index = self.person_register_list[self.pe_person_id - 1][0]
        self.pm_delete_registered_person(index)
        self.person_register_list.pop(self.pe_person_id - 1)
        self.update_pm_show_persons()

    def pm_person_7_delete_btn_clicked(self):
        self.pe_person_id = 7
        index = self.person_register_list[self.pe_person_id - 1][0]
        self.pm_delete_registered_person(index)
        self.person_register_list.pop(self.pe_person_id - 1)
        self.update_pm_show_persons()

    def pm_person_8_delete_btn_clicked(self):
        self.pe_person_id = 8
        index = self.person_register_list[self.pe_person_id - 1][0]
        self.pm_delete_registered_person(index)
        self.person_register_list.pop(self.pe_person_id - 1)
        self.update_pm_show_persons()

    def pm_person_9_delete_btn_clicked(self):
        self.pe_person_id = 9
        index = self.person_register_list[self.pe_person_id - 1][0]
        self.pm_delete_registered_person(index)
        self.person_register_list.pop(self.pe_person_id - 1)
        self.update_pm_show_persons()

    def pm_person_10_delete_btn_clicked(self):
        self.pe_person_id = 10
        index = self.person_register_list[self.pe_person_id - 1][0]
        self.pm_delete_registered_person(index)
        self.person_register_list.pop(self.pe_person_id - 1)
        self.update_pm_show_persons()

    def pe_from_camera_btn_clicked(self):
        self.show_capture_view()

        self.face_capture_thread = Thread(target=self.face_capture)
        self.face_capture_thread.setDaemon(True)
        self.face_capture_thread.start()

    def pe_from_image_btn_clicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None, "Open Face Image", "",
                                                  "All files (*.jpg *.gif *.png)", options=options)
        try:
            self.pe_face_image = cv2.imread(fileName)
            draw_img1 = self.pe_face_image.copy()
            resized1 = cv2.resize(draw_img1, (int(130 * X_SCALE), int(200 * Y_SCALE)))
            converted1 = cv2.cvtColor(resized1, cv2.COLOR_BGR2RGB)
            im1 = QtGui.QImage(converted1.data, converted1.shape[1], converted1.shape[0], QtGui.QImage.Format_RGB888)
            pix1 = QtGui.QPixmap.fromImage(im1)
            self.pe_person_face.setPixmap(pix1)
        except:
            pass

    def pe_read_doc_btn_clicked(self):
        self.show_capture_view()

        self.mrz_read_thread = Thread(target=self.mrz_recog)
        self.mrz_read_thread.setDaemon(True)
        self.mrz_read_thread.start()

    def pe_sign_btn_clicked(self):
        self.sign_win = SignWin(self)
        self.sign_win.move(sign_monitor.left(), sign_monitor.top())
        self.sign_win.showFullScreen()
        ctypes.windll.user32.SetCursorPos(2100, 80)

    def pe_oasis_btn_clicked(self):
        try:
            name = self.pe_name_value.text()
            firstname = name.split(' ')[0]
            lastname = name.split(' ')[1]
            birth = self.pe_birthday_value.text()
            result = connect_gov('12', firstname, lastname, birth)

            self.pe_gov_message_value.setText(result)

            if result == "Der Spieler ist nicht gesperrt.":
                self.pe_status_value.setText("Allow")

            else:
                self.pe_status_value.setText("Not Allow")

            self.show_pe_characteristic_setting_view()
        except:
            pass

    def pe_save_btn_clicked(self):
        if self.pe_name_value.text() and \
                self.pe_birthday_value.text() and \
                self.pe_age_value.text() and \
                self.pe_status_value.text():

            if self.pe_person_id == None:
                self.pe_add_person()
            else:
                self.pe_modify_person()

            self.hide_person_edit_view()
            self.show_person_manage_view()

    def pe_add_person(self):
        name = self.pe_name_value.text()
        birth = self.pe_birthday_value.text()
        age = self.pe_age_value.text()
        status = self.pe_status_value.text()
        face = self.pe_face_image
        face_bytes = cv2.imencode('.jpg', face)[1].tobytes()
        res = self.face_recognizer.predict(face)
        emb = res[0].embedding
        emb_bytes = emb.tobytes()

        if status == "Not Allow":
            gender = None
            guesttype = None
            safetytype = None
            blocked = None
            whenfrom = None
            whento = None
            place = None
            reason = None
            type = None
            info = None
        else:
            gender = self.pe_gender.currentText()
            guesttype = self.pe_guest_type.currentText()
            safetytype = self.pe_safety_type.currentText()
            blocked = self.pe_blocked.currentText()
            whenfrom = self.pe_when_from.text()
            whento = self.pe_when_to.text()
            place = self.pe_where.text()
            reason = self.pe_reason.text()
            type = self.pe_local_type.currentText()
            info = self.pe_info_text.toPlainText()
        # ----  save person to db --------
        try:
            cursor = self.db_conn.cursor()
            sqlite_insert_blob_query = """ INSERT INTO person_register_table
            (name, birth, age, status, photo, emb, gender, guesttype, safetytype, blocked, whenfrom, whento, place, reason, type, info) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

            # Convert data into tuple format
            data_tuple = (name, birth, age, status, face_bytes, emb_bytes, gender, guesttype, safetytype, blocked, whenfrom, whento, place, reason, type, info)

            cursor.execute(sqlite_insert_blob_query, data_tuple)
            id = cursor.lastrowid
            self.db_conn.commit()
            print("Image and file inserted successfully as a BLOB into a table")
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to insert blob data into sqlite table", error)

        self.person_register_list.append([id, name, birth, age, status, face, emb, gender, guesttype, safetytype, blocked, whenfrom, whento, place, reason, type, info])

    def pe_modify_person(self):
        id = self.person_register_list[self.pe_person_id - 1][0]
        name = self.pe_name_value.text()
        birth = self.pe_birthday_value.text()
        age = self.pe_age_value.text()
        status = self.pe_status_value.text()
        face = self.pe_face_image
        face_bytes = cv2.imencode('.jpg', face)[1].tobytes()
        res = self.face_recognizer.predict(face)
        emb = res[0].embedding
        emb_bytes = emb.tobytes()
        gender = self.pe_gender.currentText()
        guesttype = self.pe_guest_type.currentText()
        safetytype = self.pe_safety_type.currentText()
        blocked = self.pe_blocked.currentText()
        whenfrom = self.pe_when_from.text()
        whento = self.pe_when_to.text()
        place = self.pe_where.text()
        reason = self.pe_reason.text()
        type = self.pe_local_type.currentText()
        info = self.pe_info_text.toPlainText()
        # ----  save person to db --------
        try:
            cursor = self.db_conn.cursor()
            sqlite_insert_blob_query = """ UPDATE person_register_table SET
                    name = ?, birth = ?, age = ?, status = ?, photo = ?, emb = ?, gender = ?, guesttype = ?, 
                    safetytype = ?, blocked = ?, whenfrom = ?, whento = ?, place = ?, reason = ?, type = ?, info = ? 
                    WHERE id = ?"""

            # Convert data into tuple format
            data_tuple = (name, birth, age, status, face_bytes, emb_bytes, gender, guesttype, safetytype, blocked,
                          whenfrom, whento, place, reason, type, info, id)

            cursor.execute(sqlite_insert_blob_query, data_tuple)
            self.db_conn.commit()
            print("Image and file inserted successfully as a BLOB into a table")
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to insert blob data into sqlite table", error)

        self.person_register_list[self.pe_person_id - 1] = [id, name, birth, age, status, face, emb, gender, guesttype, safetytype, blocked, whenfrom, whento, place, reason, type, info]

    def pe_cancel_btn_clicked(self):
        self.hide_person_edit_view()
        self.show_person_manage_view()

    def pe_setting_btn_clicked(self):
        self.pe_setting_btn.setStyleSheet(L_BT_TAB1)
        self.pe_agreement_btn.setStyleSheet(L_BT_TAB2)
        self.show_pe_characteristic_setting_view()
        self.hide_pe_agreement_view()

    def pe_agreement_btn_clicked(self):
        self.pe_setting_btn.setStyleSheet(L_BT_TAB2)
        self.pe_agreement_btn.setStyleSheet(L_BT_TAB1)
        self.hide_pe_characteristic_setting_view()
        self.show_pe_agreement_view()

    def mo_prev_e_btn_clicked(self, event):
        self.mo_page_iter = 1

        self.update_mo_show_persons()

    def mo_prev_btn_clicked(self, event):
        self.mo_page_iter -= 1
        if self.mo_page_iter < 1:
            self.mo_page_iter = 1

        self.update_mo_show_persons()

    def mo_next_btn_clicked(self, event):
        self.mo_page_iter += 1
        if self.mo_page_iter > self.mo_total_pages:
            self.mo_page_iter = self.mo_total_pages

        self.update_mo_show_persons()

    def mo_next_e_btn_clicked(self, event):
        self.mo_page_iter = self.mo_total_pages

        self.update_mo_show_persons()

    def mo_clearall_btn_clicked(self):
        self.now_person_history_list = []
        self.mo_page_iter = 1
        self.update_mo_show_persons()

    def pm_prev_e_btn_clicked(self, event):
        self.pm_page_iter = 1

        self.update_pm_show_persons()

    def pm_prev_btn_clicked(self, event):
        self.pm_page_iter -= 1
        if self.pm_page_iter < 1:
            self.pm_page_iter = 1

        self.update_pm_show_persons()

    def pm_next_btn_clicked(self, event):
        self.pm_page_iter += 1
        if self.pm_page_iter > self.pm_total_pages:
            self.pm_page_iter = self.pm_total_pages

        self.update_pm_show_persons()

    def pm_next_e_btn_clicked(self, event):
        self.pm_page_iter = self.pm_total_pages

        self.update_pm_show_persons()

    def ph_prev_e_btn_clicked(self, event):
        self.ph_page_iter = 1

        self.update_ph_show_persons()

    def ph_prev_btn_clicked(self, event):
        self.ph_page_iter -= 1
        if self.ph_page_iter < 1:
            self.ph_page_iter = 1

        self.update_ph_show_persons()

    def ph_next_btn_clicked(self, event):
        self.ph_page_iter += 1
        if self.ph_page_iter > self.ph_total_pages:
            self.ph_page_iter = self.ph_total_pages

        self.update_ph_show_persons()

    def ph_next_e_btn_clicked(self, event):
        self.ph_page_iter = self.ph_total_pages

        self.update_ph_show_persons()

    def ph_clearall_btn_clicked(self):
        self.person_history_list = []
        self.ph_page_iter = 1
        # ----  delete all person history from db --------
        try:
            cursor = self.db_conn.cursor()
            query = """ DELETE FROM person_history_table """
            cursor.execute(query)
            self.db_conn.commit()
            print("Delete all person history successfully.")
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to delete all person history.", error)

        self.update_ph_show_persons()

    def um_prev_e_btn_clicked(self, event):
        self.um_page_iter = 1

        self.update_um_show_users()

    def um_prev_btn_clicked(self, event):
        self.um_page_iter -= 1
        if self.um_page_iter < 1:
            self.um_page_iter = 1

        self.update_um_show_users()

    def um_next_btn_clicked(self, event):
        self.um_page_iter += 1
        if self.um_page_iter > self.um_total_pages:
            self.um_page_iter = self.um_total_pages

        self.update_um_show_users()

    def um_next_e_btn_clicked(self, event):
        self.um_page_iter = self.um_total_pages

        self.update_um_show_users()

    def uh_prev_e_btn_clicked(self, event):
        self.uh_page_iter = 1

        self.update_uh_show_users()

    def uh_prev_btn_clicked(self, event):
        self.uh_page_iter -= 1
        if self.uh_page_iter < 1:
            self.uh_page_iter = 1

        self.update_uh_show_users()

    def uh_next_btn_clicked(self, event):
        self.uh_page_iter += 1
        if self.uh_page_iter > self.uh_total_pages:
            self.uh_page_iter = self.uh_total_pages

        self.update_uh_show_users()

    def uh_next_e_btn_clicked(self, event):
        self.uh_page_iter = self.uh_total_pages

        self.update_uh_show_users()

    def uh_clearall_btn_clicked(self):
        self.user_history_list = []
        self.uh_page_iter = 1
        # ----  delete all user history from db --------
        try:
            cursor = self.db_conn.cursor()
            query = """ DELETE FROM user_history_table """
            cursor.execute(query)
            self.db_conn.commit()
            print("Delete all user history successfully.")
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to delete all user history.", error)

        self.update_uh_show_users()

    def um_add_btn_clicked(self):
        self.ue_user_id = None
        self.hide_user_manage_view()
        self.show_user_edit_view()

    def um_user_1_edit_btn_clicked(self):
        self.hide_user_manage_view()
        self.ue_user_id = 1
        self.show_user_edit_view()

    def um_user_2_edit_btn_clicked(self):
        self.hide_user_manage_view()
        self.ue_user_id = 2
        self.show_user_edit_view()

    def um_user_3_edit_btn_clicked(self):
        self.hide_user_manage_view()
        self.ue_user_id = 3
        self.show_user_edit_view()

    def um_user_4_edit_btn_clicked(self):
        self.hide_user_manage_view()
        self.ue_user_id = 4
        self.show_user_edit_view()

    def um_user_5_edit_btn_clicked(self):
        self.hide_user_manage_view()
        self.ue_user_id = 5
        self.show_user_edit_view()

    def um_user_6_edit_btn_clicked(self):
        self.hide_user_manage_view()
        self.ue_user_id = 6
        self.show_user_edit_view()

    def um_user_7_edit_btn_clicked(self):
        self.hide_user_manage_view()
        self.ue_user_id = 7
        self.show_user_edit_view()

    def um_user_8_edit_btn_clicked(self):
        self.hide_user_manage_view()
        self.ue_user_id = 8
        self.show_user_edit_view()

    def um_user_9_edit_btn_clicked(self):
        self.hide_user_manage_view()
        self.ue_user_id = 9
        self.show_user_edit_view()

    def um_user_10_edit_btn_clicked(self):
        self.hide_user_manage_view()
        self.ue_user_id = 10
        self.show_user_edit_view()

    def um_delete_registered_user(self, idx):
        # ----  save person to db --------
        try:
            cursor = self.db_conn.cursor()
            sqlite_insert_blob_query = """ DELETE from user_register_table WHERE id=?"""

            cursor.execute(sqlite_insert_blob_query, (idx, ))
            self.db_conn.commit()
            print("Delete one registered user successfully.")
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to delete one registered user.", error)

    def um_user_1_delete_btn_clicked(self):
        self.ue_user_id = 1
        index = self.user_register_list[self.ue_user_id - 1][0]
        self.um_delete_registered_user(index)
        self.user_register_list.pop(self.ue_user_id - 1)
        self.update_um_show_users()

    def um_user_2_delete_btn_clicked(self):
        self.ue_user_id = 2
        index = self.user_register_list[self.ue_user_id - 1][0]
        self.um_delete_registered_user(index)
        self.user_register_list.pop(self.ue_user_id - 1)
        self.update_um_show_users()

    def um_user_3_delete_btn_clicked(self):
        self.ue_user_id = 3
        index = self.user_register_list[self.ue_user_id - 1][0]
        self.um_delete_registered_user(index)
        self.user_register_list.pop(self.ue_user_id - 1)
        self.update_um_show_users()

    def um_user_4_delete_btn_clicked(self):
        self.ue_user_id = 4
        index = self.user_register_list[self.ue_user_id - 1][0]
        self.um_delete_registered_user(index)
        self.user_register_list.pop(self.ue_user_id - 1)
        self.update_um_show_users()

    def um_user_5_delete_btn_clicked(self):
        self.ue_user_id = 5
        index = self.user_register_list[self.ue_user_id - 1][0]
        self.um_delete_registered_user(index)
        self.user_register_list.pop(self.ue_user_id - 1)
        self.update_um_show_users()

    def um_user_6_delete_btn_clicked(self):
        self.ue_user_id = 6
        index = self.user_register_list[self.ue_user_id - 1][0]
        self.um_delete_registered_user(index)
        self.user_register_list.pop(self.ue_user_id - 1)
        self.update_um_show_users()

    def um_user_7_delete_btn_clicked(self):
        self.ue_user_id = 7
        index = self.user_register_list[self.ue_user_id - 1][0]
        self.um_delete_registered_user(index)
        self.user_register_list.pop(self.ue_user_id - 1)
        self.update_um_show_users()

    def um_user_8_delete_btn_clicked(self):
        self.ue_user_id = 8
        index = self.user_register_list[self.ue_user_id - 1][0]
        self.um_delete_registered_user(index)
        self.user_register_list.pop(self.ue_user_id - 1)
        self.update_um_show_users()

    def um_user_9_delete_btn_clicked(self):
        self.ue_user_id = 9
        index = self.user_register_list[self.ue_user_id - 1][0]
        self.um_delete_registered_user(index)
        self.user_register_list.pop(self.ue_user_id - 1)
        self.update_um_show_users()

    def um_user_10_delete_btn_clicked(self):
        self.ue_user_id = 10
        index = self.user_register_list[self.ue_user_id - 1][0]
        self.um_delete_registered_user(index)
        self.user_register_list.pop(self.ue_user_id - 1)
        self.update_um_show_users()

    def ue_save_btn_clicked(self):
        if self.ue_user_id == None:
            self.ue_add_user()
        else:
            self.ue_modify_user()
        self.hide_user_edit_view()
        self.show_user_manage_view()

    def ue_add_user(self):
        name = self.ue_usrname_value.text()
        usergroup = self.ue_usrgroup_value.currentText()
        password = self.ue_passconfirm_value.text()
        creator = self.ue_creator_value.text()
        phone = self.ue_phone_value.text()
        blocked = self.ue_blocked_value.currentText()

        # ----  save person to db --------
        try:
            cursor = self.db_conn.cursor()
            query = """ INSERT INTO user_register_table (name, usergroup, password, creator, phone, blocked) 
            VALUES (?, ?, ?, ?, ?, ?)"""

            # Convert data into tuple format
            data_tuple = (name, usergroup, password, creator, phone, blocked)

            cursor.execute(query, data_tuple)
            id = cursor.lastrowid
            self.db_conn.commit()
            print("Register new user successfully.")
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to add new user.", error)

        self.user_register_list.append([id, name, usergroup, password, creator, phone, blocked])

    def ue_modify_user(self):
        id = self.user_register_list[self.ue_user_id - 1][0]
        name = self.ue_usrname_value.text()
        usergroup = self.ue_usrgroup_value.currentText()
        password = self.ue_passconfirm_value.text()
        creator = self.ue_creator_value.text()
        phone = self.ue_phone_value.text()
        blocked = self.ue_blocked_value.currentText()

        # ----  save person to db --------
        try:
            cursor = self.db_conn.cursor()
            sqlite_insert_blob_query = """ UPDATE user_register_table SET
                    name = ?, usergroup = ?, password = ?, creator = ?, phone = ?, blocked = ? 
                    WHERE id = ?"""

            # Convert data into tuple format
            data_tuple = (name, usergroup, password, creator, phone, blocked, id)

            cursor.execute(sqlite_insert_blob_query, data_tuple)
            self.db_conn.commit()
            print("Modify user information")
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to modify user information", error)

        self.user_register_list[self.ue_user_id - 1] = [id, name, usergroup, password, creator, phone, blocked]

    def ue_cancel_btn_clicked(self):
        self.hide_user_edit_view()
        self.show_user_manage_view()

    def cp_capture_btn_clicked(self):
        self.capture_stop_flag = True
        self.hide_capture_view()

    def cp_cancel_btn_clicked(self):
        self.capture_stop_flag = True
        self.hide_capture_view()

    def sign_clear_btn_clicked(self):
        self.sign_frame_label.clear()
        self.sign_canvas.fill(QtCore.Qt.white)

    def sign_capture_btn_clicked(self):
        try:
            img = self.sign_canvas.toImage()
            img_name = f'database/agreements/{self.pe_name_value.text()}.png'
            img.save(img_name)

            img = Image.new("RGB", (200, 30), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            date_text = datetime.now().strftime("%Y-%m-%d")
            draw.text((10, 10), date_text, fill=(0, 0, 0), font_size=18)
            img.save('date.jpg')

            src_pdf_filename = 'database/agreements/sample.pdf'
            dst_pdf_filename = f'database/agreements/{self.pe_name_value.text()}.pdf'

            sign_img = open(img_name, "rb").read()
            date_img = open('date.jpg', "rb").read()

            date_rect = fitz.Rect(60, 570, 260, 600)
            sign_rect = fitz.Rect(50, 600, 300, 700)

            document = fitz.open(src_pdf_filename)
            page = document[0]
            if not page.is_wrapped:
                page.wrap_contents()

            page.insert_image(date_rect, stream=date_img)
            page.insert_image(sign_rect, stream=sign_img)

            document.save(dst_pdf_filename)
            document.close()
            os.remove(img_name)
            os.remove('date.jpg')

            print("Successfully made pdf file")

            doc = fitz.open(dst_pdf_filename)
            page = doc.load_page(0)  # number of page
            pix = page.get_pixmap()
            output = dst_pdf_filename.replace('pdf', 'jpg')
            pix.save(output)
            doc.close()

            pix = QtGui.QPixmap(output)
            self.pe_agree_text.setPixmap(pix)
        except:
            print("Error to make agreement pdf with sign.")

        self.hide_sign_view()

    def sign_cancel_btn_clicked(self):
        self.hide_sign_view()

    def login_login_btn_clicked(self):
        login_user = self.login_username_value.text()
        login_pass = self.login_pass_value.text()
        for user in self.user_register_list:
            pwd = user[3]
            usr = user[1]
            if login_pass == pwd and login_user == usr:
                self.hide_login_view()

                name = user[1]
                action = 'Log in'
                time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                content = ''

                # ----  save person to db --------
                try:
                    cursor = self.db_conn.cursor()
                    sqlite_insert_blob_query = """ INSERT INTO user_history_table
                                    (name, action, time, content) 
                                    VALUES (?, ?, ?, ?)"""

                    # Convert data into tuple format
                    data_tuple = (name, action, time, content)

                    cursor.execute(sqlite_insert_blob_query, data_tuple)
                    id = cursor.lastrowid
                    self.db_conn.commit()
                    print("Add new person history successfully.")
                    cursor.close()

                    self.user_history_list.append([id, name, action, time, content])

                except sqlite3.Error as error:
                    print("Failed to add new person history.", error)

                break

    def login_cancel_btn_clicked(self):
        self.hide_login_view()
        self.close()

    def mouse_move(self, e):
        if self.last_x is None:  # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return  # Ignore the first time.

        self.painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        # self.painter.end()
        self.sign_frame_label.setPixmap(self.sign_canvas)
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouse_release(self, e):
        self.last_x = None
        self.last_y = None

    def bytes_to_image(self, bytes):
        nparr = np.frombuffer(bytes, np.uint8)
        face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return face

    def bytes_to_emb(self, bytes):
        emb = np.frombuffer(bytes, np.float32)
        return emb

    def update_mo_show_persons(self):
        val1 = int(len(self.now_person_history_list) / 7)
        val2 = int(len(self.now_person_history_list) % 7)
        if val2 == 0:
            self.mo_total_pages = val1
        else:
            self.mo_total_pages = val1 + 1

        self.mo_page_num.setText(f"{self.mo_page_iter}/{self.mo_total_pages}")

        temp = self.now_person_history_list[(self.mo_page_iter - 1) * 7: (self.mo_page_iter) * 7]

        # temp = self.now_person_history_list[::-1]

        # ------------------
        try:
            data = temp[0]

            real_face = data[0]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.mo_real_face_1.setPixmap(pix)
            self.mo_real_face_1.show()

            saved_face = data[2][5]
            converted = cv2.cvtColor(saved_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im1 = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix1 = QtGui.QPixmap.fromImage(im1)
            self.mo_saved_face_1.setPixmap(pix1)
            self.mo_saved_face_1.show()

            text = f"Gender :{data[2][7]}\nStatus : {data[2][4]}\nBlocked : {data[2][10]}\nGuest : {data[2][8]}\nSafety : {data[2][9]}\n{data[1]}"
            self.mo_detail_1.setText(text)
            self.mo_detail_1.show()

            if data[2][4] != "Allow" or \
                    data[2][10] != "Nein" or \
                    data[2][9] == "Manipulator" or \
                    data[2][9] == "Manipulationsverdacht":
                pix = QtGui.QPixmap("./icons/danger.png")
                self.mo_icon_1.setPixmap(pix)
            else:
                if data[3] == "Come In":
                    pix = QtGui.QPixmap("./icons/come_in.png")
                    self.mo_icon_1.setPixmap(pix)
                if data[3] == "Come Out":
                    pix = QtGui.QPixmap("./icons/come_out.png")
                    self.mo_icon_1.setPixmap(pix)

            self.mo_icon_1.show()

        except:
            self.mo_real_face_1.hide()
            self.mo_saved_face_1.hide()
            self.mo_detail_1.hide()
            self.mo_icon_1.hide()

        # ------------------
        try:
            data = temp[1]

            real_face = data[0]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.mo_real_face_2.setPixmap(pix)
            self.mo_real_face_2.show()

            saved_face = data[2][5]
            converted = cv2.cvtColor(saved_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im1 = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix1 = QtGui.QPixmap.fromImage(im1)
            self.mo_saved_face_2.setPixmap(pix1)
            self.mo_saved_face_2.show()

            text = f"Gender :{data[2][7]}\nStatus : {data[2][4]}\nBlocked : {data[2][10]}\nGuest : {data[2][8]}\nSafety : {data[2][9]}\n{data[1]}"
            self.mo_detail_2.setText(text)
            self.mo_detail_2.show()

            if data[2][4] != "Allow" or \
                    data[2][10] != "Nein" or \
                    data[2][9] == "Manipulator" or \
                    data[2][9] == "Manipulationsverdacht":
                pix = QtGui.QPixmap("./icons/danger.png")
                self.mo_icon_2.setPixmap(pix)
            else:
                if data[3] == "Come In":
                    pix = QtGui.QPixmap("./icons/come_in.png")
                    self.mo_icon_2.setPixmap(pix)
                if data[3] == "Come Out":
                    pix = QtGui.QPixmap("./icons/come_out.png")
                    self.mo_icon_2.setPixmap(pix)

            self.mo_icon_2.show()

        except:
            self.mo_real_face_2.hide()
            self.mo_saved_face_2.hide()
            self.mo_detail_2.hide()
            self.mo_icon_2.hide()
        # ------------------
        try:
            data = temp[2]

            real_face = data[0]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.mo_real_face_3.setPixmap(pix)
            self.mo_real_face_3.show()

            saved_face = data[2][5]
            converted = cv2.cvtColor(saved_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im1 = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix1 = QtGui.QPixmap.fromImage(im1)
            self.mo_saved_face_3.setPixmap(pix1)
            self.mo_saved_face_3.show()

            text = f"Gender :{data[2][7]}\nStatus : {data[2][4]}\nBlocked : {data[2][10]}\nGuest : {data[2][8]}\nSafety : {data[2][9]}\n{data[1]}"
            self.mo_detail_3.setText(text)
            self.mo_detail_3.show()

            if data[2][4] != "Allow" or \
                    data[2][10] != "Nein" or \
                    data[2][9] == "Manipulator" or \
                    data[2][9] == "Manipulationsverdacht":
                pix = QtGui.QPixmap("./icons/danger.png")
                self.mo_icon_3.setPixmap(pix)
            else:
                if data[3] == "Come In":
                    pix = QtGui.QPixmap("./icons/come_in.png")
                    self.mo_icon_3.setPixmap(pix)
                if data[3] == "Come Out":
                    pix = QtGui.QPixmap("./icons/come_out.png")
                    self.mo_icon_3.setPixmap(pix)

            self.mo_icon_3.show()

        except:
            self.mo_real_face_3.hide()
            self.mo_saved_face_3.hide()
            self.mo_detail_3.hide()
            self.mo_icon_3.hide()

        # ------------------
        try:
            data = temp[3]

            real_face = data[0]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.mo_real_face_4.setPixmap(pix)
            self.mo_real_face_4.show()

            saved_face = data[2][5]
            converted = cv2.cvtColor(saved_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im1 = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix1 = QtGui.QPixmap.fromImage(im1)
            self.mo_saved_face_4.setPixmap(pix1)
            self.mo_saved_face_4.show()

            text = f"Gender :{data[2][7]}\nStatus : {data[2][4]}\nBlocked : {data[2][10]}\nGuest : {data[2][8]}\nSafety : {data[2][9]}\n{data[1]}"
            self.mo_detail_4.setText(text)
            self.mo_detail_4.show()

            if data[2][4] != "Allow" or \
                    data[2][10] != "Nein" or \
                    data[2][9] == "Manipulator" or \
                    data[2][9] == "Manipulationsverdacht":
                pix = QtGui.QPixmap("./icons/danger.png")
                self.mo_icon_4.setPixmap(pix)
            else:
                if data[3] == "Come In":
                    pix = QtGui.QPixmap("./icons/come_in.png")
                    self.mo_icon_4.setPixmap(pix)
                if data[3] == "Come Out":
                    pix = QtGui.QPixmap("./icons/come_out.png")
                    self.mo_icon_4.setPixmap(pix)

            self.mo_icon_4.show()

        except:
            self.mo_real_face_4.hide()
            self.mo_saved_face_4.hide()
            self.mo_detail_4.hide()
            self.mo_icon_4.hide()

        # ------------------
        try:
            data = temp[4]

            real_face = data[0]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.mo_real_face_5.setPixmap(pix)
            self.mo_real_face_5.show()

            saved_face = data[2][5]
            converted = cv2.cvtColor(saved_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im1 = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix1 = QtGui.QPixmap.fromImage(im1)
            self.mo_saved_face_5.setPixmap(pix1)
            self.mo_saved_face_5.show()

            text = f"Gender :{data[2][7]}\nStatus : {data[2][4]}\nBlocked : {data[2][10]}\nGuest : {data[2][8]}\nSafety : {data[2][9]}\n{data[1]}"
            self.mo_detail_5.setText(text)
            self.mo_detail_5.show()

            if data[2][4] != "Allow" or \
                    data[2][10] != "Nein" or \
                    data[2][9] == "Manipulator" or \
                    data[2][9] == "Manipulationsverdacht":
                pix = QtGui.QPixmap("./icons/danger.png")
                self.mo_icon_5.setPixmap(pix)
            else:
                if data[3] == "Come In":
                    pix = QtGui.QPixmap("./icons/come_in.png")
                    self.mo_icon_5.setPixmap(pix)
                if data[3] == "Come Out":
                    pix = QtGui.QPixmap("./icons/come_out.png")
                    self.mo_icon_5.setPixmap(pix)

            self.mo_icon_5.show()

        except:
            self.mo_real_face_5.hide()
            self.mo_saved_face_5.hide()
            self.mo_detail_5.hide()
            self.mo_icon_5.hide()

        # ------------------
        try:
            data = temp[5]

            real_face = data[0]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.mo_real_face_6.setPixmap(pix)
            self.mo_real_face_6.show()

            saved_face = data[2][5]
            converted = cv2.cvtColor(saved_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im1 = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix1 = QtGui.QPixmap.fromImage(im1)
            self.mo_saved_face_6.setPixmap(pix1)
            self.mo_saved_face_6.show()

            text = f"Gender :{data[2][7]}\nStatus : {data[2][4]}\nBlocked : {data[2][10]}\nGuest : {data[2][8]}\nSafety : {data[2][9]}\n{data[1]}"
            self.mo_detail_6.setText(text)
            self.mo_detail_6.show()

            if data[2][4] != "Allow" or \
                    data[2][10] != "Nein" or \
                    data[2][9] == "Manipulator" or \
                    data[2][9] == "Manipulationsverdacht":
                pix = QtGui.QPixmap("./icons/danger.png")
                self.mo_icon_6.setPixmap(pix)
            else:
                if data[3] == "Come In":
                    pix = QtGui.QPixmap("./icons/come_in.png")
                    self.mo_icon_6.setPixmap(pix)
                if data[3] == "Come Out":
                    pix = QtGui.QPixmap("./icons/come_out.png")
                    self.mo_icon_6.setPixmap(pix)

            self.mo_icon_6.show()

        except:
            self.mo_real_face_6.hide()
            self.mo_saved_face_6.hide()
            self.mo_detail_6.hide()
            self.mo_icon_6.hide()

        # ------------------
        try:
            data = temp[6]

            real_face = data[0]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.mo_real_face_7.setPixmap(pix)
            self.mo_real_face_7.show()

            saved_face = data[2][5]
            converted = cv2.cvtColor(saved_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im1 = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix1 = QtGui.QPixmap.fromImage(im1)
            self.mo_saved_face_7.setPixmap(pix1)
            self.mo_saved_face_7.show()

            text = f"Gender :{data[2][7]}\nStatus : {data[2][4]}\nBlocked : {data[2][10]}\nGuest : {data[2][8]}\nSafety : {data[2][9]}\n{data[1]}"
            self.mo_detail_7.setText(text)
            self.mo_detail_7.show()

            if data[2][4] != "Allow" or \
                    data[2][10] != "Nein" or \
                    data[2][9] == "Manipulator" or \
                    data[2][9] == "Manipulationsverdacht":
                pix = QtGui.QPixmap("./icons/danger.png")
                self.mo_icon_7.setPixmap(pix)
            else:
                if data[3] == "Come In":
                    pix = QtGui.QPixmap("./icons/come_in.png")
                    self.mo_icon_7.setPixmap(pix)
                if data[3] == "Come Out":
                    pix = QtGui.QPixmap("./icons/come_out.png")
                    self.mo_icon_7.setPixmap(pix)

            self.mo_icon_7.show()

        except:
            self.mo_real_face_7.hide()
            self.mo_saved_face_7.hide()
            self.mo_detail_7.hide()
            self.mo_icon_7.hide()

    def update_pm_show_persons(self):
        val1 = int(len(self.person_register_list) / 10)
        val2 = int(len(self.person_register_list) % 10)
        if val2 == 0:
            self.pm_total_pages = val1
        else:
            self.pm_total_pages = val1 + 1

        self.pm_page_num.setText(f"{self.pm_page_iter}/{self.pm_total_pages}")

        temp = self.person_register_list[(self.pm_page_iter - 1) * 10: (self.pm_page_iter) * 10]

        # ------------------
        try:
            data = temp[0]

            real_face = data[5]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.pm_person_1_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.pm_person_1_detail_value.setText(text)

            self.pm_person_1.show()
            self.pm_person_1_detail_text.show()
            self.pm_person_1_detail_value.show()
            self.pm_person_1_face.show()
            self.pm_person_1_edit_btn.show()
            self.pm_person_1_delete_btn.show()

        except:
            self.pm_person_1.hide()
            self.pm_person_1_detail_text.hide()
            self.pm_person_1_detail_value.hide()
            self.pm_person_1_face.hide()
            self.pm_person_1_edit_btn.hide()
            self.pm_person_1_delete_btn.hide()

        # ------------------
        try:
            data = temp[1]

            real_face = data[5]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.pm_person_2_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.pm_person_2_detail_value.setText(text)

            self.pm_person_2.show()
            self.pm_person_2_detail_text.show()
            self.pm_person_2_detail_value.show()
            self.pm_person_2_face.show()
            self.pm_person_2_edit_btn.show()
            self.pm_person_2_delete_btn.show()

        except:
            self.pm_person_2.hide()
            self.pm_person_2_detail_text.hide()
            self.pm_person_2_detail_value.hide()
            self.pm_person_2_face.hide()
            self.pm_person_2_edit_btn.hide()
            self.pm_person_2_delete_btn.hide()

        # ------------------
        try:
            data = temp[2]

            real_face = data[5]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.pm_person_3_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.pm_person_3_detail_value.setText(text)

            self.pm_person_3.show()
            self.pm_person_3_detail_text.show()
            self.pm_person_3_detail_value.show()
            self.pm_person_3_face.show()
            self.pm_person_3_edit_btn.show()
            self.pm_person_3_delete_btn.show()

        except:
            self.pm_person_3.hide()
            self.pm_person_3_detail_text.hide()
            self.pm_person_3_detail_value.hide()
            self.pm_person_3_face.hide()
            self.pm_person_3_edit_btn.hide()
            self.pm_person_3_delete_btn.hide()

        # ------------------
        try:
            data = temp[3]

            real_face = data[5]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.pm_person_3_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.pm_person_4_detail_value.setText(text)

            self.pm_person_4.show()
            self.pm_person_4_detail_text.show()
            self.pm_person_4_detail_value.show()
            self.pm_person_4_face.show()
            self.pm_person_4_edit_btn.show()
            self.pm_person_4_delete_btn.show()

        except:
            self.pm_person_4.hide()
            self.pm_person_4_detail_text.hide()
            self.pm_person_4_detail_value.hide()
            self.pm_person_4_face.hide()
            self.pm_person_4_edit_btn.hide()
            self.pm_person_4_delete_btn.hide()

        # ------------------
        try:
            data = temp[4]

            real_face = data[5]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.pm_person_5_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.pm_person_5_detail_value.setText(text)

            self.pm_person_5.show()
            self.pm_person_5_detail_text.show()
            self.pm_person_5_detail_value.show()
            self.pm_person_5_face.show()
            self.pm_person_5_edit_btn.show()
            self.pm_person_5_delete_btn.show()

        except:
            self.pm_person_5.hide()
            self.pm_person_5_detail_text.hide()
            self.pm_person_5_detail_value.hide()
            self.pm_person_5_face.hide()
            self.pm_person_5_edit_btn.hide()
            self.pm_person_5_delete_btn.hide()

        # ------------------
        try:
            data = temp[5]

            real_face = data[5]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.pm_person_6_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.pm_person_6_detail_value.setText(text)

            self.pm_person_6.show()
            self.pm_person_6_detail_text.show()
            self.pm_person_6_detail_value.show()
            self.pm_person_6_face.show()
            self.pm_person_6_edit_btn.show()
            self.pm_person_6_delete_btn.show()

        except:
            self.pm_person_6.hide()
            self.pm_person_6_detail_text.hide()
            self.pm_person_6_detail_value.hide()
            self.pm_person_6_face.hide()
            self.pm_person_6_edit_btn.hide()
            self.pm_person_6_delete_btn.hide()

        # ------------------
        try:
            data = temp[6]

            real_face = data[5]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.pm_person_7_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.pm_person_7_detail_value.setText(text)

            self.pm_person_7.show()
            self.pm_person_7_detail_text.show()
            self.pm_person_7_detail_value.show()
            self.pm_person_7_face.show()
            self.pm_person_7_edit_btn.show()
            self.pm_person_7_delete_btn.show()

        except:
            self.pm_person_7.hide()
            self.pm_person_7_detail_text.hide()
            self.pm_person_7_detail_value.hide()
            self.pm_person_7_face.hide()
            self.pm_person_7_edit_btn.hide()
            self.pm_person_7_delete_btn.hide()

        # ------------------
        try:
            data = temp[7]

            real_face = data[5]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.pm_person_8_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.pm_person_8_detail_value.setText(text)

            self.pm_person_8.show()
            self.pm_person_8_detail_text.show()
            self.pm_person_8_detail_value.show()
            self.pm_person_8_face.show()
            self.pm_person_8_edit_btn.show()
            self.pm_person_8_delete_btn.show()

        except:
            self.pm_person_8.hide()
            self.pm_person_8_detail_text.hide()
            self.pm_person_8_detail_value.hide()
            self.pm_person_8_face.hide()
            self.pm_person_8_edit_btn.hide()
            self.pm_person_8_delete_btn.hide()

        # ------------------
        try:
            data = temp[8]

            real_face = data[5]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.pm_person_9_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.pm_person_9_detail_value.setText(text)

            self.pm_person_9.show()
            self.pm_person_9_detail_text.show()
            self.pm_person_9_detail_value.show()
            self.pm_person_9_face.show()
            self.pm_person_9_edit_btn.show()
            self.pm_person_9_delete_btn.show()

        except:
            self.pm_person_9.hide()
            self.pm_person_9_detail_text.hide()
            self.pm_person_9_detail_value.hide()
            self.pm_person_9_face.hide()
            self.pm_person_9_edit_btn.hide()
            self.pm_person_9_delete_btn.hide()

        # ------------------
        try:
            data = temp[9]

            real_face = data[5]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.pm_person_10_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.pm_person_10_detail_value.setText(text)

            self.pm_person_10.show()
            self.pm_person_10_detail_text.show()
            self.pm_person_10_detail_value.show()
            self.pm_person_10_face.show()
            self.pm_person_10_edit_btn.show()
            self.pm_person_10_delete_btn.show()

        except:
            self.pm_person_10.hide()
            self.pm_person_10_detail_text.hide()
            self.pm_person_10_detail_value.hide()
            self.pm_person_10_face.hide()
            self.pm_person_10_edit_btn.hide()
            self.pm_person_10_delete_btn.hide()

    def update_ph_show_persons(self):
        val1 = int(len(self.person_history_list)/10)
        val2 = int(len(self.person_history_list)%10)
        if val2 == 0:
            self.ph_total_pages = val1
        else:
            self.ph_total_pages = val1 + 1

        self.ph_page_num.setText(f"{self.ph_page_iter}/{self.ph_total_pages}")

        temp = self.person_history_list[(self.ph_page_iter-1) * 10: (self.ph_page_iter) * 10]

        # ------------------
        try:
            data = temp[0]

            real_face = data[8]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.ph_person_1_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}\n{data[5]}\n{data[6]}\n{data[7]}"
            self.ph_person_1_detail_value.setText(text)

            self.ph_person_1.show()
            self.ph_person_1_detail_text.show()
            self.ph_person_1_detail_value.show()
            self.ph_person_1_face.show()

        except:
            self.ph_person_1.hide()
            self.ph_person_1_detail_text.hide()
            self.ph_person_1_detail_value.hide()
            self.ph_person_1_face.hide()

        # ------------------
        try:
            data = temp[1]

            real_face = data[8]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.ph_person_2_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}\n{data[5]}\n{data[6]}\n{data[7]}"
            self.ph_person_2_detail_value.setText(text)

            self.ph_person_2.show()
            self.ph_person_2_detail_text.show()
            self.ph_person_2_detail_value.show()
            self.ph_person_2_face.show()

        except:
            self.ph_person_2.hide()
            self.ph_person_2_detail_text.hide()
            self.ph_person_2_detail_value.hide()
            self.ph_person_2_face.hide()

        # ------------------
        try:
            data = temp[2]

            real_face = data[8]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.ph_person_3_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}\n{data[5]}\n{data[6]}\n{data[7]}"
            self.ph_person_3_detail_value.setText(text)

            self.ph_person_3.show()
            self.ph_person_3_detail_text.show()
            self.ph_person_3_detail_value.show()
            self.ph_person_3_face.show()

        except:
            self.ph_person_3.hide()
            self.ph_person_3_detail_text.hide()
            self.ph_person_3_detail_value.hide()
            self.ph_person_3_face.hide()

        # ------------------
        try:
            data = temp[3]

            real_face = data[8]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.ph_person_4_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}\n{data[5]}\n{data[6]}\n{data[7]}"
            self.ph_person_4_detail_value.setText(text)

            self.ph_person_4.show()
            self.ph_person_4_detail_text.show()
            self.ph_person_4_detail_value.show()
            self.ph_person_4_face.show()

        except:
            self.ph_person_4.hide()
            self.ph_person_4_detail_text.hide()
            self.ph_person_4_detail_value.hide()
            self.ph_person_4_face.hide()

        # ------------------
        try:
            data = temp[4]

            real_face = data[8]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.ph_person_5_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}\n{data[5]}\n{data[6]}\n{data[7]}"
            self.ph_person_5_detail_value.setText(text)

            self.ph_person_5.show()
            self.ph_person_5_detail_text.show()
            self.ph_person_5_detail_value.show()
            self.ph_person_5_face.show()

        except:
            self.ph_person_5.hide()
            self.ph_person_5_detail_text.hide()
            self.ph_person_5_detail_value.hide()
            self.ph_person_5_face.hide()

        # ------------------
        try:
            data = temp[5]

            real_face = data[8]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.ph_person_6_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}\n{data[5]}\n{data[6]}\n{data[7]}"
            self.ph_person_6_detail_value.setText(text)

            self.ph_person_6.show()
            self.ph_person_6_detail_text.show()
            self.ph_person_6_detail_value.show()
            self.ph_person_6_face.show()

        except:
            self.ph_person_6.hide()
            self.ph_person_6_detail_text.hide()
            self.ph_person_6_detail_value.hide()
            self.ph_person_6_face.hide()

        # ------------------
        try:
            data = temp[6]

            real_face = data[8]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.ph_person_7_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}\n{data[5]}\n{data[6]}\n{data[7]}"
            self.ph_person_7_detail_value.setText(text)

            self.ph_person_7.show()
            self.ph_person_7_detail_text.show()
            self.ph_person_7_detail_value.show()
            self.ph_person_7_face.show()

        except:
            self.ph_person_7.hide()
            self.ph_person_7_detail_text.hide()
            self.ph_person_7_detail_value.hide()
            self.ph_person_7_face.hide()

        # ------------------
        try:
            data = temp[7]

            real_face = data[8]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.ph_person_8_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}\n{data[5]}\n{data[6]}\n{data[7]}"
            self.ph_person_8_detail_value.setText(text)

            self.ph_person_8.show()
            self.ph_person_8_detail_text.show()
            self.ph_person_8_detail_value.show()
            self.ph_person_8_face.show()

        except:
            self.ph_person_8.hide()
            self.ph_person_8_detail_text.hide()
            self.ph_person_8_detail_value.hide()
            self.ph_person_8_face.hide()

        # ------------------
        try:
            data = temp[8]

            real_face = data[8]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.ph_person_9_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}\n{data[5]}\n{data[6]}\n{data[7]}"
            self.ph_person_9_detail_value.setText(text)

            self.ph_person_9.show()
            self.ph_person_9_detail_text.show()
            self.ph_person_9_detail_value.show()
            self.ph_person_9_face.show()

        except:
            self.ph_person_9.hide()
            self.ph_person_9_detail_text.hide()
            self.ph_person_9_detail_value.hide()
            self.ph_person_9_face.hide()

        # ------------------
        try:
            data = temp[9]

            real_face = data[8]
            converted = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(converted, (int(50 * X_SCALE), int(70 * Y_SCALE)))
            im = QtGui.QImage(resized.data, resized.shape[1], resized.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(im)
            self.ph_person_10_face.setPixmap(pix)

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}\n{data[5]}\n{data[6]}\n{data[7]}"
            self.ph_person_10_detail_value.setText(text)

            self.ph_person_10.show()
            self.ph_person_10_detail_text.show()
            self.ph_person_10_detail_value.show()
            self.ph_person_10_face.show()

        except:
            self.ph_person_10.hide()
            self.ph_person_10_detail_text.hide()
            self.ph_person_10_detail_value.hide()
            self.ph_person_10_face.hide()

    def update_um_show_users(self):
        val1 = int(len(self.user_register_list) / 10)
        val2 = int(len(self.user_register_list) % 10)
        if val2 == 0:
            self.um_total_pages = val1
        else:
            self.um_total_pages = val1 + 1

        self.um_page_num.setText(f"{self.um_page_iter}/{self.um_total_pages}")

        temp = self.user_register_list[(self.um_page_iter - 1) * 10: (self.um_page_iter) * 10]

        # ------------------
        try:
            data = temp[0]

            text = f"{data[1]}\n{data[2]}\n{data[4]}\n{data[5]}\n{data[6]}"
            self.um_user_1_detail_value.setText(text)

            self.um_user_1.show()
            self.um_user_1_detail_text.show()
            self.um_user_1_detail_value.show()
            self.um_user_1_face.show()
            self.um_user_1_edit_btn.show()
            self.um_user_1_delete_btn.show()

        except:
            self.um_user_1.hide()
            self.um_user_1_detail_text.hide()
            self.um_user_1_detail_value.hide()
            self.um_user_1_face.hide()
            self.um_user_1_edit_btn.hide()
            self.um_user_1_delete_btn.hide()

        # ------------------
        try:
            data = temp[1]

            text = f"{data[1]}\n{data[2]}\n{data[4]}\n{data[5]}\n{data[6]}"
            self.um_user_2_detail_value.setText(text)

            self.um_user_2.show()
            self.um_user_2_detail_text.show()
            self.um_user_2_detail_value.show()
            self.um_user_2_face.show()
            self.um_user_2_edit_btn.show()
            self.um_user_2_delete_btn.show()

        except:
            self.um_user_2.hide()
            self.um_user_2_detail_text.hide()
            self.um_user_2_detail_value.hide()
            self.um_user_2_face.hide()
            self.um_user_2_edit_btn.hide()
            self.um_user_2_delete_btn.hide()

        # ------------------
        try:
            data = temp[2]

            text = f"{data[1]}\n{data[2]}\n{data[4]}\n{data[5]}\n{data[6]}"
            self.um_user_3_detail_value.setText(text)

            self.um_user_3.show()
            self.um_user_3_detail_text.show()
            self.um_user_3_detail_value.show()
            self.um_user_3_face.show()
            self.um_user_3_edit_btn.show()
            self.um_user_3_delete_btn.show()

        except:
            self.um_user_3.hide()
            self.um_user_3_detail_text.hide()
            self.um_user_3_detail_value.hide()
            self.um_user_3_face.hide()
            self.um_user_3_edit_btn.hide()
            self.um_user_3_delete_btn.hide()

        # ------------------
        try:
            data = temp[3]

            text = f"{data[1]}\n{data[2]}\n{data[4]}\n{data[5]}\n{data[6]}"
            self.um_user_4_detail_value.setText(text)

            self.um_user_4.show()
            self.um_user_4_detail_text.show()
            self.um_user_4_detail_value.show()
            self.um_user_4_face.show()
            self.um_user_4_edit_btn.show()
            self.um_user_4_delete_btn.show()

        except:
            self.um_user_4.hide()
            self.um_user_4_detail_text.hide()
            self.um_user_4_detail_value.hide()
            self.um_user_4_face.hide()
            self.um_user_4_edit_btn.hide()
            self.um_user_4_delete_btn.hide()

        # ------------------
        try:
            data = temp[4]

            text = f"{data[1]}\n{data[2]}\n{data[4]}\n{data[5]}\n{data[6]}"
            self.um_user_5_detail_value.setText(text)

            self.um_user_5.show()
            self.um_user_5_detail_text.show()
            self.um_user_5_detail_value.show()
            self.um_user_5_face.show()
            self.um_user_5_edit_btn.show()
            self.um_user_5_delete_btn.show()

        except:
            self.um_user_5.hide()
            self.um_user_5_detail_text.hide()
            self.um_user_5_detail_value.hide()
            self.um_user_5_face.hide()
            self.um_user_5_edit_btn.hide()
            self.um_user_5_delete_btn.hide()

        # ------------------
        try:
            data = temp[5]

            text = f"{data[1]}\n{data[2]}\n{data[4]}\n{data[5]}\n{data[6]}"
            self.um_user_6_detail_value.setText(text)

            self.um_user_6.show()
            self.um_user_6_detail_text.show()
            self.um_user_6_detail_value.show()
            self.um_user_6_face.show()
            self.um_user_6_edit_btn.show()
            self.um_user_6_delete_btn.show()

        except:
            self.um_user_6.hide()
            self.um_user_6_detail_text.hide()
            self.um_user_6_detail_value.hide()
            self.um_user_6_face.hide()
            self.um_user_6_edit_btn.hide()
            self.um_user_6_delete_btn.hide()

        # ------------------
        try:
            data = temp[6]

            text = f"{data[1]}\n{data[2]}\n{data[4]}\n{data[5]}\n{data[6]}"
            self.um_user_7_detail_value.setText(text)

            self.um_user_7.show()
            self.um_user_7_detail_text.show()
            self.um_user_7_detail_value.show()
            self.um_user_7_face.show()
            self.um_user_7_edit_btn.show()
            self.um_user_7_delete_btn.show()

        except:
            self.um_user_7.hide()
            self.um_user_7_detail_text.hide()
            self.um_user_7_detail_value.hide()
            self.um_user_7_face.hide()
            self.um_user_7_edit_btn.hide()
            self.um_user_7_delete_btn.hide()

        # ------------------
        try:
            data = temp[7]

            text = f"{data[1]}\n{data[2]}\n{data[4]}\n{data[5]}\n{data[6]}"
            self.um_user_8_detail_value.setText(text)

            self.um_user_8.show()
            self.um_user_8_detail_text.show()
            self.um_user_8_detail_value.show()
            self.um_user_8_face.show()
            self.um_user_8_edit_btn.show()
            self.um_user_8_delete_btn.show()

        except:
            self.um_user_8.hide()
            self.um_user_8_detail_text.hide()
            self.um_user_8_detail_value.hide()
            self.um_user_8_face.hide()
            self.um_user_8_edit_btn.hide()
            self.um_user_8_delete_btn.hide()

        # ------------------
        try:
            data = temp[8]

            text = f"{data[1]}\n{data[2]}\n{data[4]}\n{data[5]}\n{data[6]}"
            self.um_user_9_detail_value.setText(text)

            self.um_user_9.show()
            self.um_user_9_detail_text.show()
            self.um_user_9_detail_value.show()
            self.um_user_9_face.show()
            self.um_user_9_edit_btn.show()
            self.um_user_9_delete_btn.show()

        except:
            self.um_user_9.hide()
            self.um_user_9_detail_text.hide()
            self.um_user_9_detail_value.hide()
            self.um_user_9_face.hide()
            self.um_user_9_edit_btn.hide()
            self.um_user_9_delete_btn.hide()

        # ------------------
        try:
            data = temp[9]

            text = f"{data[1]}\n{data[2]}\n{data[4]}\n{data[5]}\n{data[6]}"
            self.um_user_10_detail_value.setText(text)

            self.um_user_10.show()
            self.um_user_10_detail_text.show()
            self.um_user_10_detail_value.show()
            self.um_user_10_face.show()
            self.um_user_10_edit_btn.show()
            self.um_user_10_delete_btn.show()

        except:
            self.um_user_10.hide()
            self.um_user_10_detail_text.hide()
            self.um_user_10_detail_value.hide()
            self.um_user_10_face.hide()
            self.um_user_10_edit_btn.hide()
            self.um_user_10_delete_btn.hide()
            
    def update_uh_show_users(self):
        val1 = int(len(self.user_history_list) / 10)
        val2 = int(len(self.user_history_list) % 10)
        if val2 == 0:
            self.uh_total_pages = val1
        else:
            self.uh_total_pages = val1 + 1

        self.uh_page_num.setText(f"{self.uh_page_iter}/{self.uh_total_pages}")

        temp = self.user_history_list[(self.uh_page_iter - 1) * 10: (self.uh_page_iter) * 10]

        # ------------------
        try:
            data = temp[0]

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.uh_user_1_detail_value.setText(text)

            self.uh_user_1.show()
            self.uh_user_1_detail_text.show()
            self.uh_user_1_detail_value.show()
            self.uh_user_1_face.show()

        except:
            self.uh_user_1.hide()
            self.uh_user_1_detail_text.hide()
            self.uh_user_1_detail_value.hide()
            self.uh_user_1_face.hide()

        # ------------------
        try:
            data = temp[1]

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.uh_user_2_detail_value.setText(text)

            self.uh_user_2.show()
            self.uh_user_2_detail_text.show()
            self.uh_user_2_detail_value.show()
            self.uh_user_2_face.show()

        except:
            self.uh_user_2.hide()
            self.uh_user_2_detail_text.hide()
            self.uh_user_2_detail_value.hide()
            self.uh_user_2_face.hide()

        # ------------------
        try:
            data = temp[2]

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.uh_user_3_detail_value.setText(text)

            self.uh_user_3.show()
            self.uh_user_3_detail_text.show()
            self.uh_user_3_detail_value.show()
            self.uh_user_3_face.show()

        except:
            self.uh_user_3.hide()
            self.uh_user_3_detail_text.hide()
            self.uh_user_3_detail_value.hide()
            self.uh_user_3_face.hide()

        # ------------------
        try:
            data = temp[3]

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.uh_user_4_detail_value.setText(text)

            self.uh_user_4.show()
            self.uh_user_4_detail_text.show()
            self.uh_user_4_detail_value.show()
            self.uh_user_4_face.show()

        except:
            self.uh_user_4.hide()
            self.uh_user_4_detail_text.hide()
            self.uh_user_4_detail_value.hide()
            self.uh_user_4_face.hide()
            
        # ------------------
        try:
            data = temp[4]

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.uh_user_5_detail_value.setText(text)

            self.uh_user_5.show()
            self.uh_user_5_detail_text.show()
            self.uh_user_5_detail_value.show()
            self.uh_user_5_face.show()

        except:
            self.uh_user_5.hide()
            self.uh_user_5_detail_text.hide()
            self.uh_user_5_detail_value.hide()
            self.uh_user_5_face.hide()

        # ------------------
        try:
            data = temp[5]

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.uh_user_6_detail_value.setText(text)

            self.uh_user_6.show()
            self.uh_user_6_detail_text.show()
            self.uh_user_6_detail_value.show()
            self.uh_user_6_face.show()

        except:
            self.uh_user_6.hide()
            self.uh_user_6_detail_text.hide()
            self.uh_user_6_detail_value.hide()
            self.uh_user_6_face.hide()
            
        # ------------------
        try:
            data = temp[6]

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.uh_user_7_detail_value.setText(text)

            self.uh_user_7.show()
            self.uh_user_7_detail_text.show()
            self.uh_user_7_detail_value.show()
            self.uh_user_7_face.show()

        except:
            self.uh_user_7.hide()
            self.uh_user_7_detail_text.hide()
            self.uh_user_7_detail_value.hide()
            self.uh_user_7_face.hide()
            
        # ------------------
        try:
            data = temp[7]

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.uh_user_8_detail_value.setText(text)

            self.uh_user_8.show()
            self.uh_user_8_detail_text.show()
            self.uh_user_8_detail_value.show()
            self.uh_user_8_face.show()

        except:
            self.uh_user_8.hide()
            self.uh_user_8_detail_text.hide()
            self.uh_user_8_detail_value.hide()
            self.uh_user_8_face.hide()
            
        # ------------------
        try:
            data = temp[8]

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.uh_user_9_detail_value.setText(text)

            self.uh_user_9.show()
            self.uh_user_9_detail_text.show()
            self.uh_user_9_detail_value.show()
            self.uh_user_9_face.show()

        except:
            self.uh_user_9.hide()
            self.uh_user_9_detail_text.hide()
            self.uh_user_9_detail_value.hide()
            self.uh_user_9_face.hide()
            
        # ------------------
        try:
            data = temp[9]

            text = f"{data[1]}\n{data[2]}\n{data[3]}\n{data[4]}"
            self.uh_user_10_detail_value.setText(text)

            self.uh_user_10.show()
            self.uh_user_10_detail_text.show()
            self.uh_user_10_detail_value.show()
            self.uh_user_10_face.show()

        except:
            self.uh_user_10.hide()
            self.uh_user_10_detail_text.hide()
            self.uh_user_10_detail_value.hide()
            self.uh_user_10_face.hide()

    def get_similarity(self, emb1, emb2):
        dot = np.sum(np.multiply(emb1, emb2), axis=0)
        norm = np.linalg.norm(emb1, axis=0) * np.linalg.norm(emb2, axis=0)
        similarity = min(1, max(-1, dot / norm))
        cosdist = min(0.5, np.arccos(similarity) / math.pi)
        pcnt = 0
        thr = 0.35
        if cosdist <= thr:
            pcnt = (0.2 / thr) * cosdist
        elif cosdist > thr and cosdist <= 0.5:
            pcnt = 5.33333 * cosdist - 1.66667
        pcnt = (1.0 - pcnt) * 100
        pcnt = min(100, pcnt)
        return pcnt

    def is_matching(self, encodding1, encodding2):
        if encodding1 is None or encodding2 is None:
            return False
        else:
            sim = self.get_similarity(encodding1, encodding2)
            if sim > 75:
                return True
            else:
                return False

    def find_person(self, img, emb):
        person_data = []
        person_data.append(img)
        person_data.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        data = ['Unknown' for x in range(0, 17)]
        data[5] = cv2.imread('icons/face_icon.png')
        data[6] = emb
        person_data.append(data)
        person_data.append("Unknown")
        for i in range(0, len(self.person_register_list)):
            emb0 = self.person_register_list[i][6] # face embbeding value

            if self.is_matching(emb0, emb):
                data = self.person_register_list[i]
                person_data[2] = data
                person_data[-1] = self.detected_person_action
                break

        return person_data

    def create_proc_threads(self):
        if not self.process_detected_faces:
            if self.cam1_faces is not None and len(self.cam1_faces) > 0:
                self.detected_faces = self.cam1_faces.copy()
                self.detected_person_action = "Come In"

                self.recog_thread = Thread(target=self.proc_detected_faces)
                self.recog_thread.setDaemon(True)
                self.recog_thread.start()
                self.process_detected_faces = True

        if not self.process_detected_faces:
            if self.cam2_faces is not None and len(self.cam2_faces) > 0:
                self.detected_faces = self.cam2_faces.copy()
                self.detected_person_action = "Come Out"

                self.recog_thread = Thread(target=self.proc_detected_faces)
                self.recog_thread.setDaemon(True)
                self.recog_thread.start()
                self.process_detected_faces = True

    def add_person_history(self):
        person_data = self.now_person_history_list[-1]
        face_img = person_data[0]
        face_bytes = cv2.imencode('.jpg', face_img)[1].tobytes()
        time = person_data[1]
        details = person_data[2]

        name = details[1]
        gender = details[7]
        age = details[3]
        place = details[13]
        view = '0'
        action = self.detected_person_action

        # ----  save person to db --------
        try:
            cursor = self.db_conn.cursor()
            sqlite_insert_blob_query = """ INSERT INTO person_history_table
                    (name, gender, age, time, place, view, action, photo) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""

            # Convert data into tuple format
            data_tuple = (name, gender, age, time, place, view, action, face_bytes)

            cursor.execute(sqlite_insert_blob_query, data_tuple)
            id = cursor.lastrowid
            self.db_conn.commit()
            print("Add new person history successfully.")
            cursor.close()

            self.person_history_list.append([id, name, gender, age, time, place, view, action, face_img])

        except sqlite3.Error as error:
            print("Failed to add new person history.", error)

    def proc_detected_faces(self):
        for img in self.detected_faces:
            inp_img = img.copy()
            inp_img = cv2.resize(inp_img, (600, 500))
            faces = self.face_recognizer.predict(inp_img)
            if len(faces) > 0:
                emb = faces[0].embedding

                person_data = self.find_person(img, emb)

                # ----  check passed persons list  ----
                match_flag = False
                try:
                    temp = self.now_person_history_list[-1]
                    emb0 = temp[2][6]
                    action = temp[3]

                    if self.is_matching(emb0, emb) and action == self.detected_person_action:
                        self.now_person_history_list[-1] = person_data
                        match_flag = True
                except:
                    pass

                if not match_flag:
                    # -------  check gov request  ------------
                    if self.detected_person_action == "Come In":
                        try:
                            firstname = person_data[2][1].split(' ')[0]
                            lastname = person_data[2][1].split(' ')[1]
                            birth = person_data[2][2]
                            result = connect_gov('12', firstname, lastname, birth)

                            if result == "Der Spieler ist nicht gesperrt.":
                                person_data[2][4] = "Allow"
                            else:
                                person_data[2][4] = "Not Allow"
                        except:
                            pass

                    self.now_person_history_list.append(person_data)
                    # ----- show latest detected faces automatically ----
                    val1 = int(len(self.now_person_history_list) / 7)
                    val2 = int(len(self.now_person_history_list) % 7)
                    if val2 == 0:
                        self.mo_page_iter = val1
                    else:
                        self.mo_page_iter = val1 + 1
                    # ---------------------------------------------------

                    self.add_person_history_signal.emit()

                # ------- show warning message  ----------
                self.warning_unregister_message.hide()
                self.warning_danger_message.hide()
                if person_data[2][4] != "Allow" or \
                        person_data[2][10] != "Nein" or \
                        person_data[2][9] == "Manipulationsverdacht" or \
                        person_data[2][9] == "Manipulator":

                    if person_data[2][0] == 'Unknown':
                        self.warning_unregister_message.show()
                    else:
                        self.warning_danger_message.show()

        if self.monitor_view_flag:
            self.update_detected_persons_signal.emit()

        self.process_detected_faces = False

    def check_cameras(self):
        while True:
            if self.cap1.isOpened() and not self.cam1_running:
                print("Camera No1 connected")

                self.cam1_thread = Thread(target=self.cam1_process)
                self.cam1_thread.setDaemon(True)
                self.cam1_thread.start()

                self.cam1_running = True

            if self.cap2.isOpened() and not self.cam2_running:
                print("Camera No2 connected")

                self.cam2_thread = Thread(target=self.cam2_process)
                self.cam2_thread.setDaemon(True)
                self.cam2_thread.start()

                self.cam2_running = True

            if self.cap3.isOpened() and not self.cam3_running:
                print("Camera No3 connected")

                self.cam3_thread = Thread(target=self.cam3_process)
                self.cam3_thread.setDaemon(True)
                self.cam3_thread.start()

                self.cam3_running = True

            if self.cap4.isOpened() and not self.cam4_running:
                print("Camera No4 connected")

                self.cam4_thread = Thread(target=self.cam4_process)
                self.cam4_thread.setDaemon(True)
                self.cam4_thread.start()

                self.cam4_running = True

            time.sleep(1)

    def cam1_process(self):
        self.mo_camera1_icon.hide()
        while True:
            ret1, frame1 = self.cap1.read()
            if not ret1:
                continue
            self.cam1_frame_ori = frame1.copy()
            img1, faces1 = self.face_detector1.detect(frame1)
            self.cam1_frame = img1.copy()
            self.cam1_faces = faces1.copy()
            self.cam1_capture_signal.emit()

    def draw_cam1_frame(self):
        draw_img1 = self.cam1_frame.copy()
        resized1 = cv2.resize(draw_img1, (int(260 * X_SCALE), int(280 * Y_SCALE)))
        converted1 = cv2.cvtColor(resized1, cv2.COLOR_BGR2RGB)
        im1 = QtGui.QImage(converted1.data, converted1.shape[1], converted1.shape[0], QtGui.QImage.Format_RGB888)
        pix1 = QtGui.QPixmap.fromImage(im1)
        self.mo_camera1_view.setPixmap(pix1)

    def cam2_process(self):
        self.mo_camera2_icon.hide()
        while True:
            ret2, frame2 = self.cap2.read()
            if not ret2:
                continue
            self.cam2_frame_ori = frame2.copy()
            img2, faces2 = self.face_detector2.detect(frame2)
            self.cam2_frame = img2.copy()
            self.cam2_faces = faces2.copy()
            self.cam2_capture_signal.emit()

    def draw_cam2_frame(self):
        draw_img2 = self.cam2_frame.copy()
        resized2 = cv2.resize(draw_img2, (int(260 * X_SCALE), int(280 * Y_SCALE)))
        converted2 = cv2.cvtColor(resized2, cv2.COLOR_BGR2RGB)
        im2 = QtGui.QImage(converted2.data, converted2.shape[1], converted2.shape[0], QtGui.QImage.Format_RGB888)
        pix2 = QtGui.QPixmap.fromImage(im2)
        self.mo_camera2_view.setPixmap(pix2)

    def cam3_process(self):
        self.mo_camera3_icon.hide()
        while True:
            ret3, frame3 = self.cap3.read()
            if not ret3:
                continue
            img3, faces3 = self.face_detector1.detect(frame3)
            self.cam3_frame = img3
            self.cam3_capture_signal.emit()

    def draw_cam3_frame(self):
        draw_img3 = self.cam3_frame.copy()
        resized3 = cv2.resize(draw_img3, (int(260 * X_SCALE), int(280 * Y_SCALE)))
        converted3 = cv2.cvtColor(resized3, cv2.COLOR_BGR2RGB)
        im3 = QtGui.QImage(converted3.data, converted3.shape[1], converted3.shape[0], QtGui.QImage.Format_RGB888)
        pix3 = QtGui.QPixmap.fromImage(im3)
        self.mo_camera3_view.setPixmap(pix3)

    def cam4_process(self):
        self.mo_camera4_icon.hide()
        while True:
            ret4, frame4 = self.cap4.read()
            if not ret4:
                continue
            img4, faces4 = self.face_detector2.detect(frame4)
            self.cam4_frame = img4
            self.cam4_capture_signal.emit()

    def draw_cam4_frame(self):
        draw_img4 = self.cam4_frame.copy()
        resized4 = cv2.resize(draw_img4, (int(260 * X_SCALE), int(280 * Y_SCALE)))
        converted4 = cv2.cvtColor(resized4, cv2.COLOR_BGR2RGB)
        im4 = QtGui.QImage(converted4.data, converted4.shape[1], converted4.shape[0], QtGui.QImage.Format_RGB888)
        pix4 = QtGui.QPixmap.fromImage(im4)
        self.mo_camera4_view.setPixmap(pix4)

    def calc_age(self, birth_str):
        try:
            b_year = int(birth_str.split('.')[2])
            b_month = int(birth_str.split('.')[1].replace('0', ''))
            b_day = int(birth_str.split('.')[0].replace('0', ''))
            birth_date = date(b_year, b_month, b_day)
            today = date.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            return age

        except ValueError:
            font = QtGui.QFont()
            font.setFamily("Arial")
            font.setPointSize(14)

            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("Please Type Birthday like format : DD.MM.YYYY")
            msg.setWindowTitle("Invalid Birthday")
            msg.setFont(font)
            msg.setStyleSheet(L_BG)
            msg.exec_()

            return None

    def draw_capture_frame(self):
        draw_img1 = self.capture_frame.copy()
        resized1 = cv2.resize(draw_img1, (int(500 * X_SCALE), int(400 * Y_SCALE)))
        converted1 = cv2.cvtColor(resized1, cv2.COLOR_BGR2RGB)
        im1 = QtGui.QImage(converted1.data, converted1.shape[1], converted1.shape[0], QtGui.QImage.Format_RGB888)
        pix1 = QtGui.QPixmap.fromImage(im1)
        self.cp_frame_label.setPixmap(pix1)

        self.cp_camera_icon.hide()

    def face_capture(self):
        while True:
            ret, image = self.capture_cap.read()
            if not ret:
                continue
            self.capture_frame = image.copy()
            self.register_capture_signal.emit()

            if self.capture_stop_flag:
                break

        self.capture_stop_flag = False

        self.pe_face_image = image.copy()
        draw_img1 = self.pe_face_image.copy()
        resized1 = cv2.resize(draw_img1, (int(130 * X_SCALE), int(200 * Y_SCALE)))
        converted1 = cv2.cvtColor(resized1, cv2.COLOR_BGR2RGB)
        im1 = QtGui.QImage(converted1.data, converted1.shape[1], converted1.shape[0], QtGui.QImage.Format_RGB888)
        pix1 = QtGui.QPixmap.fromImage(im1)
        self.pe_person_face.setPixmap(pix1)

    def mrz_recog(self):
        while True:
            ret, image = self.capture_cap.read()
            if not ret:
                continue

            self.capture_frame = image.copy()
            self.register_capture_signal.emit()

            if self.capture_stop_flag:
                break

        self.capture_stop_flag = False

        image = self.capture_frame.copy()
        mrz = self.mrz_reader.read_mrz(image)

        if mrz is not None:
            try:
                birth_date = mrz['date_of_birth']
                year = int(birth_date[:2])

                # birth_date = mrz['date_of_birth']
                # year = int(birth_date[:2])

                today = datetime.today()
                current_year = int(str(today.year)[2:])

                if year > current_year:
                    birth_year = 1900 + year
                else:
                    birth_year = 2000 + year

                card_birth = birth_date[4:] + '.' + birth_date[2:4] + '.' + str(birth_year)
                # expire_date = mrz['date_of_expire']
                real_age = self.calc_age(card_birth)
                card_firstname = mrz['first_name'].translate(spcial_char_map)
                card_lastname = mrz['last_name'].translate(spcial_char_map)

                self.pe_name_value.setText(card_firstname + ' ' + card_lastname)
                self.pe_birthday_value.setText(card_birth)
                self.pe_age_value.setText(str(real_age))

            except ValueError:
                pass

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.login_login_btn_clicked()


class SignWin(QtWidgets.QWidget):
    def __init__(self, parent):
        super(SignWin, self).__init__()

        self.parent = parent

        self.setWindowTitle("Signature Window")

        self.sign_bg_label = QtWidgets.QLabel(self)
        self.sign_bg_label.setStyleSheet(T_BG)
        self.sign_bg_label.setGeometry(int(0 * X_SCALE1), int(0 * Y_SCALE1), int(800 * X_SCALE1), int(600 * Y_SCALE1))

        self.sign_frame_label = QtWidgets.QLabel(self)
        self.sign_frame_label.setStyleSheet(L_BG)
        self.sign_frame_label.setGeometry(int(0 * X_SCALE1), int(0 * Y_SCALE1), int(800 * X_SCALE1), int(600 * Y_SCALE1))

        self.sign_canvas = QtGui.QPixmap(int(800 * X_SCALE1), int(600 * Y_SCALE1))
        self.sign_canvas.fill(QtCore.Qt.white)
        self.sign_frame_label.setPixmap(self.sign_canvas)

        self.painter = QtGui.QPainter(self.sign_canvas)
        self.painter.setPen(QtGui.QPen(Qt.black, 5, Qt.SolidLine,
                            Qt.RoundCap, Qt.RoundJoin))

        self.sign_frame_label.mouseMoveEvent = self.mouse_move
        self.sign_frame_label.mouseReleaseEvent = self.mouse_release

        self.sign_clear_btn = QtWidgets.QPushButton(self)
        self.sign_clear_btn.setGeometry(int(160 * X_SCALE1), int(550 * Y_SCALE1), int(140 * X_SCALE1), int(50 * Y_SCALE1))
        self.sign_clear_btn.setText("Clear")
        self.sign_clear_btn.setStyleSheet(L_BT_N)

        self.sign_capture_btn = QtWidgets.QPushButton(self)
        self.sign_capture_btn.setGeometry(int(330 * X_SCALE1), int(550 * Y_SCALE1), int(140 * X_SCALE1), int(50 * Y_SCALE1))
        self.sign_capture_btn.setText("Capture")
        self.sign_capture_btn.setStyleSheet(L_BT_N)

        self.sign_cancel_btn = QtWidgets.QPushButton(self)
        self.sign_cancel_btn.setGeometry(int(500 * X_SCALE1), int(550 * Y_SCALE1), int(140 * X_SCALE1), int(50 * Y_SCALE1))
        self.sign_cancel_btn.setText("Cancel")
        self.sign_cancel_btn.setStyleSheet(L_BT_N)

        self.last_x, self.last_y = None, None

        self.sign_clear_btn.pressed.connect(self.sign_clear_btn_clicked)
        self.sign_capture_btn.pressed.connect(self.sign_capture_btn_clicked)
        self.sign_cancel_btn.pressed.connect(self.sign_cancel_btn_clicked)

    def mouse_move(self, e):
        if self.last_x is None:  # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return  # Ignore the first time.

        self.painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        # self.painter.end()
        self.sign_frame_label.setPixmap(self.sign_canvas)
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouse_release(self, e):
        self.last_x = None
        self.last_y = None

    def sign_clear_btn_clicked(self):
        self.sign_frame_label.clear()
        self.sign_canvas.fill(QtCore.Qt.white)

    def sign_capture_btn_clicked(self):
        try:
            img = self.sign_canvas.toImage()
            img_name = f'database/agreements/{self.parent.pe_name_value.text()}.png'
            img.save(img_name)

            img = Image.new("RGB", (200, 30), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            date_text = datetime.now().strftime("%Y-%m-%d")
            draw.text((10, 10), date_text, fill=(0, 0, 0), font_size=18)
            img.save('date.jpg')

            src_pdf_filename = 'database/agreements/sample.pdf'
            dst_pdf_filename = f'database/agreements/{self.parent.pe_name_value.text()}.pdf'

            sign_img = open(img_name, "rb").read()
            date_img = open('date.jpg', "rb").read()

            date_rect = fitz.Rect(60, 570, 260, 600)
            sign_rect = fitz.Rect(50, 600, 300, 700)

            document = fitz.open(src_pdf_filename)
            page = document[0]
            if not page.is_wrapped:
                page.wrap_contents()

            page.insert_image(date_rect, stream=date_img)
            page.insert_image(sign_rect, stream=sign_img)

            document.save(dst_pdf_filename)
            document.close()
            os.remove(img_name)
            os.remove('date.jpg')

            print("Successfully made pdf file")

            doc = fitz.open(dst_pdf_filename)
            page = doc.load_page(0)  # number of page
            pix = page.get_pixmap()
            output = dst_pdf_filename.replace('pdf', 'jpg')
            pix.save(output)
            doc.close()

            pix = QtGui.QPixmap(output)
            self.parent.pe_agree_text.setPixmap(pix)
        except:
            print("Error to make agreement pdf with sign.")

        self.close()

    def sign_cancel_btn_clicked(self):
        self.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    screen = app.primaryScreen()
    size = screen.size()
    X_SCALE = size.width() / 800
    Y_SCALE = size.height() / 600

    screens = app.screens()

    print('Main Screen Size: %d x %d' % (size.width(), size.height()))
    mainwin = FaceRecogSys()
    mainwin.show()

    sign_monitor = QtWidgets.QDesktopWidget().screenGeometry(1)
    size1 = sign_monitor.size()
    X_SCALE1 = size1.width() / 800
    Y_SCALE1 = size1.height() / 600
    print('Second Screen Size: %d x %d' % (size1.width(), size1.height()))

    sys.exit(app.exec_())
