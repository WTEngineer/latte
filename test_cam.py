import cv2
import configparser

paramFile = "config.ini"
config_params = configparser.ConfigParser()
config_params.read(paramFile)

cam1_id = config_params.getint("camera", "came_entrance")
cam2_id = config_params.getint("camera", "came_exit")
cam3_id = config_params.getint("camera", "came_capturing")

cap1 = cv2.VideoCapture(cam1_id, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(cam2_id, cv2.CAP_DSHOW)
cap3 = cv2.VideoCapture(cam3_id, cv2.CAP_DSHOW)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap1.set(cv2.CAP_PROP_FPS, 30)

cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FPS, 30)

cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap3.set(cv2.CAP_PROP_FPS, 30)
# while True:
#     if not cap1.isOpened() or not cap2.isOpened() or not cap3.isOpened():
#         continue
#     print("All cameras is opened")
#     break

while True:
    ret1, image1 = cap1.read()
    ret2, image2 = cap2.read()
    ret3, image3 = cap3.read()

    if not ret1 or not ret2 or not ret3:
        print(ret1, ret2, ret3)
        continue

    cv2.imshow("cam1", image1)
    cv2.imshow("cam2", image2)
    cv2.imshow("cam3", image3)

    cv2.waitKey(1)

cap1.close()
cap2.close()
cap3.close()