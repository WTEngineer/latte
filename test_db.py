import sqlite3
import cv2
import numpy as np
from face_recog import FaceRecognizer


def insertBLOB(name, birth, age, photo, emb):
    try:
        sqliteConnection = sqlite3.connect('database/face_db.db')
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")
        sqlite_insert_blob_query = """ INSERT INTO face_db
                                  (name, birth, age, photo, emb) VALUES (?, ?, ?, ?, ?)"""

        # Convert data into tuple format
        data_tuple = (name, birth, age, photo, emb)
        cursor.execute(sqlite_insert_blob_query, data_tuple)
        sqliteConnection.commit()
        print("Image and file inserted successfully as a BLOB into a table")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert blob data into sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("the sqlite connection is closed")


face = cv2.imread('icons/face.jpg')
face_bytes = cv2.imencode('.jpg', face)[1].tobytes()

face_recognizer = FaceRecognizer()

faces = face_recognizer.predict(face)
emb = faces[0].embedding
emb_bytes = emb.tobytes()


name = "Zhen Zhong"
birth = "1992.04.15"
age = "33"

insertBLOB(name, birth, age, face_bytes, emb_bytes)

# nparr = np.frombuffer(face_bytes, np.uint8)
# img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
# emb_array = np.frombuffer(emb_bytes, np.float32)
#
# if (emb == emb_array).all():
#     print('True')
#
# cv2.imshow("res", img)
# cv2.waitKey(0)
