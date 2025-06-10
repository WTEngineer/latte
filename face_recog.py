import numpy as np
import math
import os
from src.analyze_face.analysis import FaceAnalysis

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class FaceRecognizer:
    def __init__(self):
        self.model = FaceAnalysis(name='analg_face', root='.')
        self.model.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
        self.expansion_rate = [0.1, 0.4, 0.1, 0.1]

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

    def crop_face(self, image, box):
        frameW = image.shape[1]
        framwH = image.shape[0]
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        w = x2 - x1
        h = y2 - y1

        cnt_x = x1 + int(w / 2)
        cnt_y = y1 + int(h / 2)

        x_margine = int(w * 0.6)
        y_margine = int(h * 0.6)

        x1 = max(0, cnt_x - x_margine)
        x2 = min(frameW, cnt_x + x_margine)

        y1 = max(0, cnt_y - y_margine)
        y2 = min(framwH, cnt_y + y_margine)

        face_img = image[y1:y2, x1:x2]

        return face_img

    def predict(self, image):
        faces = self.model.get(image)
        # emb = faces[0].embedding
        # box = faces[0].bbox
        # face = self.crop_face(image, box)
        return faces


if __name__ == '__main__':
    import cv2
    recog = FaceRecognizer()
    img = cv2.imread("face.jpg")
    img = cv2.resize(img, (600, 500))
    faces = recog.predict(img)
    if len(faces) > 0:
        for face in faces:
            box = face.bbox
            emb = face.embedding

            print(emb)
