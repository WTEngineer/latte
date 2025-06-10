import cv2
import numpy as np
import mediapipe as mp


class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.99,
        )
        # self.expansion_rate = [0.1, 0.4, 0.1, 0.0]
        self.expansion_rate = [0.0, 0.2, 0.0, 0.0]

    def calc_angle(self, pt1, pt2, pt3):
        a = np.array(pt1)
        b = np.array(pt2)
        c = np.array(pt3)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return int(np.degrees(angle))

    def get_box_kpt(self, image_width, image_height, res):
        bboxes = []
        keypoints = []
        for detection in res.detections:
            keypoint0 = detection.location_data.relative_keypoints[0]
            keypoint0_x = int(keypoint0.x * image_width)
            keypoint0_y = int(keypoint0.y * image_height)

            keypoint1 = detection.location_data.relative_keypoints[1]
            keypoint1_x = int(keypoint1.x * image_width)
            keypoint1_y = int(keypoint1.y * image_height)

            keypoint2 = detection.location_data.relative_keypoints[2]
            keypoint2_x = int(keypoint2.x * image_width)
            keypoint2_y = int(keypoint2.y * image_height)

            keypoint3 = detection.location_data.relative_keypoints[3]
            keypoint3_x = int(keypoint3.x * image_width)
            keypoint3_y = int(keypoint3.y * image_height)

            keypoint4 = detection.location_data.relative_keypoints[4]
            keypoint4_x = int(keypoint4.x * image_width)
            keypoint4_y = int(keypoint4.y * image_height)

            keypoint5 = detection.location_data.relative_keypoints[5]
            keypoint5_x = int(keypoint5.x * image_width)
            keypoint5_y = int(keypoint5.y * image_height)

            keypoints.append([
                [keypoint0_x, keypoint0_y],
                [keypoint1_x, keypoint1_y],
                [keypoint2_x, keypoint2_y],
                [keypoint3_x, keypoint3_y],
                [keypoint4_x, keypoint4_y],
                [keypoint5_x, keypoint5_y],
            ])

            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * image_width)
            y1 = int(bbox.ymin * image_height)
            w = int(bbox.width * image_width)
            h = int(bbox.height * image_height)
            x1 = x1 - int(w * self.expansion_rate[0])
            y1 = y1 - int(h * self.expansion_rate[1])
            x2 = x1 + w + int(w * self.expansion_rate[0]) + int(
                w * self.expansion_rate[2])
            y2 = y1 + h + int(h * self.expansion_rate[1]) + int(
                h * self.expansion_rate[3])

            x1 = np.clip(x1, 0, image_width)
            y1 = np.clip(y1, 0, image_height)
            x2 = np.clip(x2, 0, image_width)
            y2 = np.clip(y2, 0, image_height)

            bboxes.append([x1, y1, x2, y2])

        return bboxes, keypoints

    def check_frontal(self, keypoints):
        if len(keypoints) == 6:
            angR = self.calc_angle(keypoints[0], keypoints[1], keypoints[2])
            angL = self.calc_angle(keypoints[1], keypoints[0], keypoints[2])

            if abs(angR - angL) < 100:
                predLabel = 'Frontal'
            else:
                predLabel = 'NotFrontal'
        else:
            predLabel = 'NotFrontal'

        return predLabel

    def draw_bbox(self, image, bbox, frontal_val):
        if frontal_val == 'Frontal':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        return image

    def draw_kpts(self, img, kpts):
        cv2.circle(img, kpts[0], 5, (0, 0, 0), thickness=1)
        cv2.circle(img, kpts[1], 5, (0, 255, 0), thickness=1)
        cv2.circle(img, kpts[2], 5, (255, 255, 255), thickness=1)
        cv2.circle(img, kpts[3], 5, (255, 0, 0), thickness=1)
        cv2.circle(img, kpts[4], 5, (255, 0, 255), thickness=1)
        cv2.circle(img, kpts[5], 5, (0, 255, 255), thickness=1)
        return img

    def crop_face(self, img, box):
        margin = int((box[3] - box[1]) * 0.3)
        w = img.shape[1]
        h = img.shape[0]
        s_x = max(0, box[0] - margin)
        e_x = min(w, box[2] + margin)
        s_y = max(0, box[1] - margin)
        e_y = min(h, box[3] + margin)
        face = img[s_y:e_y, s_x:e_x]

        return face

    def detect(self, image):
        img_return = image.copy()
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.detector.process(input_image)

        image_width, image_height = image.shape[1], image.shape[0]

        frontal_faces = []

        if results.detections is not None:

            bboxes,  keypoints= self.get_box_kpt(image_width, image_height, results)

            for i in range(0, len(keypoints)):
                box = bboxes[i]
                kpt = keypoints[i]
                try:
                    frontal_val = self.check_frontal(kpt)
                    if frontal_val == 'Frontal':
                        face = self.crop_face(image, box)
                        frontal_faces.append(face)

                    # ---- drawing functions ----
                    img_return = self.draw_bbox(img_return, box, frontal_val)
                except:
                    print("No real person face.")

            return img_return, frontal_faces
        else:
            return img_return, frontal_faces


if __name__ == '__main__':
    # cap = cv2.VideoCapture('sample1.mp4')
    cap = cv2.VideoCapture(0)
    det = FaceDetector()
    while True:
        ret, image = cap.read()
        if not ret:
            break

        image, faces = det.detect(image)
        if image is not None:
            image = cv2.resize(image, (800, 600))
            cv2.imshow("res", image)
        cv2.waitKey(1)






