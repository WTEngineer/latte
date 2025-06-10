import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F
from src.analyze_face.loaders.fd import FD

from src.model_lib.minnetV1 import minnetV1, minnetV2,minnetV1SE,minnetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name


MODEL_MAPPING = {
    'minnetV1': minnetV1,
    'minnetV2': minnetV2,
    'minnetV1SE':minnetV1SE,
    'minnetV2SE':minnetV2SE
}

class Detection:

    def __init__(self):
        onnx_file = "./models/detection/det_500m_480x480.onnx"
        self.detector = FD(model_file=onnx_file)
        self.detector.det_thresh = 0.15

    def get_face(self, img, max_num=0):
        bboxes, kpss = self.detector.detect(img, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return []

        biggest_idx = 0
        biggest_area = (bboxes[0, 2] - bboxes[0, 0]) * (bboxes[0, 3] - bboxes[0, 1])
        for i in range(1, bboxes.shape[0]):
            area = (bboxes[i, 2] - bboxes[i, 0]) * (bboxes[i, 3] - bboxes[i, 1])
            if area > biggest_area:
                biggest_area = area
                biggest_idx = i
                

        bbox = bboxes[biggest_idx, 0:4]
        return bbox

    def get_bbox(self, img):
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        input_size = [480, 480]
        new_height = input_size[1]
        new_width = input_size[0]
        exp_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        exp_scale = 1.0

        if img.shape[0] >= img.shape[1] and 480 < img.shape[0]:
            half_h = 480
            half_w = int(img.shape[1] * 480 / img.shape[0])
            exp_scale = 480.0 / img.shape[0]
            half_img = cv2.resize(img, (half_w, half_h), interpolation=cv2.INTER_AREA)
            x_pos = int(new_height / 2 - half_img.shape[0] / 2)
            y_pos = int(new_width / 2 - half_img.shape[1] / 2)
            exp_img[x_pos: x_pos + half_img.shape[0], y_pos: y_pos + half_img.shape[1], :] = half_img

        elif img.shape[0] <= img.shape[1] and 480 < img.shape[1]:
            half_h = int(img.shape[0] * 480 / img.shape[1])
            half_w = 480
            exp_scale = 480.0 / img.shape[1]
            half_img = cv2.resize(img, (half_w, half_h), interpolation=cv2.INTER_AREA)
            x_pos = int(new_height / 2 - half_img.shape[0] / 2)
            y_pos = int(new_width / 2 - half_img.shape[1] / 2)
            exp_img[x_pos: x_pos + half_img.shape[0], y_pos: y_pos + half_img.shape[1], :] = half_img

        else:
            x_pos = int(new_height / 2 - img.shape[0] / 2)
            y_pos = int(new_width / 2 - img.shape[1] / 2)
            exp_img[x_pos: x_pos + img.shape[0], y_pos: y_pos + img.shape[1], :] = img

        face = self.get_face(exp_img)
        if len(face) == 0:
            return []

        face[0] = (face[0] - y_pos) / exp_scale
        face[1] = (face[1] - x_pos) / exp_scale
        face[2] = (face[2] - y_pos) / exp_scale
        face[3] = (face[3] - x_pos) / exp_scale
        face = [int(face[0]), int(face[1]), int(face[2]-face[0]+1), int(face[3]-face[1]+1)]

        return face


class LivenessDetect(Detection):
    def __init__(self, model_path):
        super(LivenessDetect, self).__init__()
        # self.device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self._load_model(model_path)

    def _load_model(self, model_path):
        model_name = os.path.basename(model_path)
        h_input, w_input, type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input,)
        self.model = MODEL_MAPPING[type](conv6_kernel=self.kernel_size, num_classes=3).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                if name_key.startswith('FT'):
                    continue
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)

        return None

    def eval(self, img):
        trans2 = trans.Compose([trans.ToTensor(),])
        #print(self.device)
        img = trans2(img)
        img = img.unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result, dim=None).cpu().numpy()
        return result
