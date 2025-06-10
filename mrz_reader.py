import cv2
import ctypes
import json
from ctypes import *
import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class MRZReader:
    def __init__(self):
        lib = ctypes.windll.LoadLibrary(ROOT_DIR + "/MRZscan.dll")
        # lib = cdll.LoadLibrary(ROOT_DIR +"/liblibmrz.so")
        a = ROOT_DIR +"/models/mMQDF_f_Passport_bottom_Gray.dic"
        b = ROOT_DIR +"/models/mMQDF_f_Passport_bottom.dic"

        # lib.InitModel(a.encode('ascii'), b.encode('ascii'))
        lib.InItEngine(a.encode('ascii'), b.encode('ascii'))

        # self.mrz_reader = lib.GetInfo
        self.mrz_reader = lib.ScanMRZ_Mode2

        #self.mrz_reader.argtypes = [c_char_p]
        self.mrz_reader.restype = c_char_p


    def read_mrz(self, image):
        # image = cv2.imread(path)
        w = image.shape[1]
        h = image.shape[0]

        b = bytes(image)

        re = self.mrz_reader(b, w, h)
        if re is not None:
            re = re.decode("utf-8")

            list = re.split(',')

            result = {}

            for ln in list:
                if "Surname" in ln:
                    surname = ln.split(':')[1].replace('"', '')
                    result['last_name'] = surname
                if "Givename" in ln:
                    givename = ln.split(':')[1].replace('"', '')
                    result['first_name'] = givename
                if "Birth" in ln:
                    birth = ln.split(':')[1].replace('"', '')
                    if len(birth) == 6:
                        result['date_of_birth'] = birth

                if "ExpirationDate" in ln:
                    expir_date = ln.split(':')[1].replace('"', '')
                    if len(expir_date) == 6:
                        result['date_of_expire'] = expir_date

            return result

        else:
            return None


def deblur_wiener(image, kernel, noise_var=1e-2):
    # Ensure the kernel has the same number of channels as the image
    if kernel.shape[-1] < image.shape[-1]:
        kernel = np.tile(kernel[:, :, np.newaxis], (1, 1, image.shape[-1]))

    # Apply Wiener filter for deblurring
    psf = np.fft.fft2(kernel, s=image.shape[:2])  # Use only the first two dimensions of the image shape
    img_fft = np.fft.fft2(image, s=image.shape[:2])
    result_fft = np.conj(psf) / (np.abs(psf) ** 2 + noise_var)

    # Reshape the arrays for compatibility during multiplication
    result_fft_reshaped = result_fft.reshape(result_fft.shape)
    img_fft_reshaped = img_fft.reshape(img_fft.shape)

    result_fft_reshaped *= img_fft_reshaped

    deblurred_image_fft = result_fft_reshaped

    # Inverse Fourier transform to obtain the deblurred image
    deblurred_image = np.abs(np.fft.ifft2(deblurred_image_fft))

    # Normalize the deblurred image
    deblurred_image = cv2.normalize(deblurred_image, None, 0, 255, cv2.NORM_MINMAX)

    return deblurred_image.astype(np.uint8)


DIM=(1920, 1080)
K=np.array([[1395.385190093491, 0.0, 988.7998276275356], [0.0, 1375.1430223190368, 615.1286086483378], [0.0, 0.0, 1.0]])
D=np.array([[0.023490310637959862], [-0.12407003043059696], [-0.47419207100720445], [1.0547016617558373]])


def un_distort(img):
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


if __name__ == '__main__':
    reader = MRZReader()


    # files = os.listdir('cali_doc')
    #
    # for file in files:
    #     path = os.path.join('cali_doc', file)
    #     img = cv2.imread(path)
    #     img = un_distort(img)
    #     res = reader.read_mrz(img)
    #     print(res)
    #     cv2.imshow("res", img)
    #     cv2.waitKey(0)

    # image = cv2.imread('1.jpg')
    #
    # mrz_result = reader.read_mrz(image)
    # print(mrz_result)
    # cv2.imshow('res', image)
    # cv2.waitKey(0)

    cap = cv2.VideoCapture(0)

    while True:
        ret, image = cap.read()

        # # Define a kernel with the same number of channels as the image
        # kernel_size = 5
        # kernel = np.ones((kernel_size, kernel_size, image.shape[-1]), np.float32) / (kernel_size * kernel_size)
        #
        # # Deblur the image using the Wiener filter
        # deblurred_image = deblur_wiener(image, kernel)

        # img = cv2.resize(deblurred_image, (800, 600))
        mrz_result = reader.read_mrz(image)
        print(mrz_result)
        cv2.imshow('res', image)
        cv2.waitKey(1)


