import cv2
import numpy as np
import glob
from scipy import ndimage
import math
import time
from PIL import Image, ImageDraw
import os

import boto3
import time
from pdf2image import convert_from_path


class DownAndLoadImage():
    def __init__(self, bucket_name) -> None:
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        self.SUPPORTED_IMG_TYPE = [
            "jpg",
            "jpeg",
            "png"
        ]

    def load_img(self, file_path):
        loaded_img = None
        temp = file_path.split(".")
        main_file_name = ".".join(temp[:-1])
        ext = temp[-1]
        
        if ext.lower() == "pdf":
            loaded_img = self.convert_pdf2img(file_path)
        elif ext.lower() in self.SUPPORTED_IMG_TYPE:
            loaded_img = cv2.imread(file_path)
        else:
            print(f"Unsupported file type: *{ext}")
        return loaded_img
    
    def convert_pdf2img(self, file_path):
        images = convert_from_path(file_path)
        cv2_img = np.array(images[0])
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        return cv2_img
        
    def down_img(self, object_name, save_dir = "temp_data/download/"):
        saved_file_path = save_dir+object_name
        with open(saved_file_path, "wb") as f:
            self.s3.download_fileobj('papaya-fwd-prod-stp', object_name, f)
        return saved_file_path 
    
    def __call__(self, object_name):
        file_path = self.down_img(object_name)
        cv2_img = self.load_img(file_path=file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        return cv2_img
    
class PreprocessImage():
    '''
    Resize, Correct Orientation, Straighten image input
    '''
    def __init__(self):
        pass

    def resize_image(self, input_image):
        """
        Doesn't matter about the type of image. You can put into this function any type of image (PIL or CV2).
        """
        image = input_image.copy()
        is_type_pil = True
        if type(image) == np.ndarray:
            is_type_pil = False

        ratio = 1
        if not is_type_pil:
            width, height = image.shape[1], image.shape[0]
        else:
            width, height = image.size

        if height > 1500:
            expected_height = 1280
            ratio = expected_height / (height + 0.01)
            width = int(width * ratio)
            height = int(height * ratio)
            dim = (width, height)

            if is_type_pil:
                image = image.resize(dim, Image.ANTIALIAS)
            else:
                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        return image, ratio

    def auto_rotate(self, input_image):
        '''
            Check slope of all horizontal lines and calcualte the median slope -> rotate the image to straighten it
        '''
        image = input_image.copy()
        is_type_pil = True
        if type(image) == np.ndarray:
            is_type_pil = False
            image_cv2 = image
        else:
            image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        image_gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        image_edges = cv2.Canny(image_gray, 100, 100, apertureSize=3)
        lines = cv2.HoughLinesP(
            image_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=10
        )
        reshaped_lines = lines.reshape(lines.shape[0], lines.shape[2])
        angles = np.degrees(
            np.arctan2(
                reshaped_lines[:, 3] - reshaped_lines[:, 1],
                reshaped_lines[:, 2] - reshaped_lines[:, 0],
            )
        )
        median_angle = np.median(angles)

        if abs(median_angle) < 45:
            rotated_image = ndimage.rotate(image_cv2, median_angle, cval=255)
        else:
            rotated_image = image_cv2

        if is_type_pil:
            image_pil = Image.fromarray(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
            return image_pil, median_angle
        else:
            return rotated_image, median_angle

    def __call__(self, raw_img) -> np.array:
        img = raw_img.copy()
        img, ratio = self.resize_image(input_image=img)
        img, median_angle = self.auto_rotate(input_image=img)
        return img