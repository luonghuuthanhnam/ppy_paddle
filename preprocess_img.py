import cv2
import numpy as np
import glob
from scipy import ndimage
import math
import time
from PIL import Image, ImageDraw

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