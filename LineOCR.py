from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import PIL
from paddleocr import PaddleOCR,draw_ocr
from PIL import Image, ImageDraw
import cv2
import numpy as np
import PredLineDet
from IPython.display import display
from concurrent.futures.thread import ThreadPoolExecutor
import time
import re
from turtle import fillcolor
import itertools
import copy
from scipy import ndimage

# config = Cfg.load_config_from_name("vgg_seq2seq")
# config.save("default_vgg_seq2seq_config.yml")
class ProcessImage():
    '''
    !python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
    !pip install "paddleocr>=2.0.1"

    !pip uninstall imgaug
    !pip install imgaug==0.4.0
    !pip install --quiet vietocr
    '''
    # /content/drive/MyDrive/PPY/DischargePaper/LineDetection/Paddle/Pretrained/exported_det_model_221011/
    def __init__(self,
                 detection_model_path = "./PaddleOCR/pretrained_models/det_r50_td_tr_inference_221206/",
                 text_recognition_model_config_file = "./default_vgg_seq2seq_config.yml",
                 text_recognition_model_path = "./weights/ocr/line_ocr_220930_3.pth"):
        # self.line_detector = PaddleOCR(use_angle_cls=True, lang='en', det_model_dir = detection_model_path, use_gpu = True, det_algorithm="PSE")
        self.line_detector = PredLineDet.LineDetInfer(det_model_dir = detection_model_path)
        
        # config = Cfg.load_config_from_file(text_recognition_model_config_file)
        config = Cfg.load_config_from_file(text_recognition_model_config_file)

        config.weights = text_recognition_model_path
        self.text_recognizer = Predictor(config)
    

    def text_line_equalization(self, image):
        """
        Doesn't matter about the type of image. You can put into this function any type of image (PIL or CV2).
        """
        is_type_pil = True
        if type(image) == np.ndarray:
            is_type_pil = False
            image_cv2 = image
        else:
            image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        image_gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
        equalized_image = clahe.apply(image_gray)
        image_cv2 = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

        if is_type_pil:
            image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
            return image_pil
        else:
            return image_cv2

    def detect_lines(self, img, vis = False):
        if type(img) != np.ndarray:
            img = np.array(img)

        # lines = self.line_detector.ocr(img, det=True, cls=False, rec=False)
        lines = self.line_detector.line_det_infer(img)
        isClosed = True
        color = (0, 255, 0)
        thickness = 2
        if vis == True:
            drawed_img = img.copy()
            for line in lines:
                pts = np.array(line)
                drawed_img = cv2.polylines(drawed_img, np.int32([pts]), isClosed, color,thickness)
            display(Image.fromarray(drawed_img))
        return lines[0]
    
    def crop_line(self, rotated_img, points, convert_gray):
        original = Image.fromarray(rotated_img.copy())
        
        polygon = []
        for pair in points:
            polygon.append(tuple(map(int,pair)))

        list_x = [x for x,y in polygon]
        list_y = [y for x,y in polygon]
        xmin = min(list_x)
        xmax = max(list_x)
        ymin = min(list_y)
        ymax = max(list_y)

        mask = Image.new("L", original.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygon, fill=255, outline=None)
        white =  Image.new("L", original.size, 255)
        result = Image.composite(original, white, mask)
        croped = original.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
        if convert_gray == True:
            croped = self.text_line_equalization(croped)
        return croped, (polygon, xmin, xmax, ymin, ymax)

    def crop_line_v2(self, rotated_img, raw_points, convert_gray):
        points = copy.deepcopy(raw_points)
        # original = Image.fromarray(rotated_img.copy())
        img_height, img_width, channels = rotated_img.shape
        original = rotated_img.copy()
        
        
        #expand height
        points[0][1] = int(points[0][1]) - 5
        if points[0][1] <= 0:
            points[0][1] = 0
        
        points[1][1] = int(points[1][1]) - 5
        if points[1][1] < 0:
            points[1][1] = 0
        
        points[2][1] = int(points[2][1]) + 5
        if points[2][1] > img_height:
            points[2][1] = img_height
        
        points[3][1] = int(points[3][1]) + 5
        if points[3][1] > img_height:
            points[3][1] = img_height
        
        
        #expand width
        points[0][0] = int(points[0][0]) - 5 #top_left -5
        if points[0][0] <= 0:
            points[0][0] = 0
            
        points[1][0] = int(points[1][0]) + 5  #top_right +5
        if points[1][0] > img_width:
            points[1][0] = img_width
            
        points[2][0] = int(points[2][0]) + 5 #bottom_right +5
        if points[2][0] > img_width:
            points[2][0] = img_width
            
        points[3][0] = int(points[3][0]) - 5 #bottom_left -5
        if points[3][0] <= 0:
            points[3][0] = 0
        
        
        polygon = []

        for pair in points:
            polygon.append(tuple(map(int,pair)))
            
        raw_polygon = []
        for pair in raw_points:
            raw_polygon.append(tuple(map(int,pair)))

        list_x = [x for x,y in polygon]
        list_y = [y for x,y in polygon]
        xmin = min(list_x)
        xmax = max(list_x)
        ymin = min(list_y)
        ymax = max(list_y)



        mask = np.ones(original.shape, dtype=np.uint8)*255
        # roi_corners = np.array([[(10,10), (300,300), (10,300)]], dtype=np.int32)
        roi_corners = [polygon]
        roi_corners = np.array(roi_corners)
        # fill the ROI so it doesn't get wiped out when the mask is applied
        channel_count = original.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (0,)*channel_count
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)
        # from Masterfool: use cv2.fillConvexPoly if you know it's convex

        # apply the mask
        masked_image = cv2.bitwise_or(original, mask)

        croped_mask = masked_image[ymin:ymax, xmin:xmax]
        if convert_gray == True:
            croped_mask = self.text_line_equalization(croped_mask)
        pil_croped_mask = Image.fromarray(croped_mask)
        return pil_croped_mask, (raw_polygon, polygon, xmin, xmax, ymin, ymax)

    def extract_lines(self, lines, rotated_img, convert_gray = True):
        extracted_dict = {
            "bbox": [],
            "polygon": [],
            "croped_pil_img": [],
        }
        for line in lines:
            croped, _points = self.crop_line_v2(rotated_img, line, convert_gray)
            raw_polygon, polygon, xmin, xmax, ymin, ymax = _points
            croped = self.single_rotate_and_transpose(np.array(croped), polygon)
            croped = Image.fromarray(croped)
            extracted_dict["polygon"].append(raw_polygon)
            extracted_dict["bbox"].append((int(xmin), int(ymin), int(xmax), int(ymax)))
            extracted_dict["croped_pil_img"].append(croped)
        return extracted_dict

    def ocr_single_task(self, i, extracted_dict):
        croped = extracted_dict["croped_pil_img"][i]
        text, prob = self.text_recognizer.predict(croped, return_prob=True)
        text = re.sub(' +', ' ', text)
        extracted_dict["text"][i] = text
        extracted_dict["ocr_score"][i] = prob

    def recognize_text(self, extracted_dict):
        data_len = len(extracted_dict["polygon"])
        # print(data_len)
        extracted_dict["text"] = [""]*data_len
        extracted_dict["ocr_score"] = [0.]*data_len
        with ThreadPoolExecutor (max_workers=100) as executor:
            for i in range(data_len):
                executor.submit(self.ocr_single_task, i, extracted_dict)
        return extracted_dict

    def begin_recognize_text(self, img):
        lines = self.detect_lines(img)
        s_time = time.time()
        extracted_dict = self.extract_lines(lines,img)
        print("extracted_line time:", time.time() - s_time)
        r_time = time.time()
        extracted_dict = self.recognize_text(extracted_dict)
        print("ocr_line time:", time.time() - r_time)
        return extracted_dict

    def individual_OCR(self, pil_line_img):
        text, prob = self.text_recognizer.predict(pil_line_img, return_prob=True)
        return text, prob
    
    def rec_transpose(self, rotate_angle, raw_cv2_img, rotated_cv2_image, rec):
        raw_img = raw_cv2_img.copy()
        raw_height, raw_width = raw_img.shape[:2]
        x_cen = round(raw_width/2)
        y_cen = round(raw_height/2)
        rotated_rec = []
        # rotated_cv2_image = ndimage.rotate(raw_img, -rotate_angle, cval=255)
        new_height, new_width = rotated_cv2_image.shape[:2]
        new_height_cen = round(new_height/2)
        new_width_cen = round(new_width/2)
        for each in rec:
            cur_x, cur_y = each[:]
            r_y = y_cen - cur_y
            r_x = x_cen - cur_x
            a1 = np.arctan2(r_y, r_x)
            d = np.sqrt(r_y**2 + r_x**2)
            a2 = a1 + np.radians(rotate_angle)
            r_y2 = d * np.sin(a2)
            r_x2 = d * np.cos(a2)
            x2 = round(new_width_cen - r_x2)
            y2 = round(new_height_cen - r_y2)
            if x2 < 0:
                print("ERROR x=", x2)
                x2 = 0
            if y2 < 0:
                print("ERROR y=", y2)
                y2 = 0
            rotated_rec.append([x2, y2])
        return rotated_rec

    def angle_between(self, p1, p2):
        x1,y1 = p1
        x2,y2 = p2
        tan_a = (y2 - y1)/(x2-x1)
        angle = np.arctan(tan_a)
        return np.rad2deg(angle)

    def single_rotate_and_transpose(self, raw_img, points):
        p1, p2, p3, p4 = points
        angle = self.angle_between(p3, p4)
        rotate_angle = angle
        rorated_img = ndimage.rotate(raw_img.copy(), rotate_angle, cval=255)
        temp_rotated_img = rorated_img.copy()
        rec = points
        xs = [each[0] for each in points]
        ys = [each[1] for each in points]
        xmin = min(xs)
        ymin = min(ys)
        xs = [each - xmin for each in xs]
        ys = [each -ymin for each in ys]
        rec = [[x,y] for x,y in zip(xs, ys)]
        # print(rec)
        new_rec = self.rec_transpose(-rotate_angle, raw_img.copy(), temp_rotated_img.copy(), rec)
        xs = [each[0] for each in new_rec]
        ys = [each[1] for each in new_rec]
        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)
        roi = temp_rotated_img[ymin:ymax, xmin:xmax]
        return roi

    def rotate_and_transpose(self, raw_img, anno_value, rotate_angle, img_root_dir = "data/"):
        rorated_img = ndimage.rotate(raw_img.copy(), -rotate_angle, cval=255)
        temp_rotated_img = rorated_img.copy()
        temp_raw_img = raw_img.copy()
        transposed_anno = copy.deepcopy(anno_value)
        for idx, each in enumerate(anno_value):
            rec = np.array(each["points"])
            
            temp_raw_img = cv2.polylines(temp_raw_img, [rec],
                            True, (255,0,0), 3)
            new_rec = self.rec_transpose(rotate_angle, raw_img.copy(), temp_rotated_img.copy(), rec)
            new_rec = np.array(new_rec)
            # temp_rotated_img = cv2.polylines(temp_rotated_img, [new_rec],
            #                 True, (255,0,0), 3)
            transposed_anno[idx]["points"] = new_rec.tolist()
        return rorated_img, transposed_anno
