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
                 detection_model_path = "./PaddleOCR/pretrained_models/exported_det_model_221011/",
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
    
    def extract_lines(self, lines, rotated_img, convert_gray = True):
        extracted_dict = {
            "bbox": [],
            "polygon": [],
            "croped_pil_img": [],
        }
        for line in lines:
            original = Image.fromarray(rotated_img.copy())
            polygon = []
            for pair in line:
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
            extracted_dict["polygon"].append(polygon)
            extracted_dict["bbox"].append((int(xmin), int(ymin), int(xmax), int(ymax)))
            extracted_dict["croped_pil_img"].append(croped)
        return extracted_dict

    def ocr_single_task(self, i, extracted_dict):
        croped = extracted_dict["croped_pil_img"][i]
        # croped = self.text_line_equalization(croped)
        text, prob = self.text_recognizer.predict(croped, return_prob=True)
        extracted_dict["text"][i] = text
        extracted_dict["ocr_score"][i] = prob

    def recognize_text(self, extracted_dict):
        data_len = len(extracted_dict["polygon"])
        # print(data_len)
        extracted_dict["text"] = [""]*data_len
        extracted_dict["ocr_score"] = [0.]*data_len
        with ThreadPoolExecutor (max_workers=100) as executor:
            for i in range(data_len):
                # croped = extracted_dict["croped_pil_img"][i]
                # # croped = self.text_line_equalization(croped)

                # text, prob = self.text_recognizer.predict(croped, return_prob=True)
                # extracted_dict["text"][i] = text
                # extracted_dict["ocr_score"][i] = prob
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
