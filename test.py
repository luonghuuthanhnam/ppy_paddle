from gzip import _PaddedFile
from PaddleOCR.tools.infer import predict_det

from PaddleOCR.ppocr.utils.utility import get_image_file_list, check_and_read
import PaddleOCR.tools.infer.utility as utility
import os
from PaddleOCR.ppocr.utils.logging import get_logger
logger = get_logger()
import cv2
import numpy as np
import time
import json

class LineDetInfer():
    def __init__(self) -> None:
        self.args = utility.parse_args()
        self.args.det_model_dir = "PaddleOCR/pretrained_models/exported_det_model_221011/"
        self.args.det_algorithm = "PSE"
        self.text_detector = predict_det.TextDetector(self.args)


    def line_det_multi_inference(self):
        self.args.image_dir = "PaddleOCR/img_file"
        image_file_list = get_image_file_list(self.args.image_dir)
        count = 0
        total_time = 0
        draw_img_save = "./inference_results"

        if self.args.warmup:
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(2):
                res = self.text_detector(img)

        if not os.path.exists(draw_img_save):
            os.makedirs(draw_img_save)
        save_results = []
        for image_file in image_file_list:
            img, flag, _ = check_and_read(image_file)
            if not flag:
                img = cv2.imread(image_file)
            if img is None:
                logger.info("error in loading image:{}".format(image_file))
                continue
            st = time.time()
            dt_boxes, _ = self.text_detector(img)
            elapse = time.time() - st
            if count > 0:
                total_time += elapse
            count += 1
            save_pred = os.path.basename(image_file) + "\t" + str(
                json.dumps([x.tolist() for x in dt_boxes])) + "\n"
            save_results.append(save_pred)
            logger.info(save_pred)
            logger.info("The predict time of {}: {}".format(image_file, elapse))
            src_im = utility.draw_text_det_res(dt_boxes, image_file)
            img_name_pure = os.path.split(image_file)[-1]
            img_path = os.path.join(draw_img_save,
                                    "det_res_{}".format(img_name_pure))
            cv2.imwrite(img_path, src_im)
            logger.info("The visualized image saved in {}".format(img_path))

        with open(os.path.join(draw_img_save, "det_results.txt"), 'w') as f:
            f.writelines(save_results)
            f.close()
        if self.sargs.benchmark:
            self.text_detector.autolog.report()

    def line_det_infer(self, image_file_path, draw_result = False):
        count = 0
        draw_img_save = "./inference_results"
        if not os.path.exists(draw_img_save):
            os.makedirs(draw_img_save)
        image_file = image_file_path
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
        else:
            st = time.time()
            dt_boxes, _ = self.text_detector(img)
            elapse = time.time() - st
            if count > 0:
                total_time += elapse
            count += 1
            save_pred = os.path.basename(image_file) + "\t" + str(
                json.dumps([x.tolist() for x in dt_boxes])) + "\n"
            logger.info(save_pred)
            logger.info("The predict time of {}: {}".format(image_file, elapse))
            if draw_result:
                src_im = utility.draw_text_det_res(dt_boxes, image_file)
                img_name_pure = os.path.split(image_file)[-1]
                img_path = os.path.join(draw_img_save,
                                        "det_res_{}".format(img_name_pure))
                cv2.imwrite(img_path, src_im)
                logger.info("The visualized image saved in {}".format(img_path))
                
                

                with open(os.path.join(draw_img_save, "det_results.txt"), 'w') as f:
                    f.writelines(save_pred)
                    f.close()
            return dt_boxes, _

lineDetInfer = LineDetInfer()
# lineDetInfer.line_det_multi_inference()
dt_boxes, _  = lineDetInfer.line_det_infer(r"D:\PPYCode\OCR\ppy_paddle\PaddleOCR\img_file\test.jpg")
print(dt_boxes)
print("-------FFFFFFFFFFFFFFFFFFFFFFF----")
print(_)