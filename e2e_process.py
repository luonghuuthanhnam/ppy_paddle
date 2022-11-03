import cv2

import preprocess_img
import LineOCR
import pandas as pd
import numpy as np
import kie_gcn
from kie_gcn import InvoiceGCN
import postprocess
import gc

# preprocessImage = preprocess_img.PreprocessImage()
# lineDetAndOCR = LineOCR.ProcessImage()
# kieGCN = kie_gcn.KieGCN()

# test_img = cv2.imread(r"imgs_test\705.jpeg")
# test_img = preprocessImage(test_img)

# result = lineDetAndOCR.begin_recognize_text(test_img)
# kie_df = kieGCN(result, test_img)

# kie_df.to_excel("result.xlsx")

class E2E_OCR_Engine():
    def __init__(self,
            detection_model_path = "PaddleOCR/pretrained_models/exported_det_model_221011/",
            text_recognition_model_path = "./weights/ocr/line_ocr_220930_3.pth",
            gcn_state_dict_path = "weights/gcn/GCN_221103_state_dict.pth") -> None:
        self.preprocessImage = preprocess_img.PreprocessImage()
        self.lineDetAndOCR = LineOCR.ProcessImage(detection_model_path=detection_model_path, text_recognition_model_path = text_recognition_model_path)
        # self.kieGCN = kie_gcn.KieGCN(gcn_model_path=gcn_model_path)
        self.kieGCN = kie_gcn.KieGCN_v2(PhoBERT_base_fairseq_dir="weights/nlp/PhoBERT_base_fairseq",
                                        PhoBERT_trained_state_dict_path="weights/nlp/phoBert_trained_state_dict/phoBert_state_dict_221101.pth",
                                        gcn_state_dict_path=gcn_state_dict_path)
        self.kiePostprocess = postprocess.KiePostProcess()
        self.empty_extracted_result = {
            "hospital_name": None,
            "patient_name": None,
            "age": None,
            "gender": None,
            "admissiion_date": [],
            "discharge_date": [],
            "sign_date": [],
            "icd-10": [],
        }

    def extract_discharge_paper_info(self, cv2_img):
        process_img = cv2_img.copy()
        process_img = self.preprocessImage(process_img)
        det_ocr_result = self.lineDetAndOCR.begin_recognize_text(process_img)
        kie_df = self.kieGCN(det_ocr_result, process_img)
        return kie_df
    
    def postprocess_kie_df(self, kie_df):
        self.kiePostprocess.append_kie_df(kie_df)

        hospital_name = self.kiePostprocess.hospital_name_postprocess()
        if hospital_name == None:
            hospital_name = self.kiePostprocess.find_hospital_name_remain()
        # print("hospital_name:", hospital_name)

        patient_name = self.kiePostprocess.patient_name_postprocess()
        if patient_name == None:
            patient_name = self.kiePostprocess.find_patient_name_remain()
        # print("patient_name:", patient_name)

        age, temp_gender = self.kiePostprocess.age_postprocess()
        gender, temp_age = self.kiePostprocess.gender_postprocess()
        if age == None and temp_age != None:
            age = temp_age
        if gender == None and temp_gender != None:
            gender = temp_gender

        if age == None:
            age, temp_gender = self.kiePostprocess.find_age_remain()
        # print("gender:", gender)
        # print("age:", age)

        admissiion_dates = self.kiePostprocess.admission_date_postprocess()
        # for each in admissiion_dates:
            # print("admissiion_date:", each)

        discharge_dates = self.kiePostprocess.discharge_date_postprocess()
        # for each in discharge_dates:
            # print("discharge_date:", each)

        admissiion_dates, discharge_dates = self.kiePostprocess.admission_discharge_correction(admissiion_dates, discharge_dates)

        sign_dates = self.kiePostprocess.sign_date_postprocess()
        # for each in sign_dates:
            # print("sign_date:", each)


        ICD_codes = self.kiePostprocess.ICD_code_postprocess()
        # print("ICD_codes:", ICD_codes)

        extracted_result = self.empty_extracted_result.copy()
        if hospital_name != "":
            extracted_result["hospital_name"] = hospital_name
        if patient_name != "":
            extracted_result["patient_name"] = patient_name
        if gender != "":
            extracted_result["gender"] = gender
        if age != "":
            extracted_result["age"] = age
        extracted_result["admissiion_date"] = admissiion_dates
        extracted_result["discharge_date"] = discharge_dates
        extracted_result["sign_date"] = sign_dates
        extracted_result["icd-10"] = ICD_codes
        return extracted_result

    def __call__(self, cv2_img):
        # cv2_img = cv2.imread(img_path)
        extracted_df = self.extract_discharge_paper_info(cv2_img)
        extracted_result = self.postprocess_kie_df(extracted_df)
        gc.collect()
        return extracted_result, extracted_df