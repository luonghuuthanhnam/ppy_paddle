import cv2
import unidecode
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
        # self.empty_extracted_result = {
        #     "hospital_name": None,
        #     "hospital_name_score": -1,
        #     "patient_name": None,
        #     "patient_name_score": -1,
        #     "age": None,
        #     "age_score": -1,
        #     "gender": None,
        #     "gender_score": -1,
        #     "admissiion_date": [],
        #     "admissiion_date_score": [],
        #     "discharge_date": [],
        #     "discharge_date_score": [],
        #     "sign_date": [],
        #     "sign_date_score": [],
        #     "icd-10": [],
        # }
        
        self.empty_extracted_result = {
        "data": [
            {
            "info":
                {
                    "medical_facility": None,
                    "medical_facility_box": [],
                    "medical_facility_confidence": 0,
                    
                    "patient_name": None,
                    "patient_name_box": [],
                    "patient_name_confidence": 0,
                    
                    "year_of_birth": None,
                    "year_of_birth_box": [],
                    "year_of_birth_confidence": 0,
                    
                    "gender": None,
                    "gender_box": [],
                    "gender_confidence": 0,
                    
                    "hospitalization_date": None,
                    "hospitalization_date_box": [],
                    "hospitalization_date_confidence": 0,
                    
                    "hospital_discharge_date": None,
                    "hospital_discharge_date_box": [],
                    "hospital_discharge_date_confidence": 0,
                    
                    "icd_10": [],

                    "image_seals": "UN_CHECKED",

                },
            "pages": [],
            "type": None,
            },
            ]
        }

    def extract_discharge_paper_info(self, cv2_img):
        process_img = cv2_img.copy()
        process_img = self.preprocessImage(process_img)
        det_ocr_result = self.lineDetAndOCR.begin_recognize_text(process_img)
        kie_df = self.kieGCN(det_ocr_result, process_img)
        return kie_df
    
    def postprocess_kie_df(self, kie_df):
        full_extracted_result = self.empty_extracted_result.copy()
        is_discharge_paper = False
        raw_text_lines = kie_df["Object"].to_list()
        for each_line in raw_text_lines:
            unaccented_line = unidecode.unidecode(each_line).lower()
            unaccented_line = unaccented_line.replace(" ", "")
            for icheck in ["giayravien", "giayxuatvien", "phieuravien", "phieuxuatvien"]:
                if icheck in unaccented_line:
                    is_discharge_paper = True
                    break
            if is_discharge_paper == True:
                break
            
        if is_discharge_paper == False:
            full_extracted_result["data"][0]["type"] = "unknow"
        else:
            full_extracted_result["data"][0]["type"] = "hospital_discharge_paper"
            
            self.kiePostprocess.append_kie_df(kie_df)

            hospital_name, hospital_name_score = self.kiePostprocess.hospital_name_postprocess()
            if hospital_name == None:
                hospital_name, hospital_name_score = self.kiePostprocess.find_hospital_name_remain()
            # print("hospital_name:", hospital_name)

            patient_name, patient_name_score = self.kiePostprocess.patient_name_postprocess()
            if patient_name == None:
                patient_name, patient_name_score = self.kiePostprocess.find_patient_name_remain()
            # print("patient_name:", patient_name)

            (age, age_score), (temp_gender, temp_gender_score) = self.kiePostprocess.age_postprocess()
            (gender, gender_score), (temp_age, temp_age_score) = self.kiePostprocess.gender_postprocess()
            if age == None and temp_age != None:
                age = temp_age
                age_score = temp_age_score
            if gender == None and temp_gender != None:
                gender = temp_gender
                gender_score = temp_gender_score

            if age == None:
                (age, age_score), (temp_gender, temp_gender_score) = self.kiePostprocess.find_age_remain()

            # print("gender:", gender)
            # print("age:", age)

            admissiion_dates, admissiion_dates_scores = self.kiePostprocess.admission_date_postprocess()
            # for each in admissiion_dates:
                # print("admissiion_date:", each)

            discharge_dates, discharge_dates_scores = self.kiePostprocess.discharge_date_postprocess()
            # for each in discharge_dates:
                # print("discharge_date:", each)

            admissiion_dates, admissiion_dates_scores , discharge_dates, discharge_dates_scores = self.kiePostprocess.admission_discharge_correction(admissiion_dates, admissiion_dates_scores , discharge_dates, discharge_dates_scores)

            sign_dates, sign_dates_scores = self.kiePostprocess.sign_date_postprocess()
            # for each in sign_dates:
                # print("sign_date:", each)


            ICD_codes = self.kiePostprocess.ICD_code_postprocess()
            # print("ICD_codes:", ICD_codes)

            main_extracted_result = full_extracted_result["data"][0]["info"]
            if hospital_name != None:
                main_extracted_result["hospital_name"] = hospital_name
                main_extracted_result["hospital_name_score"] = hospital_name_score
            if patient_name != None:
                main_extracted_result["patient_name"] = patient_name
                main_extracted_result["patient_name_score"] = patient_name_score
            if gender != None:
                main_extracted_result["gender"] = gender
                main_extracted_result["gender_score"] = gender_score
            if age != None:
                main_extracted_result["age"] = age
                main_extracted_result["age_score"] = age_score
                
            main_extracted_result["admissiion_date"] = admissiion_dates
            main_extracted_result["admissiion_date_score"] = admissiion_dates_scores
            
            main_extracted_result["discharge_date"] = discharge_dates
            main_extracted_result["discharge_date_score"] = discharge_dates_scores
            
            main_extracted_result["sign_date"] = sign_dates
            main_extracted_result["sign_date_score"] = sign_dates_scores
            
            org_icd_code = []
            for each in ICD_codes:
                temp_icd = {
                    "icd": each,
                    "box": [],
                    "score": -1
                }
                org_icd_code.append(temp_icd)
            main_extracted_result["icd-10"] = org_icd_code
            full_extracted_result["data"][0]["info"] = main_extracted_result

        return full_extracted_result

    def __call__(self, cv2_img):
        # cv2_img = cv2.imread(img_path)
        extracted_df = self.extract_discharge_paper_info(cv2_img)
        extracted_result = self.postprocess_kie_df(extracted_df)
        gc.collect()
        return extracted_result, extracted_df