import json
from PaddleOCR.ppocr import data
import e2e_process
from kie_gcn import InvoiceGCN
from PIL import Image
from IPython.display import display
import cv2
from orientation_checking import OrientationChecker
from PIL import Image
from IPython.display import display
import time
import torch.cuda
import gc
from pydantic import BaseModel
import preprocess_img
from fastapi import FastAPI

class FileName(BaseModel):
    file_name: str
    bucket_name: str

# downandLoadImage = preprocess_img.DownAndLoadImage("papaya-fwd-prod-stp")
downandLoadImage = preprocess_img.DownAndLoadImage()
app = FastAPI()

print("*****Loading models...*****")
load_model_time = time.time()
orientationChecker = OrientationChecker( model_path= "./weights/orientation/invoice_rotation_221212.pth")
preprocessImage = preprocess_img.PreprocessImage()
e2e_OCR_Engine = e2e_process.E2E_OCR_Engine(
    detection_model_path="PaddleOCR/pretrained_models/det_r50_td_tr_inference_221209",
    text_recognition_model_path="./weights/ocr/line_ocr_221214.pth",
    gcn_state_dict_path="./weights/gcn/GCN_221117_state_dict.pth"
)
print(f"*****Models has been uploaded successfully in {time.time() - load_model_time} s*****")

def extract_discharge_paper(file_name, bucket_name, return_df = False):
    s_time = time.time()
    image = downandLoadImage(file_name, bucket_name)
    print("download time: ", round(time.time() - s_time, 3))
    # image = cv2.imread(img_path)
    rotated_img, pred_class, prob = orientationChecker(image)
    # print("orientation:", pred_class, prob)
    
    process_img, (new_width, new_height), median_angle = preprocessImage(rotated_img)
    result, extracted_df = e2e_OCR_Engine(process_img)
    result["preprocess_info"]["new_width"]  = new_width
    result["preprocess_info"]["new_height"]  = new_height
    result["preprocess_info"]["rotation_angle"]  = median_angle
    result["preprocess_info"]["orientation"]["predicted_class"]  = pred_class
    result["preprocess_info"]["orientation"]["score"]  = prob
    
    torch.cuda.empty_cache()
    gc.collect()
    if return_df == True:
        return result, extracted_df
    else:
        return result

@app.get("/")
def read_root():
    return {"Papaya": "Success"}

@app.post("/OCR/DischargePaper/")
async def ocr_handeler(data: FileName):
    print("file_name:", data.file_name)
    start_time = time.time()
    file_name = data.file_name
    bucket_name = data.bucket_name
    result = extract_discharge_paper(file_name, bucket_name)
    print("TOTAL PROCESSING TIME:", round(time.time()-start_time, 3))
    return result

@app.post("/OCR/DischargePaper/Dev")
async def ocr_handeler(data: FileName):
    print("file_name:", data.file_name)
    start_time = time.time()
    file_name = data.file_name
    bucket_name = data.bucket_name
    result, extracted_df = extract_discharge_paper(file_name, bucket_name, return_df=True)
    extracted_dict = extracted_df.to_dict()
    total_result = {
        "result":result,
        "extracted_dict":extracted_dict
    }
    print("TOTAL PROCESSING TIME:", round(time.time()-start_time, 3))
    return total_result

# https://drive.google.com/file/d/1QDxM_TSRI8GXwPpkY-vn_wrBK1ywC9fn/view?usp=share_link
# !gdown "https://drive.google.com/uc?id=1QDxM_TSRI8GXwPpkY-vn_wrBK1ywC9fn"