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



class ObjectName(BaseModel):
    obj_name: str

downandLoadImage = preprocess_img.DownAndLoadImage("papaya-fwd-prod-stp")
app = FastAPI()

print("*****Loading models...*****")
load_model_time = time.time()
orientationChecker = OrientationChecker( model_path= "./weights/orientation/invoice_rotation_220920.pth")
e2e_OCR_Engine = e2e_process.E2E_OCR_Engine(
    detection_model_path="PaddleOCR/pretrained_models/det_db_inference_221109",
    text_recognition_model_path="./weights/ocr/ocr_221026.pth",
    gcn_state_dict_path="./weights/gcn/GCN_221103_state_dict.pth"
)
print(f"*****Models has been uploaded successfully in {time.time() - load_model_time} s*****")

def extract_discharge_paper(object_name):
    s_time = time.time()
    image = downandLoadImage(object_name)
    print("download time: ", time.time() - s_time)
    # image = cv2.imread(img_path)
    rotated_img, pred_class = orientationChecker(image)
    print("orientation:", pred_class)
    result, extracted_df = e2e_OCR_Engine(rotated_img)
    torch.cuda.empty_cache()
    gc.collect()
    return result

@app.get("/")
def read_root():
    return {"Papaya": "Success"}


@app.post("/OCR/DischargePaper/")
async def ocr_handeler(data: ObjectName):
    print(data.obj_name)
    result = extract_discharge_paper(data.obj_name)
    return result