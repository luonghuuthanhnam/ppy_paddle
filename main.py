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

from fastapi import FastAPI

class ImgPath(BaseModel):
    img_path: str


app = FastAPI()

print("*****Loading models...*****")
load_model_time = time.time()
orientationChecker = OrientationChecker( model_path= "./weights/orientation/invoice_rotation_220920.pth")
e2e_OCR_Engine = e2e_process.E2E_OCR_Engine(
    detection_model_path="./PaddleOCR/pretrained_models/exported_det_model_221025",
    text_recognition_model_path="./weights/ocr/ocr_221026.pth",
    gcn_model_path="./weights/gcn/GCN_221027_best_state_dict.pth"
)
print(f"*****Models has been uploaded successfully in {time.time() - load_model_time} s*****")

def extract_discharge_paper(img_path):
    image = cv2.imread(img_path)
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
async def ocr_handeler(data: ImgPath):
    print(data.img_path)
    result = extract_discharge_paper(data.img_path)
    return result