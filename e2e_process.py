import cv2

import preprocess_img
import LineOCR
import pandas as pd
import numpy as np
import kie_gcn
from kie_gcn import InvoiceGCN



preprocessImage = preprocess_img.PreprocessImage()
lineDetAndOCR = LineOCR.ProcessImage()
kieGCN = kie_gcn.KieGCN()

test_img = cv2.imread(r"imgs_test\705.jpeg")
test_img = preprocessImage(test_img)

result = lineDetAndOCR.begin_recognize_text(test_img)
kie_df = kieGCN(result, test_img)

kie_df.to_excel("result.xlsx")