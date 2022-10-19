import pandas as pd

import graph
from torch_geometric.utils.convert import from_networkx
import torch_geometric

import py_vncorenlp
import torch
from transformers import AutoModel, AutoTokenizer

import torch.nn as nn
from torch_geometric.nn import ChebConv, GCNConv
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import py_vncorenlp
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type

ID2LABEL_DICT = {
    0: "None",
    1: "sign_date",
    2: "diagnose",
    3:"hospital_name",
    4:"address",
    5:"age",
    6:"treatment",
    7:"patient_name",
    8:"admission_date",
    9:"discharge_date",
    10:"gender",
    11:"document_type",
    12:"department",
    13:"note",
    14:"BHYT",
}

class KieGCN():
    def __init__(self, 
            vncorenlp_segmentation_dir = r'D:\PPYCode\OCR\ppy_paddle\weights\nlp\vncorenlp', 
            ) -> None:
        #Load Text Segmentation engine
        self.rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_segmentation_dir)
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    # def gcn_data_transform(self, det_n_ocr_result, ):
    #     self.