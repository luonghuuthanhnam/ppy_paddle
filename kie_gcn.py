from dataclasses import InitVar
from unittest import result
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
import os
import time
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import as_completed
import uuid

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

class InvoiceGCN(nn.Module):
    def __init__(self, input_dim, chebnet=False, n_classes=5, dropout_rate=0.2, K=3):
        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate

        if chebnet:
            self.conv1 = ChebConv(self.input_dim, 64, K=K)
            # self.conv2 = ChebConv(64, 32, K=K)
            self.conv3 = ChebConv(64, 64, K=K)
            self.conv4 = ChebConv(64, self.n_classes, K=K)
        else:
            self.conv1 = GCNConv(self.first_dim, 64, improved=True, cached=True)
            self.conv2 = GCNConv(64, 32, improved=True, cached=True)
            self.conv3 = GCNConv(32, 16, improved=True, cached=True)
            self.conv4 = GCNConv(16, self.n_classes, improved=True, cached=True)

    def forward(self, data):
        # for transductive setting with full-batch update
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = F.dropout(F.relu(self.conv1(x, edge_index, edge_weight)), p=self.dropout_rate, training=self.training)
        # x = F.dropout(F.relu(self.conv2(x, edge_index, edge_weight)), p=self.dropout_rate, training=self.training)
        x = F.dropout(F.relu(self.conv3(x, edge_index, edge_weight)), p=self.dropout_rate, training=self.training)
        x = self.conv4(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)

class KieGCN():
    def __init__(self, 
            vncorenlp_segmentation_dir = 'weights/nlp/vncorenlp',
            gcn_model_path =  "weights/gcn/GCN_221017.pth"
            ) -> None:
        #Load Text Segmentation engine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
        gcn_model_path = os.path.join(os.getcwd(), gcn_model_path)
        vncorenlp_segmentation_dir = os.path.join(os.getcwd(), vncorenlp_segmentation_dir)
        self.rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_segmentation_dir)
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.current_raw_df = None
        self.feature_cols = [
            "rd_b",
            "rd_r",
            "rd_t",
            "rd_l",
            "line_number",
            "n_upper",
            "n_alpha",
            "n_spaces",
            "n_numeric",
            "n_special",
        ]
        self.temp_dir = "temp_data/temp_gcn/"
        os.chdir('../../../')
        self.gcn_model = InvoiceGCN(input_dim=778, chebnet=True, n_classes =15, dropout_rate = 0.3, K=3)
        self.gcn_model.load_state_dict(torch.load(gcn_model_path))
        # self.gcn_model = torch.load(gcn_model_path)
        self.gcn_model.to(self.device)
        self.gcn_model.eval()

    def transform_detocr_to_df_and_img(self, det_n_ocr_result, cv2_img):
        result = det_n_ocr_result
        list_xmin = []
        list_ymin = []
        list_xmax = []
        list_ymax = []
        list_Object = []
        list_polygon = []

        for iter in range(len(result["bbox"])):
            xmin, ymin, xmax, ymax = result["bbox"][iter]
            text = result["text"][iter]
            polygon = result["polygon"][iter]
            list_xmin.append(xmin)
            list_xmax.append(xmax)
            list_ymin.append(ymin)
            list_ymax.append(ymax)
            list_Object.append(text)
            list_polygon.append(polygon)
        single_df = {
            "xmin": list_xmin,
            "xmax": list_xmax,
            "ymin": list_ymin,
            "ymax": list_ymax,
            "Object": list_Object,
            "labels": ["None"]*len(list_Object),
            "polygon": list_polygon,
        }
        temp_file_name = str(uuid.uuid4())
        single_df = pd.DataFrame.from_dict(single_df)
        single_df.to_excel(f"temp_data/temp_gcn/csv/{temp_file_name}.xlsx")
        cv2.imwrite(f"temp_data/temp_gcn/img/{temp_file_name}.jpg", cv2_img)
        return temp_file_name 
        

    def get_sentence_features(self, sentence):
        with torch.no_grad():
            segmented_sentence = self.rdrsegmenter.word_segment(sentence)
            tokenized_sentence = torch.tensor([self.tokenizer.encode(segmented_sentence[0])])
            features = []
            features = self.phobert(tokenized_sentence)
        return features

    def make_sent_bert_features(self, text):
        features = self.get_sentence_features(text)
        return features[1][0].cpu().numpy()

    def make_graph_data(self, filename = 'temp'):
        connect = graph.Grapher(filename)
        G,result, df = connect.graph_formation(export_graph=False)
        df = connect.relative_distance(export_document_graph = False)
        individual_data = from_networkx(G)
        self.current_raw_df = df.copy()
        return df, individual_data

    def gcn_final_transform_data(self, df, individual_data):
        text_features = []
        text_list = []
        for _, row in df.iterrows():
            text_list.append(row["Object"])
        with ThreadPoolExecutor(max_workers=100) as executor:
            for text_feature in executor.map(self.make_sent_bert_features, text_list):
                text_features.append(text_feature)
                # text_features.append(self.make_sent_bert_features(row["Object"]))
        
        text_features = np.asarray(text_features, dtype=np.float32)

        numeric_features = df[self.feature_cols].values.astype(np.float32)

        features = np.concatenate((numeric_features, text_features), axis=1)
        features = torch.tensor(features)

        for col in df.columns:
            try:
                df[col] = df[col].str.strip()
            except AttributeError as e:
                pass
        text = df["Object"].values
        individual_data.x = features

        individual_data.text = text

        test_list_of_graphs = []
        test_list_of_graphs.append(individual_data)

        data_transformed = ""
        data_transformed = torch_geometric.data.Batch.from_data_list(test_list_of_graphs)
        data_transformed.edge_attr = None
        return data_transformed

    def model_inference(self, data_transformed):
        with torch.no_grad():
            y_preds= self.gcn_model(data_transformed.to(self.device))
            return y_preds

    def post_process(self, predicted_data):

        pred_classes = predicted_data.max(dim=1)[1].cpu().numpy()
        shorten_pred = [predicted_data[idx][pred_classes[idx]] for idx in range(len(pred_classes))]
        pred_probs = list(torch.exp(torch.tensor(shorten_pred)).cpu().numpy())
        pred_label = [ID2LABEL_DICT[each] for each in list(pred_classes)]


        cur_df = self.current_raw_df.copy()
        cur_df["pred_label"] = pred_label
        cur_df["confidence_score"] = pred_probs
        cur_df = cur_df[["index", "xmin", "ymin", "xmax", "ymax", "Object", "pred_label", "confidence_score", "polygon"]]

        self.current_raw_df = None
        return cur_df

    def __call__(self, det_n_ocr_result, cv2_img) -> pd.DataFrame:
        temp_file_name = self.transform_detocr_to_df_and_img(det_n_ocr_result, cv2_img)
        s_time = time.time()
        df, individual_data =self.make_graph_data(filename=temp_file_name)
        print("make graph: ", time.time() - s_time)
        r_time = time.time()
        data_transformed = self.gcn_final_transform_data(df, individual_data)
        print("transform: ", time.time() - r_time)
        y_preds = self.model_inference(data_transformed)
        predicted_df = self.post_process(y_preds)
        temp_xlsx_path = f"./{self.temp_dir}csv/{temp_file_name}.xlsx"
        temp_img_path = f"./{self.temp_dir}img/{temp_file_name}.jpg"
        if os.path.exists(temp_xlsx_path):
            os.remove(temp_xlsx_path)
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        return predicted_df
        