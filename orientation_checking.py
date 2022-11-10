# from __future__ import print_function, division
from turtle import up
import torch
import torch.nn as nn
# from torch.optim import lr_scheduler
import numpy as np
from torchvision import models, transforms
# import matplotlib.pyplot as plt
# import time
import cv2
# from tqdm.notebook import tqdm_notebook
# from google.colab.patches import cv2_imshow
from PIL import Image
from pytesseract import Output
import pytesseract
import imutils
import uuid
import os

class OrientationChecker():
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
        self.model_path = model_path
        self.test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_names = ['down', 'left', 'right', 'up']
        self.model = self.load_rotation_checking_model(self.model_path, self.device)
    
    def load_rotation_checking_model(self, model_path, device, is_train = False):
        model_ft = models.resnet18(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 4)
        model_ft.load_state_dict(torch.load(model_path))
        model_ft = model_ft.to(device)
        if is_train == False:
            model_ft.eval()
        return model_ft

    def __call__(self, image):
        img = image
        is_cv2 = False
        if type(img) is np.ndarray:
            img = Image.fromarray(img)
            is_cv2 = True
            # img = np.array(img)
            # img = cv2.cvtColor(cv2.COLOR_RGB2BGR)
        img = img.convert("RGB")
        X = self.test_transforms(img)
        X = X.unsqueeze_(0)
        X = X.to(self.device)
        pred_class = "up"
        with torch.no_grad():
            outputs = self.model(X.float())
            _, preds = torch.max(outputs, 1)
            pred_class = self.class_names[preds]

        cv2_img = np.array(img)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        if pred_class == "left":
            cv2_img = cv2.rotate(cv2_img, cv2.ROTATE_90_CLOCKWISE)
        elif pred_class == "right":
            cv2_img = cv2.rotate(cv2_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # elif pred_class == "down":
        #     cv2_img = cv2.rotate(cv2_img, cv2.ROTATE_180)
        # elif pred_class == "up":
        #     return img, pred_class

        if is_cv2:
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
            return cv2_img, pred_class
        else:
            return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)), pred_class