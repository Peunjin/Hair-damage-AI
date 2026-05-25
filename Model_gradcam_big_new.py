import torch
import cv2
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from scipy.interpolate import interp1d
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import streamlit as st
import random

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])])

def auto_canny(image, sigma=0.33):
    image = cv2.GaussianBlur(image, (3, 3), 0.2)
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

def find_hair_horizontal_bounds_by_gradient2(smoothed, window_size=5, split_index=440):
    # Ensure window doesn't exceed bounds
    # assert window_size > 0 and window_size < len(smoothed)

    # --- Top half: smoothed[:split_index] ---
    top_half = smoothed[100:split_index-100]
    top_diffs = [np.abs(top_half[i + window_size] - top_half[i]) for i in range(len(top_half) - window_size)]
    top_diffs = np.array(top_diffs)
    top_idx = np.argmax(top_diffs)
    top_edge = min(top_idx, top_idx + window_size)
    top_edge += 100  # Adjust for the offset

    # --- Bottom half: smoothed[split_index:] ---
    bottom_half = smoothed[split_index+100:-50]
    bottom_diffs = [np.abs(bottom_half[i + window_size] - bottom_half[i]) for i in range(len(bottom_half) - window_size)]
    bottom_diffs = np.array(bottom_diffs)
    if len(bottom_diffs) == 0:
        bottom_edge = split_index + 200
    else:
        bottom_idx = np.argmax(bottom_diffs)
        # Adjust index to global position
        bottom_edge = split_index + max(bottom_idx, bottom_idx + window_size)
        bottom_edge += 100  # Adjust for the offset

    # Merge diffs for optional visualization
    # full_diffs = np.concatenate([top_diffs, np.zeros(window_size), bottom_diffs])

    return top_edge, bottom_edge

def find_hair_bounds(edge_img, window_size=5):
    # 각 행마다 edge 픽셀 개수 계산
    row_sums = np.sum(edge_img > 0, axis=1)
    smoothed = cv2.GaussianBlur(row_sums.astype(np.float32), (5, 1), 0)

    # 경계 찾기
    # _, _, diffs = self.find_hair_horizontal_bounds_by_gradient1(smoothed, window_size)
    top_edge, bottom_edge = find_hair_horizontal_bounds_by_gradient2(smoothed, window_size)
    return top_edge, bottom_edge


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    # version 0
    # model.fc = nn.Linear(num_ftrs, 1)
    # model.load_state_dict(torch.load('cosmax_micro_gradcam.pth', map_location=torch.device('cpu')))
    # version 1
    model.fc = nn.Linear(num_ftrs, 5)
    model.load_state_dict(torch.load('./best_ordinal_reg_maeloss_sigmoid_preprocess.pth', map_location=torch.device('cpu')))
    return model

def predict_gradcam(img):
    # image given as numpy array
    # print(img.shape)
    img_org = img
    # img_org = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_org.shape[:2]
    gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    gray = gray[:860]
    edge = auto_canny(gray)
    top_boundary, bottom_boundary = find_hair_bounds(edge, window_size=20)
    img = cv2.resize(img_org[top_boundary:bottom_boundary, :], (1280, 510))

    img = test_transform(img)
    img = img.unsqueeze(0)

    # model = models.resnet18(weights=None)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 1)
    # model.load_state_dict(torch.load('cosmax_micro_gradcam.pth', map_location=torch.device('cpu')))
    model = load_model()
    model.eval()

    # Grad-CAM target layer
    target_layers = [model.layer4[-1]]

    with torch.no_grad():
        # version 0
        # output = model(img)
        # score = torch.sigmoid(output).item()

        # version 1
        output = model(img)
        score = torch.sigmoid(output).sum()
        score = torch.clamp(score, 0, 5).item()
        score = score / 5

    pred = (score > 0.5)
    
    with GradCAM(model=model, target_layers=target_layers) as grad_cam:
        grayscale_cam = grad_cam(input_tensor=img, targets=None)[0]

        # Resize only the region between top and bottom boundaries
        region_height = bottom_boundary - top_boundary
        resized_cam = cv2.resize(grayscale_cam, (orig_w, region_height))

        # Create full-size cam with zeros and insert the resized cam in the middle
        full_cam = np.zeros((orig_h, orig_w), dtype=np.float32) 
        full_cam[top_boundary:bottom_boundary, :] = resized_cam

        # Normalize the original image to [0, 1]
        original_image_normalized = np.float32(img_org) / 255.0

        if pred==0:
            full_cam *= 0.4

        # Overlay CAM on the original image
        visualization = show_cam_on_image(original_image_normalized, full_cam, use_rgb=True)
    
    # score and pred to int
    score = round(score*100)
    pred = int(pred)

    # plot image using st.pyplot
    return score, pred, visualization

def predict_gradcam_score(img):
    # image given as numpy array
    # print(img.shape)
    img_org = img
    # img_org = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_org.shape[:2]
    gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    gray = gray[:860]
    edge = auto_canny(gray)
    top_boundary, bottom_boundary = find_hair_bounds(edge, window_size=20)
    img = cv2.resize(img_org[top_boundary:bottom_boundary, :], (1280, 510))

    img = test_transform(img)
    img = img.unsqueeze(0)

    # model = models.resnet18(weights=None)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 1)
    # model.load_state_dict(torch.load('cosmax_micro_gradcam.pth', map_location=torch.device('cpu')))
    model = load_model()
    model.eval()

    # Grad-CAM target layer
    target_layers = [model.layer4[-1]]

    with torch.no_grad():
        output = model(img)
        # version 0
        # score = torch.sigmoid(output).item()

        # version 1
        score = torch.sigmoid(output).sum()
        score = torch.clamp(score, 0, 5).item() / 5
    return score


            
