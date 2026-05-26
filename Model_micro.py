import torch
import cv2
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from scipy.interpolate import interp1d
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
import streamlit as st
import random

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])])

class Cosmax(nn.Module):
    def __init__(self):
        super(Cosmax, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(7*7*256, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x (224, 224, 3)
        x = F.max_pool2d(F.relu(self.conv1_bn(self.conv1(x))), 2) # (112, 112, 16)
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), 2) # (56, 56, 32)
        x = F.max_pool2d(F.relu(self.conv3_bn(self.conv3(x))), 2) # (28, 28, 64)
        x = F.max_pool2d(F.relu(self.conv4_bn(self.conv4(x))), 2) # (14, 14, 128)
        x = F.max_pool2d(F.relu(self.conv5_bn(self.conv5(x))), 2) # (7, 7, 256)
        x = x.view(-1, 7*7*256)
        x = self.dropout(x)
        x = self.fc1(x)

        return x

# def find_contours(img):
#     img = cv2.resize(img, (224, 224))
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Apply GaussianBlur to reduce noise
#     blur = cv2.GaussianBlur(gray, (7, 7), 1)

#     # Use Canny edge detection
#     edges = cv2.Canny(blur, 0, 120)

#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     # Filter contours based on number of contour points, take the max 2
#     contours = sorted(contours, key=lambda x: len(x), reverse=True)[:2]

#     return contours

# def interpolate_contours(top_contour, bottom_contour, width, height):
#     """
#     Interpolates the top and bottom contours to ensure that for every x in the range [0, h-1],
#     there is a corresponding y value for both the top and bottom contours.
    
#     Args:
#     top_contour: The top contour (array of shape Nx2).
#     bottom_contour: The bottom contour (array of shape Nx2).
#     width: The width of the image (range for x values).
#     height: The height of the image.
    
#     Returns:
#     top_y_interpolated: Interpolated y values for the top contour (for all x in the range [0, width-1]).
#     bottom_y_interpolated: Interpolated y values for the bottom contour (for all x in the range [0, width-1]).
#     """
#     # Get x and y coordinates from top and bottom contours
#     top_x = top_contour[:, 0]  # X coordinates of top contour
#     top_y = top_contour[:, 1]  # Y coordinates of top contour

#     bottom_x = bottom_contour[:, 0]  # X coordinates of bottom contour
#     bottom_y = bottom_contour[:, 1]  # Y coordinates of bottom contour

#     # Create interpolation functions for both top and bottom contours
#     # kind='linear' means we're using linear interpolation
#     top_interp = interp1d(top_x, top_y, kind='linear', fill_value="extrapolate")
#     bottom_interp = interp1d(bottom_x, bottom_y, kind='linear', fill_value="extrapolate")

#     # Generate x values for the full range (0 to width-1)
#     x_values = np.arange(0, width)

#     # Get interpolated y values for the full x range
#     top_y_interpolated = top_interp(x_values)
#     bottom_y_interpolated = bottom_interp(x_values)

#     # Clip the y values to stay within image bounds (0 to height-1)
#     top_y_interpolated = np.clip(top_y_interpolated, 0, height - 1)
#     bottom_y_interpolated = np.clip(bottom_y_interpolated, 0, height - 1)

#     return top_y_interpolated, bottom_y_interpolated

# @st.cache_resource
# def load_model():
#     model = models.resnet18(weights=None)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, 1)
#     model.load_state_dict(torch.load('best_micro.pth', map_location=torch.device('cpu'), weights_only=True))
#     return model

def crop_image(img, transform = None, test = False, img_path = None):
    img_list = []
    coordinates = []

    # gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1)
    edges = cv2.Canny(blur, 0, 120)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=lambda x: len(x), reverse=True)[:2]
    
    cont1 = contours[0][:,0,1].mean()
    cont2 = contours[1][:,0,1].mean()
    
    if cont1 > cont2:
        bottom_boundary = contours[0][:,0,1].max()
        top_boundary = contours[1][:,0,1].min()
    else:
        top_boundary = contours[0][:,0,1].min()
        bottom_boundary = contours[1][:,0,1].max()
        
    if abs(top_boundary - bottom_boundary) < 224:
        diff = 224 - abs(top_boundary - bottom_boundary)
        top_boundary = max(0, top_boundary - diff // 2 - 1)
        bottom_boundary = min(img.shape[0], bottom_boundary + diff // 2 + 1)
    
    img = img[top_boundary:bottom_boundary, :]

    # After horizontal crop, apply random crop to the image with size 224, 224 to 10 times
    h, w = img.shape[:2]
    for i in range(100):
        x = random.randint(0, w - 224)
        if h < 224:
            print(img.shape)
        y = random.randint(0, h - 224)
        crop_img = img[y:y + 224, x:x + 224]
        if transform:
            crop_img = transform(crop_img)
        img_list.append(crop_img)
        coordinates.append((x, y))
    
    if test:
        return img_list, coordinates, top_boundary, bottom_boundary
    else:
        return img_list

def predict_micro(img):
    # image given as numpy array
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_list, coordinates, top_boundary, bottom_boundary = crop_image(img, test_transform, test=True)
    
    model = Cosmax()
    model.load_state_dict(torch.load('cosmax_micro_old.pth', map_location=torch.device('cpu')))
    model.eval()
    
    result = []
    coordinates_damage = []
    probs = []
    
    for i in range(0, 100, 25):
        img_batch = torch.stack(img_list[i:i+25])
        output = model(img_batch)
        weight, preds = torch.max(output, 1)
        result.extend(preds.tolist())
        prob = F.softmax(output, dim=1).cpu().detach().numpy()
        result.extend(preds.tolist())
        
        for j in range(len(preds)):
            if preds[j] == 0:
                coordinates_damage.append([coordinates[i+j][0], coordinates[i+j][1]])
                probs.append(prob[j][0].item())
                
    return len(coordinates_damage), coordinates_damage, top_boundary, bottom_boundary, probs

    # model = models.resnet18(weights=None)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 1)
    # model.load_state_dict(torch.load('best_micro.pth', map_location=torch.device('cpu'), weights_only=True))
    # model = load_model()
    # model.eval()

    # # Grad-CAM target layer
    # target_layers = [model.layer4[-1]]

    # output = model(img_input)
    # score = torch.sigmoid(output).item()
    # pred = (score > 0.5).float()

    # if pred == 0:
    #     contours = find_contours(img)
    #     if contours[0][:,0,1].mean() < contours[1][:,0,1].mean():
    #         top = contours[0][:,0,:]
    #         bottom = contours[1][:,0,:]
    #     else:
    #         top = contours[1][:,0,:]
    #         bottom = contours[0][:,0,:]

    #     # Remove duplicate points
    #     top = np.unique(top, axis=0)
    #     bottom = np.unique(bottom, axis=0)
        
    #     # remove points that are zero y coordinate from top
    #     top = top[top[:,1] != 0]

    #     top_min = top[:,1].min()
    #     bottom_max = bottom[:,1].max()

    #     top_y_interpolated, bottom_y_interpolated = interpolate_contours(top, bottom, orig_w, orig_h)
    #     top_y_interpolated = np.where(np.isnan(top_y_interpolated), top_min, top_y_interpolated)
    #     bottom_y_interpolated = np.where(np.isnan(bottom_y_interpolated), bottom_max, bottom_y_interpolated)
    
    # with GradCAM(model=model, target_layers=target_layers) as cam:
    #     grayscale_cam = cam(input_tensor=img_input, targets=None)
    #     grayscale_cam = grayscale_cam[0, :]

    #     grayscale_cam = cv2.resize(grayscale_cam, (orig_w, orig_h))
        
    #     mask = np.zeros_like(grayscale_cam)

    #     if pred == 0:
    #         for x in range(mask.shape[1]):
    #             min_top_y = int(top_y_interpolated[x])
    #             max_bottom_y = int(bottom_y_interpolated[x])
    #             mask[min_top_y:max_bottom_y, x] = 1
    #     else:
    #         mask = np.ones_like(mask)

    #     maksed_cam = grayscale_cam * mask
    #     original_image_normalized = np.float32(img) / 255.0

    #     visualization = show_cam_on_image(original_image_normalized, maksed_cam, use_rgb=True)
    
    # # score and pred to int
    # score = int(score*100)
    # pred = int(pred)

    # # plot image using st.pyplot
    # return score*100, pred, visualization

