import torch
import cv2
import random
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

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
        self.fc1 = nn.Linear(14*14*128, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x (224, 224, 3)
        x = F.max_pool2d(F.relu(self.conv1_bn(self.conv1(x))), 2) # (112, 112, 16)
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), 2) # (56, 56, 32)
        x = F.max_pool2d(F.relu(self.conv3_bn(self.conv3(x))), 2) # (28, 28, 64)
        x = F.max_pool2d(F.relu(self.conv4_bn(self.conv4(x))), 2) # (14, 14, 128)
        x = x.view(-1, 14*14*128)
        x = self.dropout(x)
        x = self.fc1(x)

        return x

def crop_image(img, transform = None, test = False):
    img = img[0:870, 0:1280]
    img_list = []
    coordinates = []

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    top_half = gray[:gray.shape[0]//7*3, :]
    bottom_half = gray[gray.shape[0]//7*4:, :]
    sobel_top = cv2.Sobel(top_half, cv2.CV_64F, 0, 1, ksize=5)
    sobel_bottom = cv2.Sobel(bottom_half, cv2.CV_64F, 0, 1, ksize=5)
    avg_gradient_top = np.average(sobel_top, axis=1)
    top_boundary = np.argmax(avg_gradient_top)
    avg_gradient_bottom = np.average(sobel_bottom, axis=1)
    bottom_boundary = np.argmax(avg_gradient_bottom) + gray.shape[0]//7*4

    top_boundary = max(0, top_boundary - 50)
    bottom_boundary = min(gray.shape[0], bottom_boundary + 50)

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
    

def predict(img):
    # image given as numpy array RGB
    img_list, coordinates, top_boundary, bottom_boundary = crop_image(img, test_transform, test=True)

    # model = Cosmax()
    # model.load_state_dict(torch.load('cosmax.pth', weights_only=True))
    # model.load_state_dict(torch.load('cosmax.pth'))
    model = models.resnet18(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load('best_resnet_big_patch.pth', map_location=torch.device('cpu')))
    model.eval()

    # predict 25 images at once
    # img_list has 100 images
    result = []
    coordinates_damage = []
    probs = []
    for i in range(0, 100, 25):
        img_batch = torch.stack(img_list[i:i+25])
        output = model(img_batch)
        prob = (torch.sigmoid(output)).cpu().detach().numpy()
        preds = (prob > 0.5)
        result.extend(preds.tolist())

        for j in range(len(preds)):
            if preds[j] == 1:
                coordinates_damage.append([coordinates[i+j][0], coordinates[i+j][1]])
                probs.append(prob[j].item())

    # print(result)
    # print(probs)
    # print(coordinates_damage)

    # return len(result) - sum(result), coordinates_damage, top_boundary, bottom_boundary
    return len(coordinates_damage), coordinates_damage, top_boundary, bottom_boundary, probs