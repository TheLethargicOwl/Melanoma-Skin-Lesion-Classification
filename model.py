import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchtoolbox.transform as transforms
import pandas as pd
import numpy as np
import cv2

test_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

class Net(nn.Module):
    def __init__(self, arch, n_meta_features: int):
        super(Net, self).__init__()
        self.arch = arch
        if 'ResNet' in str(arch.__class__):
            self.arch.fc = nn.Linear(in_features=512, out_features=500, bias=True)
        if 'EfficientNet' in str(arch.__class__):
            self.arch._fc = nn.Linear(in_features=1280, out_features=500, bias=True)
        self.meta = nn.Sequential(nn.Linear(n_meta_features, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(500, 250),  # FC layer output will have 250 features
                                  nn.BatchNorm1d(250),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        self.ouput = nn.Linear(500 + 250, 1)
        
    def forward(self, inputs):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x, meta = inputs
        cnn_features = self.arch(x)
        meta_features = self.meta(meta)
        features = torch.cat((cnn_features, meta_features), dim=1)
        output = self.ouput(features)
        return output

model = torch.load('./model_1.pth',map_location=torch.device("cpu"))

model.eval()

site_dict = { 'head/neck':2, 'lower extremity':3, 'oral/genital':4, 
				'palms/soles':5, 'torso':6,'upper extremity':7,'unknown':8}


def predict(img,sex,age,site):

	meta_features = np.array([0,0,0,0,0,0,0,0,0],dtype=np.float32)

	if sex == 'male':
		meta_features [0] = 1.0
	else:
		meta_features [0] = 0.0

	meta_features [1] = age/90.0

	meta_features[site_dict[site]] = 1

	img = test_transform(img)

	img = torch.tensor(img, device=torch.device("cpu"), dtype=torch.float32)

	meta_features = torch.tensor(meta_features, device=torch.device("cpu"), dtype=torch.float32)

	img = img[None,:,:,:]

	meta_features = meta_features[None,:]

	inputs = (img,meta_features)

	preds = model(inputs)

	preds = torch.sigmoid(preds)

	return preds