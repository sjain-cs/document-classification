import fnmatch
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import image_classification as ic
import time
import os
import pickle

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


base_dir = 'data/document_classification'
model_file = 'models/document_classifcation.pth'
meta_file = 'models/document_classifcation_meta.p'
img_path = sys.argv[1]

metadata = pickle.load(open(meta_file,'rb'))
class_names = metadata['classes']

trained_model = torch.load(model_file)
pred = ic.test_model(trained_model, img_path, data_transforms['val'])
pred = pred[0]
pred_class = class_names[pred]

print('Predicted class:', pred_class)


