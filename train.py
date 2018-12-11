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

num_workers = 4
batch_size = 32
learning_rate=0.001
num_epochs = 400
finetune = 'all'
metadata = {}

train_dir = os.path.join(base_dir,'train')
val_dir = os.path.join(base_dir,'val')

train_dataset = ic.load_dataset(train_dir,'train', data_transforms)
val_dataset   = ic.load_dataset(val_dir,'val', data_transforms)

ic.print_dataset_stats(train_dataset,'train')
ic.print_dataset_stats(val_dataset,'val')

train_loader = ic.create_data_loader(train_dataset, batch_size, True, num_workers)
val_loader   = ic.create_data_loader(val_dataset, batch_size, False, num_workers)

data_loaders = {}
data_loaders['train'] = train_loader
data_loaders['val'] = val_loader

num_classes = len(train_dataset.classes)
metadata['classes'] = train_dataset.classes

[net, param_list] = ic.load_resnet18(finetune=finetune, num_classes=num_classes)

optimizer = optim.Adam(param_list, lr=learning_rate)
criterion = nn.CrossEntropyLoss()
net = ic.train_model(net, data_loaders, optimizer, criterion, num_epochs, model_file)

with open(meta_file,'wb') as handle:
    pickle.dump(metadata, handle)

print('Training complete, validating with the best model')
trained_model = torch.load(model_file)
[val_gt, val_pred] = ic.validate_model(trained_model, data_loaders['val'], criterion)
stats = ic.compute_stats(val_gt, val_pred)
