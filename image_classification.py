import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from PIL import Image

use_gpu = torch.cuda.is_available()

def load_image(path, transform):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        img = transform(img)
        return img.unsqueeze(0)

def load_dataset(data_dir, dataset_type, data_transforms):
    dataset = datasets.ImageFolder(data_dir, data_transforms[dataset_type])
    return dataset

def print_dataset_stats(dataset, dataset_type):
    print('Dataset type: %s'%(dataset_type))
    print('Dataset classes: %s'%(dataset.classes))
    print('Number of images: %s'%(len(dataset)))
    print()

def create_data_loader(dataset, batch_size, shuffle, num_workers):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

def compute_stats(gt,pred):
    print('='*50)   

    c_mat = confusion_matrix(gt, pred)
    
    #c_mat = c_mat.astype('float') / c_mat.sum(axis=1)[:, np.newaxis]
    #c_mat = 100*c_mat
    #c_mat = c_mat.astype('int')

    print('Confusion matrix:')
    print(c_mat)
    print()

    accuracy = 100.0*accuracy_score(gt, pred)
    print('Accuracy:')
    print(accuracy)
    print()
    
    print('='*50)

    stats = {}
    stats['c_mat'] = c_mat
    stats['accuracy'] = accuracy

    return stats

def load_resnet18(finetune, num_classes):    
    #Fix this: loads from internet the first time
    net = torchvision.models.resnet18(pretrained=True)
    
    if finetune=='last':
        for param in net.parameters():
            param.requires_grad = False                
        
        #requires_grad=True by default for new modules
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        param_list = net.fc.parameters()

    if finetune=='all':
        #requires_grad=True by default for new modules
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        param_list = net.parameters()

    if finetune=='none':
        param_list = []
        
    if use_gpu:
        net = net.cuda()

    return [net, param_list]

def load_resnet50(finetune, num_classes):    
    #Fix this: loads from internet the first time
    net = torchvision.models.resnet50(pretrained=True)
    
    if finetune=='last':
        for param in net.parameters():
            param.requires_grad = False                
        
        #requires_grad=True by default for new modules
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        param_list = net.fc.parameters()

    if finetune=='all':
        #requires_grad=True by default for new modules
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        param_list = net.parameters()

    if finetune=='none':
        param_list = []
        
    if use_gpu:
        net = net.cuda()

    return [net, param_list]

def validate_model(net, val_loader, criterion):
    running_loss = 0.0
    num_batches = len(val_loader)

    predictions = []
    gt = []
    
    for data in val_loader:
        inputs, labels = data

        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        inputs = Variable(inputs)
        labels = Variable(labels)

        outputs = net(inputs)
        _, pred = torch.max(outputs.data,1)

        predictions.extend(pred.cpu().numpy().flatten())
        gt.extend(labels.data.cpu().numpy().flatten())
        
        loss = criterion(outputs, labels)
        running_loss += loss.data[0]

    print('Validation Set-- Avg. loss [%.3f]'%((running_loss/num_batches)))
    return [gt, predictions]

def train_model(net, data_loaders, optimizer, criterion, num_epochs, model_file):
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    
    num_batches = len(train_loader)
    
    net.train()

    print('Training started')

    best_metric = 0.0

    for epoch in range(1,num_epochs+1):
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            inputs = Variable(inputs)
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

        print('Epoch [%d] -- Avg. loss [%.3f]'%(epoch,(running_loss/num_batches)))

        net.eval()
        
        [val_gt, val_pred] = validate_model(net, val_loader, criterion)
        stats = compute_stats(val_gt, val_pred)

        if stats['accuracy'] >= best_metric:
            best_metric = stats['accuracy']
            print('Saving model file')
            torch.save(net, model_file)

        
        print('Best model performance: [%.3f]\n'%(best_metric))
        
        net.train()

    return net

def test_model(net, img_path, data_transform):
    inputs = load_image(img_path, data_transform)
    
    if use_gpu:
        inputs = inputs.cuda()

    inputs = Variable(inputs)
    outputs = net(inputs)
    _, pred = torch.max(outputs.data,1)
    pred = pred.cpu().numpy().flatten()
    return pred
