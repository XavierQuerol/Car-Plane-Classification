# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:35:38 2022

@author: xavid
"""

#%% IMPORT DE LLIBRERIES

import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from imutils import paths
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


#%% CREACIÓ CLASSE MODEL

class Net(nn.Module):
    def __init__(self):
        # in_channel / out_channel / kernel_size  --> nn.Conv2d
        
        super(Net, self).__init__()                                
        self.backbone = nn.Sequential(
          nn.Conv2d( 3, 3, 3, padding=2,stride=1), nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d( 3,8, 3, padding=1,stride=1), nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(8,8, 3, padding=1,stride=1), nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(8,16, 3, padding=1,stride=1), nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(16,16, 3, padding=1,stride=1), nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Flatten()
          )
        
        self.car = nn.Sequential(
        nn.Linear(576,20),
        nn.Linear(20,1),
        nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.backbone(x)
        bbox = self.car(features)
        return bbox

summary(Net(),(3,200,200))


#%% CREACIÓ CLASSE DATALOADER

class labelFpsDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        
        #img_dir és una carpeta de directoris
        #guardem en una llista el nom dels arxius de les imatges del directori
        self.img_dir = img_dir
        self.img_paths = []
        for element in img_dir:
            self.img_paths += [el for el in paths.list_images(element)]
        
        #barregem les imatges de la llista
        random.seed(4)
        random.shuffle(self.img_paths)
        
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        
        #agafem la imatge que volem
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
        
        #ajustem la imatge a la mida que volem i definim els valors RGB entre 0 i 1
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2,0,1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        
        #transformem a torch.Tensor
        resizedImage = torch.from_numpy(resizedImage)
        
        if self.is_transform:
            resizedImage = self.is_transform(resizedImage)
            
        if "cars" in img_name:
            label = torch.Tensor([1])
        else:
            label = torch.Tensor([0])
        

        return {'resizedImage': resizedImage, 'label':label, 'img_name':img_name}
    
    
#%% CÀRREGA DE DADES

###Variables globals###
imgSize = (200,200)
carpeta_prova_train = r"C:\Users\xavid\Documents\uni\DADES\2n any\2n sem\PSIV\projecte\multi_dataset\Dataset\train"
carpeta_prova_test = r"C:\Users\xavid\Documents\uni\DADES\2n any\2n sem\PSIV\projecte\multi_dataset\Dataset\test"
train_kwargs = {'batch_size': 16}
test_kwargs  = {'batch_size': 16}

transform = torch.nn.Sequential(T.RandomHorizontalFlip(0.5), T.RandomInvert(0.5))

###creem els databases###
#dst = labelFpsDataLoader([carpeta_prova_train, carpeta_prova_train2], imgSize)
dst = labelFpsDataLoader([carpeta_prova_train], imgSize)
dst_test = labelFpsDataLoader([carpeta_prova_test], imgSize)

###creem els data_loaders###
train_loader = DataLoader(dst, **train_kwargs)
test_loader = DataLoader(dst_test, **test_kwargs)
        
        
#%% CREACIÓ FUNCIÓ TEST i TRAIN

def train(model, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    loss_values = []
    Closs=nn.BCELoss()  
    
    for batch_idx, sample_batched in enumerate(train_loader):

        data = sample_batched['resizedImage']
        target = sample_batched['label']

        output = model(data)
        loss=0.0
        
        loss = Closs(output, target)
        
        loss_values.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        if scheduler is not None:
            scheduler.step()
    
    return loss_values

def test(model, test_loader, name):
    model.eval()
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(test_loader):
            data = sample_batched['resizedImage']
            
            batch_size = data.size(dim=0)
            
            result = model(data)
            
            predicted2[batch_idx*16:batch_idx*16+batch_size] = result  #modificar
            real[batch_idx*16:batch_idx*16+batch_size] = sample_batched['label'][:]
    
    predicted = (predicted2>0.5)
    tp = sum(np.logical_and( (real==predicted), (predicted==1)))
    tn = sum(np.logical_and( (real==predicted), (predicted==0)))
    fp = sum(np.logical_and( (real==0), (predicted==1)))
    fn = sum(np.logical_and( (real==1), (predicted==0)))
    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    f1score = 2 / ((1/recall) + (1/precision))
    accuracy =(tp+tn)/(tp+fp+fn+tn)
            
    #correct = correct/(len(test_loader.dataset)/test_loader.batch_size)
    print(name)
    print('Recall: '+ str(recall))
    print('Precision: '+ str(precision))
    print('F1score: '+ str(f1score))
    print('Accuracy: '+ str(accuracy))


#%% INICIALITZACIÓ MODEL I WEIGHTS

model = Net()

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
    
# Applying it to our net
model.apply(initialize_weights)

for name, param in model.named_parameters():
    print(name)
    if str(name) == "car.1.bias":
        param.data =torch.Tensor([0.5])


#%% TRAINING

from torch.autograd import Variable
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)

epochs = 20
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader))
#scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(test_loader))

log_interval = 20 # how many batches to wait before logging training status

loss_history = []
for epoch in range(1, epochs + 1):
    if epochs%4 == 0:
        optimizer.param_groups[0]["lr"] /= 3
    loss_values= train(model, train_loader, optimizer, epoch, scheduler)
    loss_history += loss_values
    

#%% DESAR WEIGHTS MODEL 

PATH = r"C:\Users\xavid\Documents\uni\DADES\2n any\2n sem\PSIV\projecte\models"
torch.save(model.state_dict(), PATH + "model6dt2.pth")


#%% CARREGAR WEIGHTS MODEL

PATH = "C:/Users/xavid/Documents/uni/DADES/2n any/2n sem/PSIV/projecte/"
model = Net()
model.load_state_dict(torch.load(PATH + "modelsmodel6dt2.pth"))
            

#%% TESTING TEST SET

predicted2 = np.zeros([len(test_loader.dataset),1])
real = np.zeros([len(test_loader.dataset),1])
test(model, test_loader, "test")


#%% TESTING TRAIN SET
predicted2 = np.zeros([len(train_loader.dataset),1])
real = np.zeros([len(train_loader.dataset),1])
test(model, train_loader, "train")
