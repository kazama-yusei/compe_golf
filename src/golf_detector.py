import os
import sys
from pathlib import Path
from numpy.core.fromnumeric import ndim
from skimage import io 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import random
from scipy import ndimage
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter
import preresnet
from torch import nn, optim
import torch
from torchinfo import summary

sys.path.append('/Users/user/Gitclone/pytorch-classification/models/cifar/')
sys.path.append(str(Path('__file__').resolve().parent.parent))

writer = SummaryWriter(log_dir='./logs')

data = pd.read_csv('train_master.tsv', sep='\t')
#print(data.head(10))

image_dir = './train/'
image_name_list = data['file_name'].values
label_list = data['flag'].values

x_train, x_val, y_train, y_val = train_test_split(image_name_list, label_list, test_size=0.3, stratify=label_list, random_state=42)

class Normalize():
    def __call__(self,image):
        max = 30000
        min = 5000

        image_normalized = np.clip(image, min, max)
        image_normalized = (image_normalized - min) / (max + min)
        return image_normalized

class HorizontalFlip():
    def __call__(self, image):
        p = random.random()
        if p < 0.5:
            image_transformed = np.fliplr(image).copy()
            return image_transformed
        else:
            return image

class VerticalFlip():
    def __call__(self, image):
        p = random.random()
        if p < 0.5:
            image_transfrmed = np.flipud(image).copy()
            return image_transfrmed
        else:
            return image

class Rotate():
    def __call__(self, image):
        p = random.random()
        if p <= 0.25:
            return image
        elif 0.25 < p <= 0.5:
            image_transformed = ndimage.rotate(image, 90)
            return image_transformed
        elif 0.5 < p <= 0.75:
            image_transformed = ndimage.rotate(image, 180)
            return image_transformed
        else:
            image_transformed = ndimage.rotate(image, 270)
            return image_transformed


        
transform = {
    'train':transforms.Compose([
        Normalize(),
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),
        transforms.ToTensor(),
    ]),
    'val':transforms.Compose([
        Normalize(),
        transforms.ToTensor(),
    ])
}

class Satellite(Dataset):
    def __init__(self, image_name_list, label_list, phase=None):
        self.image_name_list = image_name_list
        self.label_list = label_list
        self.phase = phase

    def __len__(self,):
        return len(self.image_name_list)
    
    def __getitem__(self, index):
        image_name = self.image_name_list[index]
        image = io.imread(image_dir + image_name)
        image = transform[self.phase](image)

        label = self.label_list[index]

        return image, label

train_dataset = Satellite(x_train, y_train, phase='train')
val_dataset = Satellite(x_val, y_val, phase='val')
batch_size = 64
lr = 0.1
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {
    'train':train_dataloader,
    'val':val_dataloader 
}

def train_model(net, epochs, loss_fn, optimizer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    net.to(device)
    best_iou = 0.0

    for epoch in range(1, epochs+1):
        print('-'*40)
        print(f'Epoch: {epoch} / {epochs}')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            
            epoch_loss = 0.0
            pred_list = []
            true_list = []

            for images, labels in dataloaders_dict[phase]:

                images = images.float().to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = net(images)
                    loss = loss_fn(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * images.size(0)

                    preds = preds.to('cpu').numpy()
                    pred_list.extend(preds)

                    labels = labels.to('cpu').numpy()
                    true_list.extend(labels)
            
            #1epoch内の損失値の平均
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)

            TN,FP,FN,TP = confusion_matrix(true_list, pred_list).flatten()
            epoch_iou =  TP / (TP+FP+FN)
            print(f'{phase} Loss: {epoch_loss:.4f} IOU: {epoch_iou:.4f}')

            writer.add_scalar(f'{phase} Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}:IoU', epoch_iou, epoch+1)
            

            if (phase == 'val') and (epoch_iou > best_iou):
                best_iou = epoch_iou
                param_name = f'Epoch{epoch+1}_iou_{epoch_iou:.4f}.pth'
                torch.save(net.state_dict(), param_name)
        writer.close()

if __name__=='__main__':
    net = preresnet.preresnet(depth=20)
    net.conv1 = nn.Conv2d(7, 16, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    net.fc = nn.Linear(in_features=64, out_features=2, bias=True)
    #print(net)
    summary(net, (batch_size, 7, 32, 32))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    epochs=20
    train_model(net=net, epochs=epochs, loss_fn=loss_fn, optimizer=optimizer)




                



