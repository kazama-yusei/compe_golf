#20210722
#tqdm処理を追加
import os
import sys
from pathlib import Path
from numpy.core.fromnumeric import ndim
from numpy.lib.type_check import imag
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import random
from scipy import ndimage
from torchvision.transforms.transforms import Resize
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from natsort import natsorted #自然数ソート

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter
import preresnet
from torch import nn, optim
import torch
from torchinfo import summary
from torch.autograd import Variable

sys.path.append('/Users/user/Gitclone/pytorch-classification/models/cifar/')
#sys.path.append(str(Path('__file__').resolve().parent.parent))

writer = SummaryWriter(log_dir='../logs')

def seed_set(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

seed_set()

data = pd.read_csv('train_master.tsv', sep='\t')
#print(data.head(10))
#print(data.info())

image_dir = '../train/'
test_dir = '../test/'
image_name_list = data['file_name'].values
label_list = data['flag'].values

x_train, x_val, y_train, y_val = train_test_split(image_name_list, label_list, test_size=0.3, stratify=label_list, random_state=42)

class Resize():
    def __call__(self,image):
        image = resize(image,(32,32))
        return image

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
        #transforms.Resize((32,32)),
        Resize(),
        Normalize(),
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),
        transforms.ToTensor(),
    ]),
    'val':transforms.Compose([
        #transforms.Resize((32,32)),
        Resize(),
        Normalize(),
        transforms.ToTensor(),
    ]),
    'test':transforms.Compose([
        #transforms.Resize((32,32)),
        Resize(),
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

#print(train_dataset.__getitem__(1))

batch_size = 64
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {
    'train':train_dataloader,
    'val':val_dataloader
}


def train_model(net, dataloaders_dict, epochs, loss_fn, optimizer):
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

            with tqdm(dataloaders_dict[phase], unit='batch') as pbar:
                pbar.set_description(f'({phase})Epoch{epoch}')

                for images, labels in pbar:

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

                    pbar.set_postfix(loss=loss.item())
                        
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


def predict(model, data_path, param, batchsize=64, gpu=True, threshold=0.5):

    data = pd.DataFrame()
    data_dir = [filename for filename in os.listdir(data_path) if not filename.startswith('.')]
    data['file_name'] = natsorted(data_dir)
    predictions = np.array([])
    X = []

    model = model.eval()
    model.load_state_dict(torch.load(param, map_location=torch.device('cpu')))#gpuならmap_location不要

    for idx, img in enumerate(data['file_name']):

        
        #image = Variable(image)
        img_path = os.path.join(data_path, img)
        print(img)

        image = io.imread(img_path)
        image = transform['test'](image)
        image = image.float()
        image = image.unsqueeze(0)

        m = nn.Softmax(dim=1)
        pred = (m(model(image)))
        pred = torch.argmax(pred)
        x = pred.detach().numpy().copy()
    
        X.append(int(x))
    x = np.concatenate((predictions,X))
    x = list(map(int,x))#python3系はlist化
    print(x)
    
    data['predictions'] = x
    print(data)

    return data

 
def main(train_flag,predict_flag):
    
    model = preresnet.preresnet(depth=20)
    model.conv1 = nn.Conv2d(7, 16, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    model.fc = nn.Linear(in_features=64, out_features=2, bias=True)
    #print(net)
    #summary(model, (batch_size, 7, 32, 32))
    lr = 1e-4
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=1e-4)

    epochs=20

    if train_flag:
        train_model(net=model, dataloaders_dict=dataloaders_dict, epochs=epochs, loss_fn=loss_fn, optimizer=optimizer)
    elif predict_flag:
        param_path = '../prams/20210722(DA)/Epoch21_iou_0.7796.pth'

        submit = predict(model=model, data_path=test_dir, param=param_path)
        submit[['file_name', 'predictions']].to_csv('submit.tsv', sep='\t', header=None, index=None)

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train model", action="store_true")
    parser.add_argument("--predict", help="predict model", action="store_true")
    args = parser.parse_args()

    main(args.train, args.predict)




                



