
import azureml.core
import os,sys
import numpy as np
import azureml.data
from azureml.data.data_reference import DataReference
from azureml.core import Workspace, Datastore
from azureml.core import Run
import pandas as pd
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.ranking import roc_auc_score
from sklearn.model_selection import train_test_split
from PIL import Image
import json
import argparse
import multiprocessing
import pretrainedmodels


run = Run.get_context()

print(azureml.core.VERSION)
#print(ws.name)
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
args = parser.parse_args()

data_img_dir = args.data_folder
print('Data folder:', data_img_dir)

base_path = os.path.dirname(os.getcwd())
label_file = 'Data_Entry_2017.csv'

assert torch.cuda.is_available()

import pickle
patient_id_partition_file = 'train_test_valid_data_partitions.pickle'

with open(patient_id_partition_file, 'rb') as f:
    [train_set,valid_set,test_set, nih_annotated_set]=pickle.load(f)

print("train:{} valid:{} test:{} nih-annotated:{}".format(len(train_set), len(valid_set), \
                                                     len(test_set), len(nih_annotated_set)))

# Globals
# With small batch may be faster on P100 to do one 1 GPU
MULTI_GPU = True
CLASSES = 14
WIDTH = 331
HEIGHT = 331
CHANNELS = 3
LR = 0.0001
EPOCHS = 10 #100
# Can scale to max for inference but for training LR will be affected
# Prob better to increase this though on P100 since LR is not too low
# Easier to see when plotted
BATCHSIZE = 64 #64*2
IMAGENET_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGENET_RGB_SD = [0.229, 0.224, 0.225]

class XrayData(Dataset):
    def __init__(self, img_dir, lbl_file, patient_ids, transform=None):
        
        # Read labels-csv
        df = pd.read_csv(lbl_file)
        
        df_label = df['Finding Labels'].str.split(
            '|', expand=False).str.join(sep='*').str.get_dummies(sep='*')
        if 'No Finding' in df_label.columns:
            df_label.drop(['No Finding'], axis=1, inplace=True)
        # Filter by patient-ids
        self.labels = df_label.values[df['Patient ID'].isin(patient_ids)]
        df = df[df['Patient ID'].isin(patient_ids)]
        # Split labels
        
                
        # List of images (full-path)
        self.img_locs =  df['Image Index'].map(lambda im: os.path.join(img_dir, im)).values
        # One-hot encoded labels (float32 for BCE loss)
        
        # Processing
        self.transform = transform
        print("Loaded {} labels and {} images".format(len(self.labels), 
                                                      len(self.img_locs)))
    
    def __getitem__(self, idx):
        
        im_file = self.img_locs[idx]
        im_rgb = Image.open(im_file).convert('RGB')
        label = self.labels[idx]
        if self.transform is not None:
            im_rgb = self.transform(im_rgb)
        return im_rgb, torch.FloatTensor(label)
        
    def __len__(self):
        return len(self.img_locs)

def no_augmentation_dataset(img_dir, lbl_file, patient_ids, normalize):
    dataset = XrayData(img_dir, lbl_file, patient_ids,
                       transform=transforms.Compose([
                           #transforms.Resize(331),
                           transforms.Resize((331,331),interpolation=Image.NEAREST),
                           transforms.ToTensor(),  
                           normalize]))
    return dataset

normalize = transforms.Normalize(IMAGENET_RGB_MEAN, IMAGENET_RGB_SD)

# the following transformations will help generalize the model and prevent overfitting.
train_dataset = XrayData(img_dir=data_img_dir,
                         lbl_file=label_file,
                         patient_ids=train_set,
                         transform=transforms.Compose([
                             transforms.Resize(350),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomResizedCrop(size=WIDTH),
                             transforms.ColorJitter(0.15, 0.15),
                             transforms.RandomRotation(15),
                             transforms.ToTensor(),  
                             normalize]))



valid_dataset = no_augmentation_dataset(data_img_dir, label_file, valid_set, normalize)
test_dataset = no_augmentation_dataset(data_img_dir, label_file, test_set, normalize)

model_name = 'nasnetalarge'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

for param in model.parameters():
    param.requires_grad = False

n_classes=14

in_ftrs=model.last_linear.in_features
model.last_linear=nn.Sequential(
nn.Linear(in_ftrs,256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256,n_classes),
nn.Sigmoid())

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model).cuda()
else:
    model.cuda()
import pretrainedmodels.utils as utils

def init_symbol(sym, lr=LR):
    # torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    opt = optim.Adam(sym.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.BCELoss()
    scheduler = ReduceLROnPlateau(opt, factor = 0.1, patience = 5, mode = 'min')
    return opt, criterion, scheduler

def compute_roc_auc(data_gt, data_pd, mean=True, classes=CLASSES):
    roc_auc = []
    data_gt = data_gt.cpu().numpy()
    data_pd = data_pd.cpu().numpy()
    for i in range(classes):
        try:
            roc_auc.append(roc_auc_score(data_gt[:, i], data_pd[:, i]))
        except ValueError:
            pass
    if mean:
        roc_auc = np.mean(roc_auc)
    return roc_auc

def train_epoch(model, dataloader, optimizer, criterion, epoch, batch=BATCHSIZE):
    model.train()
    print("Training epoch {}".format(epoch+1))
    loss_val = 0
    loss_cnt = 0
    batch_count = 0
    for data, target in dataloader:
        # Get samples
        batch_count = batch_count + 1
        #print(batch_count)
        data = torch.FloatTensor(data).cuda()
        target = torch.FloatTensor(target).cuda()
        # Init
        
        optimizer.zero_grad()
        # Forwards
        
        output = model(data)
        
        # Loss
        loss = criterion(output, target)
        
        # Back-prop
        loss.backward()
        optimizer.step()   
         # Log the loss
        loss_val += loss.data.item()
        loss_cnt += 1
    print("Training loss: {0:.4f}".format(loss_val/loss_cnt))
    
@torch.no_grad()
def valid_epoch(model, dataloader, criterion, epoch, phase='valid', batch=BATCHSIZE):
    model.eval()
    if phase == 'testing':
        print("Testing epoch {}".format(epoch+1))
    else:
        print("Validating epoch {}".format(epoch+1))
    out_pred = torch.FloatTensor().cuda()
    out_gt = torch.FloatTensor().cuda()
    loss_val = 0
    loss_cnt = 0
    batch_count = 0
    for data, target in dataloader:
        # Get samples
        batch_count = batch_count + 1
        #print(batch_count)
        data = torch.FloatTensor(data).cuda()
        target = torch.FloatTensor(target).cuda()
         # Forwards
        output = model(data)
        # Loss
        loss = criterion(output, target)
        # Log the loss
        loss_val += loss.data.item()
        loss_cnt += 1
        # Log for AUC
        out_pred = torch.cat((out_pred, output.data), 0)
        out_gt = torch.cat((out_gt, target.data), 0)
    loss_mean = loss_val/loss_cnt
    if phase == 'testing':
        print("Test-Dataset loss: {0:.4f}".format(loss_mean))
        print("Test-Dataset AUC: {0:.4f}".format(compute_roc_auc(out_gt, out_pred)))

    else:
        print("Validation loss: {0:.4f}".format(loss_mean))
        print("Validation AUC: {0:.4f}".format(compute_roc_auc(out_gt, out_pred)))
    return loss_mean

def print_learning_rate(opt):
    for param_group in opt.param_groups:
        print("Learining rate: ", param_group['lr'])

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCHSIZE,
                          shuffle=True, num_workers=0, pin_memory=False)

valid_loader = DataLoader(dataset=valid_dataset, batch_size=2*BATCHSIZE,
                          shuffle=False, num_workers=0, pin_memory=False)

test_loader = DataLoader(dataset=test_dataset, batch_size=2*BATCHSIZE,
                         shuffle=False, num_workers=0, pin_memory=False)


# Load optimiser, loss
optimizer, criterion, scheduler = init_symbol(model)

with torch.no_grad():
    valid_epoch(model, valid_loader, criterion, -1)

loss_min = float("inf")    
stime = time.time()
# Main train/val/test loop
for j in range(EPOCHS):
    train_epoch(model, train_loader, optimizer, criterion, j)
    print("after train")
    with torch.no_grad():
        loss_val = valid_epoch(model, valid_loader, criterion, j)
        test_loss_val = valid_epoch(model, test_loader, criterion, j, 'testing')
    # LR Schedule
    scheduler.step(loss_val)
    print_learning_rate(optimizer)
    
    if loss_val < loss_min:
        print("Loss decreased. Saving ...")
        loss_min = loss_val
        torch.save({'epoch': j + 1, 
                    'state_dict': model.state_dict(), 
                    'best_loss': loss_min, 
                    'optimizer' : optimizer.state_dict()}, 'best_chexray_nasnet.pth.tar')
    etime = time.time()
    print("Epoch time: {0:.0f} seconds".format(etime-stime))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    
run.log('loss', loss_min)



# Load model for testing
azure_chexray_sym_test = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
for param in model.parameters():
    param.requires_grad = False

n_classes=14

in_ftrs=azure_chexray_sym_test.last_linear.in_features
azure_chexray_sym_test.last_linear=nn.Sequential(
nn.Linear(in_ftrs,256),
    nn.ReLU(),
    nn.Dropout(p=0.4),
    nn.Linear(256,n_classes),
nn.Sigmoid())
azure_chexray_sym_test.eval()
optimizer, criterion, scheduler = init_symbol(azure_chexray_sym_test)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs! for testing")
    azure_chexray_sym_test = nn.DataParallel(azure_chexray_sym_test).cuda()
else:
    azure_chexray_sym_test.cuda()
chkpt = torch.load("best_chexray_nasnet.pth.tar")
azure_chexray_sym_test.load_state_dict(chkpt['state_dict'])

with torch.no_grad():
    valid_loss = valid_epoch(azure_chexray_sym_test, valid_loader, criterion, -1)
    test_loss = valid_epoch(azure_chexray_sym_test, test_loader, criterion, -1, 'testing')
